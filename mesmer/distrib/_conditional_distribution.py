# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

from typing import Literal

import numpy as np
import xarray as xr

from mesmer._core.utils import (
    _check_dataarray_form,
    _check_dataset_form,
    _ignore_warnings,
)
from mesmer.distrib import _distrib_checks, _optimizers
from mesmer.distrib._expression import Expression
from mesmer.distrib._first_guess import _FirstGuess
from mesmer.distrib._optimizers import MinimizeOptions, OptimizerNLL
from mesmer.geospatial import geodist_exact
from mesmer.stats import gaspari_cohn
from mesmer.weighted import _weighted_median


class ConditionalDistribution:
    def __init__(
        self,
        expression: Expression,
        *,
        minimize_options: MinimizeOptions = MinimizeOptions(),
        second_minimizer: MinimizeOptions | None = None,
        threshold_min_proba=1.0e-9,
    ):
        """
        A conditional distribution.

        Parameters
        ----------
        Expression : class py:class:`Expression`
            Expression defining the conditional distribution.
        minimize_options : `py:class:`MinimizeOptions`, default: MinimizeOptions()
            Class defining the optimizer options used during first guess and training of
            distributions. If not passed uses "Nelder-Mead" minimizer with default
            settings.
        second_minimizer : class py:class:`MinimizeOptions` | None, default: None
            Run a second minimization algorithm for all steps. ``method="Powell"``
            is recommended. It can be beneficial to run more than one minimization
            to get a more stable estimate.

        """
        # initialization

        if not isinstance(expression, Expression):
            raise TypeError("'expression' must be an `Expression`")

        if second_minimizer is not None:
            if minimize_options.method == second_minimizer.method:
                raise ValueError("First and second minimizer have the same method")

        self.expression = expression
        self.minimize_options = minimize_options
        self.second_minimizer = second_minimizer

        # keep optimizer as attr (https://github.com/MESMER-group/mesmer/issues/729)
        self._optimizer = OptimizerNLL()
        self._coefficients = None

        self.threshold_min_proba = threshold_min_proba
        if threshold_min_proba is not None and (
            (threshold_min_proba <= 0) or (0.5 <= threshold_min_proba)
        ):
            raise ValueError("`threshold_min_proba` must be in [0, 0.5]")

    def fit(
        self,
        predictors: dict[str, xr.DataArray] | xr.Dataset,
        target: xr.DataArray,
        weights: xr.DataArray,
        first_guess: xr.Dataset,
        *,
        sample_dim: str = "sample",
        smooth_coeffs: bool = False,
        r_gasparicohn: float = 500,
        on_failed_fit: Literal["error", "ignore"] = "error",
    ):
        """fit conditional distribution over all gridpoints.

        Parameters
        ----------
        predictors : dict of xr.DataArray | xr.Dataset
            A dict of DataArray objects used as predictors or a Dataset, holding each
            predictor as a data variable. Each predictor must be 1D and contain
            `sample_dim`.
        target : xr.DataArray
            Target DataArray.
        weights : xr.DataArray.
            Individual weights for each sample.
        first_guess : xr.Dataset
            First guess for the coefficients, each coefficient is a data variable in the
            Dataset and has the corresponding name of the coefficient in the expression.
        sample_dim : str
            Dimension along which to fit the distribution.
        smooth_coeffs : bool, default: False
            If True, smooth the provided coefficients using a weighted median.
            The weights are the correlation matrix of the Gaspari-Cohn function.
            This is typically used for the 2nd round of the fit.
        r_gasparicohn : float, default: 500
            Radius used to compute the correlation matrix of the Gaspari-Cohn function.
            Used if ``smooth_coeffs`` is True.
        on_failed_fit : "error" | "ignore", default: "error"
            Behaviour when the fit fails. Careful: currently the using "ignore" returns
            the first guess.

        Returns
        -------
        :obj:`xr.Dataset`
            Fitted coefficients of the conditional distribution (gridpoint, coefficient)
        """

        # basic check on first guess
        n_coeffs_fg = len(first_guess.data_vars)
        if n_coeffs_fg != self.expression.n_coeffs:
            raise ValueError(
                "The provided first guess does not have the correct number of coeffs,"
                f" Expression suggests a number of {self.expression.n_coeffs}"
                f" coefficients, but `first_guess` has {n_coeffs_fg} data_variables."
            )

        first_guess_da = first_guess.to_dataarray(dim="coefficient")

        predictors_da = _concatenate_predictors(predictors)
        self.predictor_names = predictors_da.coords["predictor"].values.tolist()

        _check_dataarray_form(target, "target", required_dims=[sample_dim])
        _check_dataarray_form(weights, "weights", required_dims=[sample_dim])

        _distrib_checks._check_no_nan_no_inf(predictors_da, "predictor data")
        _distrib_checks._check_no_nan_no_inf(target, "target data")
        _distrib_checks._check_no_nan_no_inf(weights, "weights")

        # checking for smoothing of coefficients, eg for 2nd round of fit
        if smooth_coeffs:
            gridcell_dim = (set(target.dims) - {sample_dim}).pop()
            coords = target[gridcell_dim].coords
            # preserve non-smoothed first guess in case it causes the fit to fails
            first_guess_initial = first_guess_da.copy()
            first_guess_da = _smooth_first_guess(
                first_guess_da, gridcell_dim, coords, r_gasparicohn
            )
        else:
            first_guess_initial = first_guess_da

        # training
        result = xr.apply_ufunc(
            self._fit_np,
            predictors_da,
            target,
            weights,
            first_guess_da,
            first_guess_initial,
            input_core_dims=[
                [sample_dim, "predictor"],
                [sample_dim],
                [sample_dim],
                ["coefficient"],
                ["coefficient"],
            ],
            output_core_dims=[["coefficient"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={"on_failed_fit": on_failed_fit},
        )

        # creating a dataset with the coefficients
        coefficients = xr.Dataset()
        for i, coef in enumerate(self.expression.coefficients_list):
            coefficients[coef] = result.isel(coefficient=i)

        coefficients = coefficients.drop_vars("coefficient")

        # add expression as attribute
        coefficients = coefficients.assign_attrs(
            {
                "expression_name": self.expression.expression_name,
                "expression": self.expression.expression,
            }
        )

        self._coefficients = coefficients

    # suppress nan & inf warnings; TODO: don't suppress all warnings
    @_ignore_warnings()
    def _fit_np(
        self, data_pred, data_targ, data_weights, fg, fg_initial, on_failed_fit
    ):
        """
        Fit the coefficients of the conditional distribution by minimizing _func_optim.
        """

        # correcting format: must be dict(str, DataArray or array) for Expression
        # NOTE: extremely important that the order is the right one
        data_pred = {key: data_pred[:, i] for i, key in enumerate(self.predictor_names)}

        # initialize optimizer function
        func = _optimizers._optimization_function(
            optimizer=self._optimizer,
            data_pred=data_pred,
            data_targ=data_targ,
            data_weights=data_weights,
            expression=self.expression,
            threshold_min_proba=self.threshold_min_proba,
        )

        # training
        m = _optimizers._minimize(
            func=func,
            x0=fg,
            args=(),
            minimize_options=self.minimize_options,
            second_minimizer=self.second_minimizer,
        )

        # checking if the fit has failed
        if not m.success:
            if on_failed_fit == "error":
                raise ValueError("Failed fit")

            # NOTE: warnings are hidden by apply_ufunc

        # returning best value between failsafe of the first guess and the result fitted here
        if func(m.x) < func(fg_initial):
            return m.x
        else:
            return fg_initial

    def find_first_guess(
        self,
        predictors: dict[str, xr.DataArray] | xr.Dataset,
        target: xr.DataArray,
        weights: xr.DataArray,
        first_guess: xr.Dataset | None = None,
        *,
        sample_dim: str = "sample",
    ):
        """
        Find a first guess for all coefficients of a conditional distribution

        Parameters
        ----------
        conditional_distrib : ConditionalDistribution
            Conditional distribution object to find the first guess for.
        predictors : dict of xr.DataArray | xr.Dataset
            A dict of DataArray objects used as predictors or a Dataset, holding each
            predictor as a data variable. Each predictor must be 1D and contain
            `sample_dim`.
        target : xr.DataArray
            Target DataArray, contains at least `sample_dim`.
        weights : xr.DataArray.
            Individual weights for each sample, must be 1D along `sample_dim`.
        first_guess : xr.Dataset, default: None
            If provided, will use these values as first guess for the first guess. If
            None, will use all zeros. Must contain the first guess for each coefficient
            in a DataArray with the name of the coefficient.
        sample_dim : str
            Dimension along which to fit the first guess.

        Returns
        -------
        :obj:`xr.Dataset`
            Dataset of first guess for each coefficient of the conditional distribution
            as a data variable with the name of the coefficient.
        """

        # make fg with zeros if none
        if first_guess is None:
            first_guess = xr.Dataset()
            fg_dims = set(target.dims) - {sample_dim}
            fg_size = [target.sizes[dim] for dim in fg_dims]
            for coef in self.expression.coefficients_list:
                first_guess[coef] = xr.DataArray(np.zeros(fg_size), dims=fg_dims)
        first_guess_da = first_guess.to_dataarray(dim="coefficient")

        # preparing data
        predictors_da = _concatenate_predictors(predictors)
        predictor_names = predictors_da.coords["predictor"].values.tolist()

        _check_dataarray_form(target, "target", required_dims=[sample_dim])
        _check_dataarray_form(weights, "weights", required_dims=[sample_dim])

        _distrib_checks._check_no_nan_no_inf(predictors_da, "predictor data")
        _distrib_checks._check_no_nan_no_inf(target, "target data")
        _distrib_checks._check_no_nan_no_inf(weights, "weights")

        # search for each gridpoint
        result = xr.apply_ufunc(
            self._find_fg_np,
            predictors_da,
            target,
            weights,
            first_guess_da,
            input_core_dims=[
                [sample_dim, "predictor"],
                [sample_dim],
                [sample_dim],
                ["coefficient"],
            ],
            output_core_dims=[["coefficient"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={
                "predictor_names": predictor_names,
            },
        )

        # creating a dataset with the coefficients
        out = xr.Dataset()
        for i, coef in enumerate(self.expression.coefficients_list):
            out[coef] = result.isel(coefficient=i)
        return out.drop_vars("coefficient")

    def _find_fg_np(
        self,
        data_pred,
        data_targ,
        data_weights,
        first_guess,
        predictor_names,
    ):
        """
        Setup instance of _FirstGuess and fit a first guess for the coefficients.
        """

        fg = _FirstGuess(
            expression=self.expression,
            minimize_options=self.minimize_options,
            data_pred=data_pred,
            predictor_names=predictor_names,
            data_targ=data_targ,
            data_weights=data_weights,
            first_guess=first_guess,
            threshold_min_proba=self.threshold_min_proba,
            second_minimizer=self.second_minimizer,
        )

        return fg._find_fg()

    def compute_quality_scores(
        self,
        predictors: dict[str, xr.DataArray] | xr.Dataset,
        target: xr.DataArray,
        weights: xr.DataArray,
        *,
        sample_dim: str = "sample",
        scores=["func_optim", "nll", "bic"],
    ):
        """Compute scores for fit coefficients.

        Parameters
        ----------
        predictors : dict of xr.DataArray | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor as a data variable. Each predictor must be 1D and contain
            `sample_dim`.
        target : xr.DataArray
            Target DataArray.
        sample_dim : str
            Dimension along which to calculate the scores.
        weights : xr.DataArray.
            Individual weights for each sample.
        scores : list of str, default: ['func_optim', 'nll', 'bic']
            After the fit, several scores can be calculated to assess the performance:

            - "func_optim": function optimized, as described in
              options_optim['type_fun_optim']: negative log likelihood or full
              conditional negative log likelihood
            - "nll": Negative Log Likelihood
            - "bic": Bayesian Information Criteria
            - "crps": Continuous Ranked Probability Score (warning: takes a long time to
              compute)

        Returns
        -------
        scores: xr.Dataset
            Dataset containing the scores for each gridpoint.

        Notes
        -----
        "nll" may or may not be the same as "func_optim"
        """

        da_coeffs = self.coefficients.to_dataarray(dim="coefficient")

        data_pred = _concatenate_predictors(predictors)
        self.predictor_names = data_pred.predictor.values.tolist()

        _check_dataarray_form(target, "target", required_dims=[sample_dim])
        _check_dataarray_form(weights, "weights", required_dims=[sample_dim])

        _distrib_checks._check_no_nan_no_inf(data_pred, "predictor data")
        _distrib_checks._check_no_nan_no_inf(target, "target data")
        _distrib_checks._check_no_nan_no_inf(weights, "weights")

        n_scores = len(scores)

        # compute for each gridpoint
        result = xr.apply_ufunc(
            self._compute_quality_scores_np,
            da_coeffs,
            data_pred,
            target,
            weights,
            input_core_dims=[
                ["coefficient"],
                [sample_dim, "predictor"],
                [sample_dim],
                [sample_dim],
            ],
            output_core_dims=(set(),) * n_scores,
            kwargs={"scores": scores},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float] * n_scores,
        )

        data_vars = {key: score for key, score in zip(scores, result)}

        return xr.Dataset(data_vars)

    def _compute_quality_scores_np(
        self, coefficients, data_pred, data_targ, data_weights, scores
    ):
        """Compute quality scores for the fit coefficients."""

        quality_scores = []

        # correcting format: must be dict(str, DataArray or array) for Expression
        data_pred = {key: data_pred[:, i] for i, key in enumerate(self.predictor_names)}

        # need to loop to ensure correct order
        for score in scores:

            # basic result: optimized value
            if score == "func_optim":

                func = _optimizers._optimization_function(
                    optimizer=self._optimizer,
                    data_pred=data_pred,
                    data_targ=data_targ,
                    data_weights=data_weights,
                    expression=self.expression,
                    threshold_min_proba=self.threshold_min_proba,
                )
                res = func(coefficients)
                quality_scores.append(res)

            # calculating parameters for the next ones
            params = self.expression.evaluate_params(coefficients, data_pred)

            # NLL averaged over sample
            if score == "nll":
                res = _optimizers._neg_loglike(
                    self.expression, data_targ, params, data_weights
                )
                quality_scores.append(res)

            # BIC averaged over sample
            if score == "bic":
                res = _optimizers._bic(self.expression, data_targ, params, data_weights)
                quality_scores.append(res)

            # CRPS
            if score == "crps":
                res = _optimizers._crps(
                    self.expression,
                    data_targ,
                    data_pred,
                    data_weights,
                    coefficients,
                )

                quality_scores.append(res)

        return tuple(quality_scores)

    @property
    def coefficients(self):
        """The coefficients of this conditional distribution."""

        if self._coefficients is None:
            raise ValueError(
                "'coefficients' not set - call `fit` or assign them to "
                "`ConditionalDistribution().coefficients`."
            )

        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        """The coefficients of this conditional distribution."""

        required_vars = set(self.expression.coefficients_list)
        _check_dataset_form(
            coefficients,
            "coefficients",
            required_vars=required_vars,
            # optional_vars="weights",
            requires_other_vars=False,
        )

        self._coefficients = coefficients

    @classmethod
    def from_netcdf(cls, filename: str, **kwargs):
        """read coefficients from a netCDF file with default solver options

        Parameters
        ----------
        filename : str
            Name of the netCDF file to open.
        **kwargs : Any
            Additional keyword arguments passed to ``xr.open_dataset``
        """
        ds = xr.open_dataset(filename, **kwargs)

        return cls.from_dataset(ds)

    @classmethod
    def from_dataset(cls, ds):
        """set coefficients from a dataset with default solver options

        Parameters
        ----------
        ds : Dataset
            Dataset which was previously fit using this class.
        """

        expression_str = ds.attrs.get("expression", None)
        if expression_str is None:
            raise ValueError("The 'expression' attribute is missing")

        expression_name = ds.attrs.get("expression_name", None)
        if expression_name is None:
            raise ValueError("The 'expression_name' attribute is missing")

        expression = Expression(expression_str, expression_name)
        obj = cls(expression)
        obj.coefficients = ds

        return obj

    def to_netcdf(self, filename: str, **kwargs):
        """save coefficients dataset to a netCDF file

        Parameters
        ----------
        filename : str
            Name of the netCDF file to save.
        **kwargs : Any
            Additional keyword arguments passed to ``xr.Dataset.to_netcdf``
        """

        self.coefficients.to_netcdf(filename, **kwargs)


def _smooth_first_guess(
    first_guess_stacked: xr.Dataset, dim, grid_coords, r_gasparicohn
):
    """
    smooth first guess over

    Parameters
    ----------
    first_guess : xr.DataArray
        First guess for the coefficients. Stacked over the coefficients.
    dim : str
        Dimension along which to smooth the coefficients.
    grid_coords : dict
        Coordinates of the grid points, used to compute the distance between points.
        Must contain the coordinates of the grid points in the form of a dictionary
        with keys being the coordinate names and values being 1D arrays of coordinates.
    r_gasparicohn : float
        Radius used to compute the correlation matrix of the Gaspari-Cohn function.

    Returns
    -------
    first_guess : :obj:`xr.Dataset`
        Smoothed ``first_guess``
    """
    # calculating distance between points
    geodist = geodist_exact(**grid_coords)
    # deducing correlation matrix
    corr_gc = gaspari_cohn(geodist / r_gasparicohn)

    # will avoid taking gridpoints with nan
    second_guess = xr.apply_ufunc(
        _weighted_median,
        first_guess_stacked,
        corr_gc,
        input_core_dims=[[dim], [f"{dim}_i"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # we loop over dim_j -> have to replace the dim and cooords
    second_guess = second_guess.rename({f"{dim}_j": dim})
    second_guess = second_guess.assign_coords({dim: first_guess_stacked[dim]})

    return second_guess


def _concatenate_predictors(
    predictors: dict[str, xr.DataArray] | xr.Dataset,
) -> xr.DataArray:
    """
    If predictors is a dict, put the concat the values in a xr.DataArray along
    a "predictor" dimension, with the keys as coordinates. If predictors is a
    DataArray (with predictors as data variables), stack the data variables
    along a "predictor" dimension with the names of the variables as coordinates.
    """
    if not isinstance(predictors, dict | xr.Dataset):
        msg = (
            "predictors is supposed to be a dict of xr.DataArray or a xr.Dataset"
            f" got '{type(predictors)}'"
        )

        raise TypeError(msg)

    predictors_concat = xr.concat(
        tuple(predictors.values()),
        dim="predictor",
        join="exact",
        coords="minimal",
    )
    predictors_concat = predictors_concat.assign_coords(
        {"predictor": list(predictors.keys())}
    )

    return predictors_concat
