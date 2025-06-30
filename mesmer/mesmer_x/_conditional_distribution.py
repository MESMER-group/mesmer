# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr

from mesmer.core.geospatial import geodist_exact
from mesmer.core.utils import _check_dataarray_form, _check_dataset_form
from mesmer.core.weighted import weighted_median
from mesmer.mesmer_x import _distrib_checks, _optimizers
from mesmer.mesmer_x._expression import Expression
from mesmer.mesmer_x._first_guess import _FirstGuess
from mesmer.mesmer_x._optimizers import OptimizerFCNLL, OptimizerNLL
from mesmer.mesmer_x._utils import _ignore_warnings
from mesmer.stats import gaspari_cohn


class ConditionalDistributionOptions:
    def __init__(
        self,
        threshold_min_proba=1.0e-9,
        options_optim=None,  # removed
        options_solver=None,
    ):
        """Class to define optimizers used during first guess and training.

        Parameters
        ----------
        threshold_min_proba :  float or None, default: 1e-9
            If numeric imposes a check during the fitting that every sample fulfills
            `cdf(sample) >= threshold_min_proba and 1 - cdf(sample) >= threshold_min_proba`,
            i.e. each sample lies within some confidence interval of the distribution.
            Note that it follows that threshold_min_proba math::\\in (0,0.5). Important to
            ensure that all points are feasible with the fitted distribution.
            If `None` this test is skipped.

        options_solver : dict, optional
            A dictionary with options for the solvers, used to determine an adequate
            first guess and for the final optimization.

            * method_fit: string, default: "Powell"
                Type of algorithm used during the optimization, using the function
                'minimize'. Prepared options: BFGS, L-BFGS-B, Nelder-Mead, Powell, TNC,
                trust-constr. The default 'Powell', is HIGHLY RECOMMENDED for its
                stability & speed.

            * xtol_req: float, default: 1e-3
                Accuracy of the fit in coefficients. Interpreted differently depending
                on 'method_fit'.

            * ftol_req: float, default: 1e-6
                Accuracy of the fit in objective.

            * maxiter: int, default: None
                Maximum number of iteration of the optimization. Uses the default of the
                choosen minimizer.

            * maxfev: int, default: None
                Maximum number of evaluation of the function during the optimization.
                Uses the default of the choosen minimizer.

            * error_failedfit : boolean, default: True.
                If True, will raise an issue if the fit failed.


        """

        if options_optim is not None:

            msg = (
                "`options_optim` was removed - pass `optimizer` directly to "
                "`ConditionalDistribution`"
            )

            raise ValueError(msg)

        # preparing solver
        default_options_solver = {
            "method_fit": "Powell",
            "xtol_req": 1e-6,
            "ftol_req": 1.0e-6,
            "maxiter": None,
            "maxfev": None,
            "error_failedfit": False,
        }

        options_solver = options_solver or {}
        if not isinstance(options_solver, dict):
            raise ValueError("`options_solver` must be a dictionary")

        # TODO: use get? (e.g. self.method_fit = options_solver.get("method_fit", "Powell")
        options_solver = default_options_solver | options_solver

        self.xtol_req = options_solver["xtol_req"]
        self.ftol_req = options_solver["ftol_req"]
        self.maxiter = options_solver["maxiter"]
        self.maxfev = options_solver["maxfev"]
        self.method_fit = options_solver["method_fit"]

        if self.method_fit not in (
            "BFGS",
            "L-BFGS-B",
            "Nelder-Mead",
            "Powell",
            "TNC",
            "trust-constr",
        ):
            raise ValueError("method for this fit not prepared, to avoid")

        xtol = {
            "BFGS": "xrtol",
            "L-BFGS-B": "gtol",
            "Nelder-Mead": "xatol",
            "Powell": "xtol",
            "TNC": "xtol",
            "trust-constr": "xtol",
        }
        ftol = {
            "BFGS": "gtol",
            "L-BFGS-B": "ftol",
            "Nelder-Mead": "fatol",
            "Powell": "ftol",
            "TNC": "ftol",
            "trust-constr": "gtol",
        }

        self.name_xtol = xtol[self.method_fit]
        self.name_ftol = ftol[self.method_fit]
        self.error_failedfit = options_solver["error_failedfit"]

        # initialization and basic checks on threshold_min_proba
        self.threshold_min_proba = threshold_min_proba
        if threshold_min_proba is not None and (
            (threshold_min_proba <= 0) or (0.5 <= threshold_min_proba)
        ):
            raise ValueError("`threshold_min_proba` must be in [0, 0.5]")


class ConditionalDistribution:
    def __init__(
        self,
        expression: Expression,
        options: ConditionalDistributionOptions,
        optimizer: OptimizerNLL | OptimizerFCNLL = OptimizerNLL(),
    ):
        """
        A conditional distribution.

        Parameters
        ----------
        Expression : class py:class:Expression
            Expression defining the conditional distribution.
        options : class py:class:ConditionalDistributionOptions
            Class defining the optimizer options used during first guess and training of
            distributions.
        optimizer : OptimizerNLL | OptimizerFCNLL, default: OptimizerNLL
            Optimizer to use.

            * OptimizerNLL: negative log likelihood (default)
            * OptimizerFCNLL: full conditional negative log likelihood based on the
              stopping rule

        """
        # initialization
        self.expression = expression
        self.options = options
        self.optimizer = optimizer
        self._coefficients = None

    def fit(
        self,
        predictors: dict[str, xr.DataArray] | xr.Dataset,
        target: xr.DataArray,
        weights: xr.DataArray,
        first_guess: xr.Dataset,
        sample_dim: str = "sample",
        smooth_coeffs: bool = False,
        r_gasparicohn: float = 500,
        option_smooth_coeffs: None = None,  # deprecated in favor of smooth_coeffs
    ):
        """fit conditional distribution over all gridpoints.

        Parameters
        ----------
        predictors : dict of xr.DataArray | xr.Dataset
            A dict of DataArray objects used as predictors or a Dataset, holding each
            predictor as a data variable. Each predictor must be 1D and contain `sample_dim`.
        target : xr.DataArray
            Target DataArray.
        weights : xr.DataArray.
            Individual weights for each sample.
        first_guess : xr.Dataset
            First guess for the coefficients, each coefficient is a data variable in the Dataset
            and has the corresponding name of the coefficient in the expression.
        sample_dim : str
            Dimension along which to fit the distribution.
        option_smooth_coeffs : bool, default: False
            If True, smooth the provided coefficients using a weighted median.
            The weights are the correlation matrix of the Gaspari-Cohn function.
            This is typically used for the 2nd round of the fit.
        r_gasparicohn : float, default: 500
            Radius used to compute the correlation matrix of the Gaspari-Cohn function.
            This is typically used for the 2nd round of the fit.

        Returns
        -------
        :obj:`xr.Dataset`
            Fitted coefficients of the conditional distribution (gridpoint, coefficient)
        """

        if option_smooth_coeffs is not None:
            raise ValueError("option_smooth_coeffs has been renamed to smooth_coeffs")

        # basic check on first guess
        n_coeffs_fg = len(first_guess.data_vars)
        if n_coeffs_fg != self.expression.n_coeffs:
            raise ValueError(
                "The provided first guess does not have the correct number of coeffs, "
                f"Expression suggests a number of {self.expression.n_coeffs} coefficients, "
                f"but `first_guess` has {n_coeffs_fg} data_variables."
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
            first_guess = _smooth_first_guess(
                first_guess, gridcell_dim, coords, r_gasparicohn
            )

        # training
        result = xr.apply_ufunc(
            self._fit_np,
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

    @_ignore_warnings  # suppress nan & inf warnings
    def _fit_np(self, data_pred, data_targ, data_weights, fg):
        """
        Fit the coefficients of the conditional distribution by minimizing _func_optim.
        """
        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        # NOTE: extremely important that the order is the right one
        data_pred = {key: data_pred[:, i] for i, key in enumerate(self.predictor_names)}

        # initialize optimizer function
        func = _optimizers._optimization_function(
            optimizer=self.optimizer,
            data_pred=data_pred,
            data_targ=data_targ,
            data_weights=data_weights,
            expression=self.expression,
            threshold_min_proba=self.options.threshold_min_proba,
        )

        # training
        m = _optimizers._minimize(
            func=func,
            x0=fg,
            method_fit=self.options.method_fit,
            args=(),
            option_NelderMead="best_run",
            options={
                "maxfev": self.options.maxfev,
                "maxiter": self.options.maxiter,
                self.options.name_xtol: self.options.xtol_req,
                self.options.name_ftol: self.options.ftol_req,
            },
        )

        # checking if the fit has failed
        if self.options.error_failedfit and not m.success:
            raise ValueError("Failed fit.")
        else:
            return m.x

    def find_first_guess(
        self,
        predictors: dict[str, xr.DataArray] | xr.Dataset,
        target: xr.DataArray,
        weights: xr.DataArray,
        first_guess: xr.Dataset | None = None,
        sample_dim: str = "sample",
    ):
        """
        Find a first guess for all coefficients of a conditional distribution for each grid point.

        Parameters
        ----------
        conditional_distrib : ConditionalDistribution
            Conditional distribution object to find the first guess for.
        predictors : dict of xr.DataArray | xr.Dataset
            A dict of DataArray objects used as predictors or a Dataset, holding each
            predictor as a data variable. Each predictor must be 1D and contain `sample_dim`.
        target : xr.DataArray
            Target DataArray, contains at least `sample_dim`.
        weights : xr.DataArray.
            Individual weights for each sample, must be 1D along `sample_dim`.
        first_guess : xr.Dataset, default: None
            If provided, will use these values as first guess for the first guess. If None,
            will use all zeros. Must contain the first guess for each coefficient in a
            DataArray with the name of the coefficient.
        sample_dim : str
            Dimension along which to fit the first guess.

        Returns
        -------
        :obj:`xr.Dataset`
            Dataset of first guess for each coefficient of the conditional distribution as a
            data variable with the name of the coefficient.
        """
        # TODO: some smoothing on first guess? cf 2nd fit with MESMER-X given results.

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
            kwargs={
                "predictor_names": predictor_names,
            },
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
            self.expression,
            self.options,
            data_pred,
            predictor_names,
            data_targ,
            data_weights,
            first_guess,
        )

        return fg._find_fg()

    def compute_quality_scores(
        self,
        predictors: dict[str, xr.DataArray] | xr.Dataset,
        target: xr.DataArray,
        weights: xr.DataArray,
        sample_dim: str = "sample",
        scores=["func_optim", "nll", "bic"],
    ):
        """Compute scores for fit coefficients.

        Parameters
        ----------
        predictors : dict of xr.DataArray | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor as a data variable. Each predictor must be 1D and contain `sample_dim`.
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
            output_core_dims=[["score"]],
            kwargs={"scores": scores},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        result["score"] = scores
        return xr.Dataset({"scores": result})

    def _compute_quality_scores_np(
        self, coefficients, data_pred, data_targ, data_weights, scores
    ):
        """Compute quality scores for the fit coefficients."""

        quality_scores = []

        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        data_pred = {key: data_pred[:, i] for i, key in enumerate(self.predictor_names)}

        # basic result: optimized value
        if "func_optim" in scores:

            func = _optimizers._optimization_function(
                optimizer=self.optimizer,
                data_pred=data_pred,
                data_targ=data_targ,
                data_weights=data_weights,
                expression=self.expression,
                threshold_min_proba=self.options.threshold_min_proba,
            )
            scores = func(coefficients)
            quality_scores.append(scores)

        # calculating parameters for the next ones
        params = self.expression.evaluate_params(coefficients, data_pred)

        # NLL averaged over sample
        if "nll" in scores:
            score = _optimizers._neg_loglike(
                self.expression, data_targ, params, data_weights
            )
            quality_scores.append(score)

        # BIC averaged over sample
        if "bic" in scores:
            scores = _optimizers._bic(self.expression, data_targ, params, data_weights)
            quality_scores.append(score)

        # CRPS
        if "crps" in scores:
            score = _optimizers._crps(
                self.expression,
                data_targ,
                data_pred,
                data_weights,
                coefficients,
            )

            quality_scores.append(score)
        return np.array(quality_scores)

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
        """read coefficients from a netCDF file with default options

        Parameters
        ----------
            Name of the netCDF file to open.
        **kwargs : Any
            Additional keyword arguments passed to ``xr.open_dataset``
        """
        ds = xr.open_dataset(filename, **kwargs)
        expression_str = ds.attrs.get("expression", None)
        if expression_str is None:
            # NOTE: make this a warning?
            raise ValueError(
                "The netCDF file does not contain the 'expression' attribute."
            )

        expression_name = ds.attrs.get("expression_name", None)
        if expression_name is None:
            raise ValueError(
                "The netCDF file does not contain the 'expression_name' attribute."
            )

        expression = Expression(expression_str, expression_name)
        obj = cls(expression, ConditionalDistributionOptions())
        obj.coefficients = ds

        return obj

    def to_netcdf(self, filename: str, **kwargs):
        """save coefficients dataset to a netCDF file

        Parameters
        ----------
        filename : str
            Name of the netCDF file to save.
        **kwargs : Any
            Additional keyword arguments passed to ``xr.Dataset.to_netcf``
        """

        coefficients = self.coefficients
        coefficients.to_netcdf(filename, **kwargs)


def _smooth_first_guess(first_guess: xr.Dataset, dim, grid_coords, r_gasparicohn):
    """
    smooth first guess over

    Parameters
    ----------
    first_guess : xr.Dataset
        First guess for the coefficients.
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
    fg_coeffs = first_guess.to_array("coeff")

    second_guess_stacked = xr.apply_ufunc(
        weighted_median,
        fg_coeffs,
        corr_gc,
        input_core_dims=[[dim], [f"{dim}_i"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    second_guess = xr.full_like(first_guess, fill_value=np.nan)
    for coeff in first_guess.data_vars:
        second_guess[coeff] = second_guess_stacked.sel(coeff=coeff)
    second_guess = second_guess.rename({f"{dim}_j": dim})
    second_guess = second_guess.drop_vars("coeff")

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
    if isinstance(predictors, dict | xr.Dataset):
        predictors_concat = xr.concat(
            tuple(predictors.values()),
            dim="predictor",
            join="exact",
            coords="minimal",
        )
        predictors_concat = predictors_concat.assign_coords(
            {"predictor": list(predictors.keys())}
        )

    else:
        raise TypeError(
            "predictors is supposed to be a dict of xr.DataArray or a xr.Dataset"
        )
    return predictors_concat
