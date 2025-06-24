# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr

from mesmer.core.geospatial import geodist_exact
from mesmer.core.utils import _check_dataset_form
from mesmer.core.weighted import weighted_median
from mesmer.mesmer_x import _distrib_checks, _optimizers
from mesmer.mesmer_x._expression import Expression
from mesmer.mesmer_x._first_guess import _FirstGuess
from mesmer.mesmer_x._utils import _ignore_warnings
from mesmer.stats import gaspari_cohn


class ConditionalDistributionOptions:
    def __init__(
        self,
        threshold_min_proba=1.0e-9,
        options_optim=None,
        options_solver=None,
    ):
        """Class to define optimizers used during first guess and training.

        Parameters
        ----------
        expression : py:class:Expression
            Expression to train. The string provided to the class can be found in
            'expr_fit.expression'.

        options_optim : dict, default: None
            A dictionary with options for the function to optimize:

            * type_fun_optim: string, default: "nll"
                If 'nll', will optimize using the negative log likelihood. If 'fcnll',
                will use the full conditional negative log likelihood based on the
                stopping rule. The arguments `threshold_stopping_rule`, `ind_year_thres`
                and `exclude_trigger` only apply to 'fcnll'.

            * threshold_stopping_rule: float > 1, default: None
                Maximum return period, used to define the threshold of the stopping
                rule.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

            * ind_year_thres: np.array, default: None
                Positions in the predictors where the thresholds have to be tested.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

            * exclude_trigger: boolean, default: None
                Whether the threshold will be included or not in the stopping rule.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

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

            * fg_with_global_opti : boolean, default: False
                If True, will add a step where the first guess is improved using a
                global optimization. In theory, better, but in practice, no improvement
                in performances, but makes the fit much slower (+ ~10s).

        threshold_min_proba :  float or None, default: 1e-9
            If numeric imposes a check during the fitting that every sample fulfills
            `cdf(sample) >= threshold_min_proba and 1 - cdf(sample) >= threshold_min_proba`,
            i.e. each sample lies within some confidence interval of the distribution.
            Note that it follows that threshold_min_proba math::\\in (0,0.5). Important to
            ensure that all points are feasible with the fitted distribution.
            If `None` this test is skipped.
        """

        # preparing solver
        default_options_solver = {
            "method_fit": "Powell",
            "xtol_req": 1e-6,
            "ftol_req": 1.0e-6,
            "maxiter": None,
            "maxfev": None,
            "error_failedfit": False,
            "fg_with_global_opti": False,
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
        self.fg_with_global_opti = options_solver["fg_with_global_opti"]

        # preparing information on function to optimize
        default_options_optim = dict(
            type_fun_optim="nll",
            threshold_stopping_rule=None,
            exclude_trigger=None,
            ind_year_thres=None,
        )

        options_optim = options_optim or {}

        if not isinstance(options_optim, dict):
            raise ValueError("`options_optim` must be a dictionary")

        options_optim = default_options_optim | options_optim

        # preparing information for the stopping rule
        self.type_fun_optim = options_optim["type_fun_optim"]
        self.threshold_stopping_rule = options_optim["threshold_stopping_rule"]
        self.ind_year_thres = options_optim["ind_year_thres"]
        self.exclude_trigger = options_optim["exclude_trigger"]

        if self.type_fun_optim == "nll" and (
            self.threshold_stopping_rule is not None or self.ind_year_thres is not None
        ):
            raise ValueError(
                "`threshold_stopping_rule` and `ind_year_thres` not used for"
                " `type_fun_optim='nll'`"
            )

        if self.type_fun_optim == "fcnll" and (
            self.threshold_stopping_rule is None or self.ind_year_thres is None
        ):
            raise ValueError(
                "`type_fun_optim='fcnll'` needs both, `threshold_stopping_rule`"
                "  and `ind_year_thres`."
            )

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
        """
        # initialization
        self.expression = expression
        self.options = options
        self._coefficients = None

    def fit(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        first_guess: xr.Dataset,
        weights: xr.DataArray,
        sample_dim: str = "sample",
        smooth_coeffs: bool = False,
        r_gasparicohn: float = 500,
        option_smooth_coeffs: None = None,  # deprecated in favor of smooth_coeffs
    ):
        """fit conditional distribution over all gridpoints.

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
            is a xr.Dataset, it must have each predictor as a DataArray.
        target : xr.DataArray
            Target DataArray.
        first_guess : xr.Dataset
            First guess for the coefficients, each coefficient is a data variable in the Dataset
            and has the corresponding name of the coefficient in the expression.
        weights : xr.DataArray.
            Individual weights for each sample.
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

        # checking for smoothing of coefficients, eg for 2nd round of fit
        if smooth_coeffs:
            gridcell_dim = (set(target.dims) - {sample_dim}).pop()
            coords = target[gridcell_dim].coords
            first_guess = _smoothen_first_guess(
                first_guess, gridcell_dim, coords, r_gasparicohn
            )

        # training
        coefficients = self._fit_xr(
            predictors, target, first_guess, weights, sample_dim
        )
        coefficients = coefficients.assign_attrs(
            {
                "expression_name": self.expression.expression_name,
                "expression": self.expression.expression,
            }
        )  # add expression as attribute

        self._coefficients = coefficients

    def find_first_guess(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        weights: xr.DataArray,
        sample_dim: str = "sample",
        first_guess: xr.Dataset | None = None,
    ):
        """
        Find a first guess for all coefficients of a conditional distribution for each grid point.

        Parameters
        ----------
        conditional_distrib : ConditionalDistribution
            Conditional distribution object to find the first guess for.
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `sample_dim`. If predictors
            is a xr.Dataset, it must have each predictor as a DataArray.
        target : xr.DataArray
            Target DataArray, contains at least `sample_dim`.
        weights : xr.DataArray.
            Individual weights for each sample, must be 1D along `sample_dim`.
        sample_dim : str
            Dimension along which to fit the first guess.
        first_guess : xr.Dataset, default: None
            If provided, will use these values as first guess for the first guess. If None,
            will use all zeros. Must contain the first guess for each coefficient in a
            DataArray with the name of the coefficient.

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

        # preparing data
        data_pred, data_targ, data_weights, first_guess = _distrib_checks._prepare_data(  # type: ignore
            predictors, target, weights, first_guess
        )

        # NOTE: extremely important that the order is the right one
        predictor_names = data_pred.coords["predictor"].values.tolist()

        # search for each gridpoint
        result = xr.apply_ufunc(
            self._find_fg_np,
            data_pred,
            data_targ,
            data_weights,
            first_guess,
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

    def _fit_xr(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        first_guess: xr.Dataset,
        weights: xr.DataArray,
        sample_dim: str,
    ) -> xr.Dataset:
        """
        xarray wrapper to fit over all gridpoints.

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
            is a xr.Dataset, it must have each predictor as a DataArray.
        target : xr.DataArray
            Target DataArray.
        first_guess : xr.Dataset
            First guess for the coefficients.
        sample_dim : str
            Dimension along which to fit the distribution.
        weights : xr.DataArray.
            Individual weights for each sample.

        Returns
        -------
        :obj:`xr.Dataset`
            Dataset of result of optimization (gridpoint, coefficient)
        """

        # shaping inputs
        data_pred, data_targ, data_weights, first_guess = _distrib_checks._prepare_data(  # type: ignore
            predictors, target, weights, first_guess
        )
        self.predictor_names = data_pred.coords["predictor"].values

        # search for each gridpoint
        result = xr.apply_ufunc(
            self._fit_np,
            data_pred,
            data_targ,
            first_guess,
            data_weights,
            input_core_dims=[
                [sample_dim, "predictor"],
                [sample_dim],
                ["coefficient"],
                [sample_dim],
            ],
            output_core_dims=[["coefficient"]],
            vectorize=True,  # Enable vectorization for automatic iteration over gridpoints
            dask="parallelized",
            output_dtypes=[float],
        )
        # creating a dataset with the coefficients
        out = xr.Dataset()
        for i, coef in enumerate(self.expression.coefficients_list):
            out[coef] = result.isel(coefficient=i)

        return out.drop_vars("coefficient")

    @_ignore_warnings  # suppress nan & inf warnings
    def _fit_np(self, data_pred, data_targ, fg, data_weights):
        _distrib_checks._check_no_nan_no_inf(data_pred, "predictor data")
        _distrib_checks._check_no_nan_no_inf(data_targ, "target data")
        _distrib_checks._check_no_nan_no_inf(data_weights, "weights")

        # basic check on first guess
        if fg is not None and len(fg) != len(self.expression.coefficients_list):
            raise ValueError(
                "The provided first guess does not have the correct number of coeffs, "
                f"Expression suggests a number of {len(self.expression.coefficients_list)} coefficients."
            )

        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        # NOTE: extremely important that the order is the right one
        data_pred = {key: data_pred[:, i] for i, key in enumerate(self.predictor_names)}

        # training
        m = _optimizers._minimize(
            func=_optimizers._func_optim,
            x0=fg,
            method_fit=self.options.method_fit,
            args=(
                data_pred,
                data_targ,
                data_weights,
                self.expression,
                self.options.threshold_min_proba,
                self.options.type_fun_optim,
                self.options.threshold_stopping_rule,
                self.options.exclude_trigger,
                self.options.ind_year_thres,
            ),
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

    def eval_quality_fit(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        dim: str,
        weights: xr.DataArray,
        scores_fit=["func_optim", "nll", "bic"],
    ):
        """Evaluate the scores for this fit.

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
            is a xr.Dataset, it must have each predictor as a DataArray.
        target : xr.DataArray
            Target DataArray.
        coefficients_fit : xr.DataArray
            Result from the optimization for the coefficients.
        dim : str
            Dimension along which to calculate the scores.
        weights : xr.DataArray.
            Individual weights for each sample.
        scores_fit : list of str, default: ['func_optim', 'nll', 'bic']
            After the fit, several scores can be calculated to assess the performance:
            - func_optim: function optimized, as described in
              options_optim['type_fun_optim']: negative log likelihood or full
              conditional negative log likelihood
            - nll: Negative Log Likelihood
            - bic: Bayesian Information Criteria
            - crps: Continuous Ranked Probability Score (warning, takes a long time to
              compute)
        """
        self.scores_fit = scores_fit

        coefficients_fit = self.coefficients

        # training
        quality_fit = self._eval_quality_fit_xr(
            predictors, target, coefficients_fit, dim, weights
        )
        return quality_fit

    def _eval_quality_fit_xr(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        coefficients_fit: xr.Dataset,
        dim: str,
        weights: xr.DataArray,
    ):
        """Evaluate the scores for this fit.

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
            is a xr.Dataset, it must have each predictor as a DataArray.
        target : xr.DataArray
            Target DataArray.
        coefficients_fit : xr.DataArray
            Result from the optimization for the coefficients.
        dim : str
            Dimension along which to calculate the scores.
        weights : xr.DataArray.
            Individual weights for each sample.
        scores_fit : list of str, default: ['func_optim', 'nll', 'bic']
            After the fit, several scores can be calculated to assess the performance:
            - func_optim: function optimized, as described in
              options_optim['type_fun_optim']: negative log likelihood or full
              conditional negative log likelihood
            - nll: Negative Log Likelihood
            - bic: Bayesian Information Criteria
            - crps: Continuous Ranked Probability Score (warning, takes a long time to
              compute)
        """
        # shaping inputs
        data_pred, data_targ, data_weights, _ = _distrib_checks._prepare_data(
            predictors, target, weights
        )
        self.predictor_names = data_pred.predictor.values

        # shaping coefficients
        da_coefficients = coefficients_fit.to_dataarray(dim="coefficient")

        # search for each gridpoint
        result = xr.apply_ufunc(
            self._eval_quality_fit_np,
            data_pred,
            data_targ,
            da_coefficients,
            data_weights,
            input_core_dims=[[dim, "predictor"], [dim], ["coefficient"], [dim]],
            output_core_dims=[["score"]],
            vectorize=True,  # Enable vectorization for automatic iteration over gridpoints
            dask="parallelized",
            output_dtypes=[float],
        )
        result["score"] = self.scores_fit
        return xr.Dataset({"scores": result})

    def _eval_quality_fit_np(
        self, data_pred, data_targ, coefficients_fit, data_weights
    ):
        _distrib_checks._check_no_nan_no_inf(data_pred, "predictor data")
        _distrib_checks._check_no_nan_no_inf(data_targ, "target data")
        _distrib_checks._check_no_nan_no_inf(data_weights, "weights")

        # initialize
        quality_fit = []

        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        data_pred = {key: data_pred[:, i] for i, key in enumerate(self.predictor_names)}

        for score in self.scores_fit:
            # basic result: optimized value
            if score == "func_optim":
                score = _optimizers._func_optim(
                    coefficients_fit,
                    data_pred,
                    data_targ,
                    data_weights,
                    self.expression,
                    self.options.threshold_min_proba,
                    self.options.type_fun_optim,
                    self.options.threshold_stopping_rule,
                    self.options.ind_year_thres,
                    self.options.exclude_trigger,
                )

            # calculating parameters for the next ones
            params = self.expression.evaluate_params(coefficients_fit, data_pred)

            # NLL averaged over sample
            if score == "nll":
                score = _optimizers._neg_loglike(
                    self.expression, data_targ, params, data_weights
                )

            # BIC averaged over sample
            if score == "bic":
                score = _optimizers._bic(
                    self.expression, data_targ, params, data_weights
                )

            # CRPS
            if score == "crps":
                score = _optimizers._crps(
                    self.expression,
                    data_targ,
                    data_pred,
                    data_weights,
                    coefficients_fit,
                )

            quality_fit.append(score)
        return np.array(quality_fit)

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


def _smoothen_first_guess(first_guess: xr.Dataset, dim, grid_coords, r_gasparicohn):
    """
    smoothen first guess over

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
