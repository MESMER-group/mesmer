# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
import functools
import warnings

import numpy as np
import xarray as xr

# TODO: to replace with outputs from PR #607
from mesmer.core.geospatial import geodist_exact
import mesmer.mesmer_x._distrib_tests as distrib_tests
from mesmer.mesmer_x._expression import Expression
import mesmer.mesmer_x._optimizers as distrib_optimizers
from mesmer.mesmer_x._probability_integral_transform import weighted_median
from mesmer.stats import gaspari_cohn


def ignore_warnings(func):

    # adapted from https://stackoverflow.com/a/70292317
    # TODO: don't suppress all warnings
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return func(*args, **kwargs)

    return _wrapper

class ConditionalDistributionOptions:
    def __init__(
        self,
        expression: Expression,
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

            * maxiter: int, default: 10000
                Maximum number of iteration of the optimization.

            * maxfev: int, default: 10000
                Maximum number of evaluation of the function during the optimization.

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

        n_coeffs = len(expression.coefficients_list)

        # preparing solver
        default_options_solver = {
            "method_fit": "Powell",
            "xtol_req": 1e-6,
            "ftol_req": 1.0e-6,
            "maxiter": 1000 * n_coeffs * (np.log(n_coeffs) + 1),
            "maxfev": 1000 * n_coeffs * (np.log(n_coeffs) + 1),
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
            raise ValueError("`threshold_min_proba` must be in (0, 0.5)")

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

    def fit(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        first_guess: xr.DataArray,
        dim: str,
        weights: xr.DataArray,
        option_smooth_coeffs: bool = False,
        r_gasparicohn: float = 500,
    ):
        """Wrapper to fit over all gridpoints.

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
        dim : str
            Dimension along which to find the first guess.
        weights : xr.DataArray.
            Individual weights for each sample.
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
            Dataset of result of optimization (gridpoint, coefficient)
        """
        self.option_smooth_coeffs = option_smooth_coeffs
        self.r_gasparicohn = r_gasparicohn

        # training
        coefficients = self._fit_xr(predictors, target, first_guess, dim, weights)

        return coefficients

    def _fit_xr(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        first_guess: xr.DataArray,
        dim: str,
        weights: xr.DataArray,
    ):
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
        dim : str
            Dimension along which to find the first guess.
        weights : xr.DataArray.
            Individual weights for each sample.

        Returns
        -------
        :obj:`xr.Dataset`
            Dataset of result of optimization (gridpoint, coefficient)
        """
        # checking for smoothing of coefficients, eg for 2nd round of fit
        if self.option_smooth_coeffs:
            # calculating distance between points
            geodist = geodist_exact(target["lon"], target["lat"])

            # deducing correlation matrix
            corr_gc = gaspari_cohn(geodist / self.r_gasparicohn)

            # will avoid taking gridpoints with nan
            gp_nonan = first_guess.notnull().to_array().all(dim=["variable"]).values

            # creating new dataset of coefficients
            second_guess = np.nan * xr.ones_like(first_guess)

            # calculating for each coef and each gridpoint the weighted median
            for coef in self.expression.coefficients_list:
                for gp in first_guess.gridpoint.values:
                    fg = weighted_median(
                        data=first_guess[coef].sel(gridpoint=gp_nonan).values,
                        weights=corr_gc.sel(
                            gridpoint_i=gp, gridpoint_j=gp_nonan
                        ).values,
                    )
                    second_guess[coef].loc[dict(gridpoint=gp)] = fg

            # preparing for training
            first_guess = second_guess

        # shaping inputs
        data_pred, data_targ, data_weights = distrib_tests.prepare_data(
            predictors, target, weights
        )
        self.predictor_dim = data_pred.predictor.values

        # shaping coefficients
        da_first_guess = first_guess.to_dataarray(dim="coefficient")

        # search for each gridpoint
        result = xr.apply_ufunc(
            self._fit_np,
            data_pred,
            data_targ,
            da_first_guess,
            data_weights,
            input_core_dims=[[dim, "predictor"], [dim], ["coefficient"], [dim]],
            output_core_dims=[["coefficient"]],
            vectorize=True,  # Enable vectorization for automatic iteration over gridpoints
            dask="parallelized",
            output_dtypes=[float],
        )
        # creating a dataset with the coefficients
        out = xr.Dataset()
        for icoef, coef in enumerate(self.expression.coefficients_list):
            out[coef] = result.isel(coefficient=icoef)
        return out

    @ignore_warnings  # suppress nan & inf warnings
    def _fit_np(self, data_pred, data_targ, fg, data_weights):
        # check data
        if not isinstance(data_pred, np.ndarray):
            raise Exception("data_pred must be a numpy array.")
        if not isinstance(data_targ, np.ndarray):
            raise Exception("data_targ must be a numpy array.")
        if not isinstance(data_weights, np.ndarray):
            raise Exception("data_weights must be a numpy array.")
        distrib_tests.validate_data(data_pred, data_targ, data_weights)

        # basic check on first guess
        if (fg is not None) and (len(fg) != len(self.expression.coefficients_list)):
            raise ValueError(
                f"The provided first guess does not have the correct shape: {len(self.expression.coefficients_list)}"
            )

        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        data_pred = {pp: data_pred[:, ii] for ii, pp in enumerate(self.predictor_dim)}

        # training
        m = distrib_optimizers._minimize(
            func=distrib_optimizers.func_optim,
            x0=fg,
            method_fit=self.options.method_fit,
            args=(data_pred, data_targ, data_weights),
            option_NelderMead="best_run",
            options={
                "maxfev": self.options.maxfev,
                "maxiter": self.options.maxiter,
                self.options.name_xtol: self.options.xtol_req,
                self.options.name_ftol: self.options.ftol_req,
            }
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
        coefficients_fit: xr.DataArray,
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

        # training
        quality_fit = self._eval_quality_fit_xr(
            predictors, target, coefficients_fit, dim, weights
        )
        return quality_fit

    def _eval_quality_fit_xr(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        coefficients_fit: xr.DataArray,
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
        data_pred, data_targ, data_weights = distrib_tests.prepare_data(
            predictors, target, weights
        )
        self.predictor_dim = data_pred.predictor.values

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
        # check data
        if not isinstance(data_pred, np.ndarray):
            raise Exception("data_pred must be a numpy array.")
        if not isinstance(data_targ, np.ndarray):
            raise Exception("data_targ must be a numpy array.")
        if not isinstance(data_weights, np.ndarray):
            raise Exception("data_weights must be a numpy array.")
        distrib_tests.validate_data(data_pred, data_targ, data_weights)

        # initialize
        quality_fit = []

        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        data_pred = {pp: data_pred[:, ii] for ii, pp in enumerate(self.predictor_dim)}

        for score in self.scores_fit:
            # basic result: optimized value
            if score == "func_optim":
                score = distrib_optimizers.func_optim(
                    self.expression, coefficients_fit, data_pred, data_targ, data_weights,
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
                score = distrib_optimizers.neg_loglike(self.expression, data_targ, params, data_weights)

            # BIC averaged over sample
            if score == "bic":
                score = distrib_optimizers.bic(self.options, data_targ, params, data_weights)

            # CRPS
            if score == "crps":
                score = distrib_optimizers.crps(
                    self.expression, data_targ, data_pred, data_weights, coefficients_fit
                )

            quality_fit.append(score)
        return np.array(quality_fit)
