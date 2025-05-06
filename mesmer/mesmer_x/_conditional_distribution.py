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
from mesmer.mesmer_x._expression import Expression
from mesmer.mesmer_x._distrib_tests import distrib_tests
from mesmer.mesmer_x._optimizers import distrib_optimizer
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

class ConditionalDistribution:
    def __init__(
        self,
        expr_fit: Expression,
        class_tests: distrib_tests,
        class_optim: distrib_optimizer,
    ):
        """Fit a conditional distribution.

        Parameters
        ----------
        expr_fit : class 'expression'
            Expression to train. The string provided to the class can be found in
            'expr_fit.expression'.

        class_tests : class 'distrib_tests'
            Class defining the tests to perform during first guess and training.

        class_optim : class 'distrib_optimizer'
            Class defining the optimizer used during first guess and training of
            distributions.
        """
        # initialization
        self.expr_fit = expr_fit
        self.class_optim = class_optim
        self.class_tests = class_tests

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
            for coef in self.expr_fit.coefficients_list:
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
        data_pred, data_targ, data_weights = self.class_tests.prepare_data(
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
        for icoef, coef in enumerate(self.expr_fit.coefficients_list):
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
        self.class_tests.validate_data(data_pred, data_targ, data_weights)

        # basic check on first guess
        if (fg is not None) and (len(fg) != len(self.expr_fit.coefficients_list)):
            raise ValueError(
                f"The provided first guess does not have the correct shape: {len(self.expr_fit.coefficients_list)}"
            )

        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        data_pred = {pp: data_pred[:, ii] for ii, pp in enumerate(self.predictor_dim)}

        # training
        m = self.class_optim._minimize(
            func=self.class_optim.func_optim,
            x0=fg,
            args=(data_pred, data_targ, data_weights),
            fact_maxfev_iter=1,
            option_NelderMead="best_run",
        )

        # checking if the fit has failed
        if self.class_optim.error_failedfit and not m.success:
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
        data_pred, data_targ, data_weights = self.class_tests.prepare_data(
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
        self.class_tests.validate_data(data_pred, data_targ, data_weights)

        # initialize
        quality_fit = []

        # correcting format: must be dict(str, DataArray or array) for Expression
        # TODO: to change with stabilization of data format
        data_pred = {pp: data_pred[:, ii] for ii, pp in enumerate(self.predictor_dim)}

        for score in self.scores_fit:
            # basic result: optimized value
            if score == "func_optim":
                score = self.class_optim.func_optim(
                    coefficients_fit, data_pred, data_targ, data_weights
                )

            # calculating parameters for the next ones
            params = self.expr_fit.evaluate_params(coefficients_fit, data_pred)

            # NLL averaged over sample
            if score == "nll":
                score = self.class_optim.neg_loglike(data_targ, params, data_weights)

            # BIC averaged over sample
            if score == "bic":
                score = self.class_optim.bic(data_targ, params, data_weights)

            # CRPS
            if score == "crps":
                score = self.class_optim.crps(
                    data_targ, data_pred, data_weights, coefficients_fit
                )

            quality_fit.append(score)
        return np.array(quality_fit)
