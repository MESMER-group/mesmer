# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import functools
import warnings

import numpy as np
import xarray as xr
from scipy.optimize import basinhopping, shgo

import mesmer.mesmer_x._distrib_tests as distrib_tests
import mesmer.mesmer_x._optimizers as distrib_optimizers
from mesmer.mesmer_x._conditional_distribution import ConditionalDistribution


def _finite_difference(f_high, f_low, x_high, x_low):
    return (f_high - f_low) / (x_high - x_low)


def _smooth_data(data, length=5):
    """
    Smooth 1D data using convolution.

    Parameters
    ----------
    data : numpy array 1D
        Data to smooth
    length: integer, default: 5
        The length of the half-window for the convolution.

    Returns
    -------
    obj: numpy array 1D
        Smoothed data
    """
    tmp = np.convolve(data, np.ones(2 * length + 1) / (2 * length + 1), mode="valid")
    # removing the bias in the mean
    tmp += np.mean(data[length:-length]) - np.mean(tmp)
    return tmp


def ignore_warnings(func):

    # adapted from https://stackoverflow.com/a/70292317
    # TODO: don't suppress all warnings
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return func(*args, **kwargs)

    return _wrapper


def find_first_guess(
    conditional_distrib: ConditionalDistribution,
    predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
    target: xr.DataArray,
    dim: str,
    weights: xr.DataArray,
    first_guess: xr.Dataset | None = None,
):
    """
    Find a first guess for all grid points.

    Parameters
    ----------
    predictors : dict of xr.DataArray | DataTree | xr.Dataset
        A dict of DataArray objects used as predictors or a DataTree, holding each
        predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
        is a xr.Dataset, it must have each predictor as a DataArray.
    target : xr.DataArray
        Target DataArray.
    dim : str
        Dimension along which to fit the polynomials.
    weights : xr.DataArray.
        Individual weights for each sample.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset of first guess (gridpoint, coefficient)
    """
    # TODO: some smoothing on first guess? cf 2nd fit with MESMER-X given results.

    # TODO: make fg if none
    if first_guess is None:
        pass

    # preparing data
    data_pred, data_targ, data_weights = distrib_tests.prepare_data(
        predictors, target, weights
    )

    # search for each gridpoint
    result = xr.apply_ufunc(
        _find_fg_np,
        data_pred,
        data_targ,
        data_weights,
        # first_guess,
        kwargs={"conditional_distrib": conditional_distrib},
        input_core_dims=[[dim, "predictor"], [dim], [dim]],  # [dim, "coefficient"]],
        output_core_dims=[["coefficient"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    # creating a dataset with the coefficients
    out = xr.Dataset()
    for icoef, coef in enumerate(conditional_distrib.expression.coefficients_list):
        out[coef] = result.isel(coefficient=icoef)
    return out


def _find_fg_np(
    data_pred, data_targ, data_weights, conditional_distrib: ConditionalDistribution
):

    fg = FirstGuess(conditional_distrib, data_pred, data_targ, data_weights)

    # TODO split up into the several steps
    return fg._find_fg_allsteps()


class FirstGuess:
    def __init__(
        self,
        conditional_distrib: ConditionalDistribution,
        data_pred,
        data_targ,
        data_weights,
        first_guess=None,
        func_first_guess=None,
    ):
        """
        First guess for the coefficients of the conditional distribution.

        Parameters
        ----------
        conditional_distrib : ConditionalDistribution
            Conditional distribution object. Must contain the expression and the
            options.

        first_guess : numpy array, default: None
            If provided, will use these values as first guess for the first guess.

        func_first_guess : callable, default: None
            If provided, and that 'first_guess' is not provided, will be called to
            provide a first guess for the fit. This is an experimental feature, thus not
            tested.
            !! BE AWARE THAT THE ESTIMATION OF A FIRST GUESS BY YOUR MEANS COMES AT YOUR
            OWN RISKS.

        """
        # initialization
        self.options = conditional_distrib.options
        self.expression = conditional_distrib.expression
        self.func_first_guess = func_first_guess

        # preparing additional information
        self.n_coeffs = len(self.expression.coefficients_list)

        distrib_tests.validate_data(data_pred, data_targ, data_weights)

        expression = self.expression

        self.predictor_names = expression.predictors_list
        n_preds = len(self.predictor_names)

        # ensuring format of numpy predictors
        if data_pred.ndim == 0:
            if n_preds == 0:
                data_pred = np.ones((data_targ.size, 1))
            else:
                raise Exception("Missing data on the predictors.")
        elif data_pred.ndim == 1:
            data_pred = data_pred[:, np.newaxis]
        elif data_pred.ndim == 2:
            if n_preds == data_pred.shape[0]:
                data_pred = data_pred.T
        else:
            raise Exception("Numpy predictors should not have a shape greater than 2.")

        # build dictionary
        self.data_pred = {
            pp: data_pred[:, ii] for ii, pp in enumerate(self.predictor_names)
        }

        self.data_targ = data_targ

        # smooting to help with location & scale
        self.l_smooth = 5
        self.smooth_targ = _smooth_data(data_targ, length=self.l_smooth)
        self.smooth_targ_dev = (
            data_targ[self.l_smooth : -self.l_smooth] - self.smooth_targ
        )
        self.smooth_pred = {
            pp: _smooth_data(self.data_pred[pp], length=self.l_smooth)
            for pp in self.predictor_names
        }

        self.data_weights = data_weights

        if first_guess is None:
            first_guess = np.zeros(self.n_coeffs)
        self.fg_coeffs = np.copy(first_guess)
        # make sure all values are floats bc if fg_coeff[ind] = type(int) we can only put ints in it too
        self.fg_coeffs = self.fg_coeffs.astype(float)

    # suppress nan & inf warnings
    @ignore_warnings
    def _find_fg_allsteps(self):
        """
        compute first guess of the coefficients, to ensure convergence of the incoming
        fit.

        Motivation:
            In many situations, the fit may be complex because of complex expressions
            for the conditional distributions & because large domains in the set of
            coefficients lead to invalid fits (e.g. sample out of support).

        Criteria:
            The method must return a first guess that is ROBUST (close to the global
            minimum) & VALID (respect all conditions implemented in the tests), must be
            FLEXIBLE (any sample, any distribution & any expression).

        Method:
            1. Improve very first guess for the location by finding a global fit of the
                coefficients for the location by fitting for coefficients that give location that
                is a) close to the mean of the target samples and b) has a similar change with the
                predictor as the target, i.e. the approximation of the first derivative of the location
                as function of the predictors is close to the one of the target samples. The first
                derivative is approximated by computing the quotient of the differences in the mean
                of high (outside +1 std) samples and low (outside -1 std) samples and their
                corresponding predictor values.
            2. Fit of the coefficients of the location, assuming that the center of the
                distribution should be close to its location by minimizing mean squared error
                between location and target samples.
            3. Fit of the coefficients of the scale, assuming that the deviation of the
                distribution should be close from its scale, by minimizing mean squared error between
                the scale and the absolute deviations of the target samples from the location.
            4. Fit of remaining coefficients, assuming that the sample must be within
                the support of the distribution, with some margin. Optimizing for coefficients
                that yield a support such that all samples are within this support.
            5. Improvement of all coefficients: better coefficients on location & scale,
                and especially estimating those on shape. Optimized using the Negative Log
                Likelihoog, albeit without the validity of the coefficients.
            6. If step 5 yields invalid coefficients, fit coefficients on log-likelihood to the power n
                to ensure that all points are within a likely
                support of the distribution. Two possibilities tried: based on CDF or based on NLL^n. The idea is
                to penalize very unlikely values, both works, but NLL^n works as well for
                extremely unlikely values, that lead to division by 0 with CDF)
            7. If required (distrib_optimizers.fg_with_global_opti), global fit within boundaries

        Risks for the method:
            The only risk that I identify is if the user sets boundaries on coefficients
            or parameters that would reject this optimal first guess, and the domain of
            the next local minimum is far away.

        Justification for the method:
            This is a surprisingly complex problem to satisfy the criteria of
            robustness, validity & flexibility.

            a. In theory, this problem could be solved with a global optimization. Among
                the global optimizers available in scipy, they come with two types of
                requirements:
                - basinhopping: requires a first guess. Tried with first guess close from
                    optimum, good performances, but lacks in reproductibility and
                    stability: not reliable enough here. Ok if runs much longer.
                - brute, differential_evolution, shgo, dual_annealing, direct: requires
                    bounds
                    - brute, dual_annealing, direct: performances too low & too slow
                    - differential_evolution: lacks in reproductibility & stability
                    - shgo: good performances with the right sampling method, relatively
                    fast, but still adds ~10s. Highly dependent on the bounds, must not
                    be too large.
                The best global optimizer, shgo, would then require bounds that are not
                too large.

            b. The set of coefficients have only sparse valid domains. The distance
                between valid domains is often bigger than the adequate width of bounds
                for shgo.
                It implies that the bounds used for shgo must correspond to the valid
                domain that already contains the global minimum, and no other domain.
                It implies that the region of the global minimum must have already been
                identified...
                This is the tricky part, here are the different elements that I used to
                tackle this problem:
                - all combinations of local & global optimizers of scipy
                - including or not the tests for validity
                - assessment of bounds for global optimizers based on ratio of NLL or
                    domain of data_targ
                - first optimizations based on mean & spread
                - brute force approach on logspace valid for parameters (not scalable
                    with # of parameters)
                c. The only solution that was working is inspired by what the
                semi-analytical solution used in the original code of MESMER-X. Its
                principle is to use fits first for location coefficients, then scale,
                then improve.
                The steps are described in the section Method, steps 1-5. At step 5 that
                these coefficients are very close from the global minimum. shgo usually
                does not bring much at the expense of speed.
                Thus skipping global minimum unless asked.

        Warnings
        --------
        To anyone trying to improve this part:
        If you attempt to modify the calculation of the first guess, it is *absolutely
        mandatory* to test the new code on all criteria: ROBUSTNESS, VALIDITY,
        FLEXIBILITY. In particular, it is mandatory to test it for different
        situations: variables, grid points, distributions & expressions.
        """

        # Step 1: fit coefficients of location (objective: generate an adequate
        # first guess for the coefficients of location. proven to be necessary
        # in many situations, & accelerate step 2)
        fg_ind_loc = self.expression.ind_loc_coeffs

        # location might not be used (beta distribution) or set in the expression
        if len(fg_ind_loc) > 0:

            # preparing derivatives to estimate derivatives of data along predictors,
            # and infer a very first guess for the coefficients facilitates the
            # representation of the trends
            m_smooth_targ = np.mean(self.smooth_targ)
            s_smooth_targ = np.std(self.smooth_targ)
            ind_targ_low = np.where(self.smooth_targ < m_smooth_targ - s_smooth_targ)[0]
            ind_targ_high = np.where(self.smooth_targ > m_smooth_targ + s_smooth_targ)[
                0
            ]
            mean_high_preds = {
                pp: np.mean(self.smooth_pred[pp][ind_targ_high], axis=0)
                for pp in self.predictor_names
            }
            mean_low_preds = {
                pp: np.mean(self.smooth_pred[pp][ind_targ_low], axis=0)
                for pp in self.predictor_names
            }

            derivative_targ = {
                pp: _finite_difference(
                    np.mean(self.smooth_targ[ind_targ_high]),
                    np.mean(self.smooth_targ[ind_targ_low]),
                    mean_high_preds[pp],
                    mean_low_preds[pp],
                )
                for pp in self.predictor_names
            }

            minimizer_kwargs = {
                "args": (
                    mean_high_preds,
                    mean_low_preds,
                    derivative_targ,
                    m_smooth_targ,
                )
            }
            # TODO: do we move the basinhopping part to the class optimizers?
            globalfit_d01 = basinhopping(
                func=self._fg_fun_deriv01,
                x0=self.fg_coeffs[fg_ind_loc],
                niter=10,
                interval=100,
                minimizer_kwargs=minimizer_kwargs,
            )
            # warning, basinhopping tends to introduce non-reproductibility in fits,
            # reduced when using 2nd round of fits

            self.fg_coeffs[fg_ind_loc] = globalfit_d01.x

        # Step 2: fit coefficients of location (objective: improving the subset of
        # location coefficients)
        if len(fg_ind_loc) > 0:
            fact_maxfev_iter = len(fg_ind_loc) / self.n_coeffs

            localfit_loc = distrib_optimizers._minimize(
                func=self._fg_fun_loc,
                x0=self.fg_coeffs[fg_ind_loc],
                args=(),
                method_fit=self.options.method_fit,
                option_NelderMead="best_run",
                options={
                    "maxfev": self.options.maxfev * fact_maxfev_iter,
                    "maxiter": self.options.maxiter * fact_maxfev_iter,
                    self.options.name_xtol: self.options.xtol_req,
                    self.options.name_ftol: self.options.ftol_req,
                },
            )
            self.fg_coeffs[fg_ind_loc] = localfit_loc.x

        # Step 3: fit coefficients of scale (objective: improving the subset of
        # scale coefficients)
        fg_ind_sca = self.expression.ind_sca_coeffs
        # scale might not be used or set in the expression
        if len(fg_ind_sca) > 0:
            x0 = self.fg_coeffs[fg_ind_sca]
            fact_maxfev_iter = len(fg_ind_sca) / self.n_coeffs

            localfit_sca = distrib_optimizers._minimize(
                func=self._fg_fun_sca,
                x0=x0,
                args=(),
                method_fit=self.options.method_fit,
                option_NelderMead="best_run",
                options={
                    "maxfev": self.options.maxfev * fact_maxfev_iter,
                    "maxiter": self.options.maxiter * fact_maxfev_iter,
                    self.options.name_xtol: self.options.xtol_req,
                    self.options.name_ftol: self.options.ftol_req,
                },
            )
            self.fg_coeffs[fg_ind_sca] = localfit_sca.x

        # Step 4: fit other coefficients (objective: improving the subset of
        # other coefficients. May use multiple coefficients, eg beta distribution)
        other_params = [
            p for p in self.expression.parameters_list if p not in ["loc", "scale"]
        ]
        if len(other_params) > 0:
            fg_ind_others = self.expression.ind_others
            fact_maxfev_iter = len(fg_ind_others) / self.n_coeffs

            localfit_others = distrib_optimizers._minimize(
                func=self._fg_fun_others,
                x0=self.fg_coeffs[fg_ind_others],
                args=(),
                method_fit=self.options.method_fit,
                option_NelderMead="best_run",
                options={
                    "maxfev": self.options.maxfev * fact_maxfev_iter,
                    "maxiter": self.options.maxiter * fact_maxfev_iter,
                    self.options.name_xtol: self.options.xtol_req,
                    self.options.name_ftol: self.options.ftol_req,
                },
            )
            self.fg_coeffs[fg_ind_others] = localfit_others.x

        # Step 5: fit coefficients using NLL (objective: improving all coefficients,
        # necessary to get good estimates for shape parameters, and avoid some local minima)
        localfit_nll = distrib_optimizers._minimize(
            func=self._fg_fun_nll_no_tests,
            x0=self.fg_coeffs,
            args=(),
            method_fit=self.options.method_fit,
            option_NelderMead="best_run",
            options={
                "maxfev": self.options.maxfev,
                "maxiter": self.options.maxiter,
                self.options.name_xtol: self.options.xtol_req,
                self.options.name_ftol: self.options.ftol_req,
            },
        )
        fg_coeffs = localfit_nll.x

        test_coeff, test_param, test_distrib, test_proba, _ = (
            distrib_tests.validate_coefficients(
                self.expression,
                self.data_pred,
                self.data_targ,
                fg_coeffs,
                self.options.threshold_min_proba,
            )
        )

        # if any of validate_coefficients test fail (e.g. any of the coefficients are out of bounds)
        if not (test_coeff and test_param and test_distrib and test_proba):
            # Step 6: fit on LL^n (objective: improving all coefficients, necessary
            # to have all points within support. NB: NLL does not behave well enough here)

            localfit_opti = distrib_optimizers._minimize(
                func=self._fg_fun_ll_n,
                x0=fg_coeffs,
                args=(),
                method_fit=self.options.method_fit,
                option_NelderMead="best_run",
                options={
                    "maxfev": self.options.maxfev,
                    "maxiter": self.options.maxiter,
                    self.options.name_xtol: self.options.xtol_req,
                    self.options.name_ftol: self.options.ftol_req,
                },
            )
            if ~np.any(np.isnan(localfit_opti.x)):
                fg_coeffs = localfit_opti.x

        # Step 7: if required, global fit within boundaries
        if self.options.fg_with_global_opti:

            # find boundaries on each coefficient
            bounds = []

            # TODO: does this assume the coeffs are ordered?
            for i_c in np.arange(self.n_coeffs):
                a = self.find_bound(i_c=i_c, x0=fg_coeffs, fact_coeff=-0.05)
                b = self.find_bound(i_c=i_c, x0=fg_coeffs, fact_coeff=0.05)
                vals_bounds = (a, b)

                bounds.append([np.min(vals_bounds), np.max(vals_bounds)])

            # global minimization, using the one with the best performances in this
            # situation. sobol or halton, observed lower performances with
            # simplicial. n=1000, options={'maxiter':10000, 'maxev':10000})
            globalfit_all = shgo(
                func=distrib_optimizers.func_optim,
                bounds=bounds,
                args=(self.data_pred, self.data_targ, self.data_weights),
                sampling_method="sobol",
            )
            if not globalfit_all.success:
                raise ValueError(
                    "Global optimization for first guess failed, please check boundaries_coeff or ",
                    "disable fg_with_global_opti in options_solver of distrib_optimizers.",
                )
            fg_coeffs = globalfit_all.x
        return fg_coeffs

    def _fg_fun_deriv01(self, x_loc, pred_high, pred_low, derivative_targ, mean_targ):
        r"""
        Loss function for very fist guess of the location coefficients. The objective is
        to 1) get a location with a similar change with the predictor as the target, and
        2) get a location close to the mean of the target samples.

        The loss is computed as follows:

        .. math::

        \sum^{p}{(\frac{\Delta loc(x)}{\Delta pred} - \frac{\Delta targ}{\Delta pred}^2)} + (mean_{loc} - mean_{targ})^2

        Parameters
        ----------
        x : numpy array
            Coefficients for the location
        pred_high : dict[str, numpy array]
            Predictors for the high samples of the targets
        pred_low : dict[str, numpy array]
            Predictors for the low samples of the targets
        derivative_targ : dict
            Derivatives of the target samples for every predictor
        mean_targ : float
            Mean of the (smoothed) target samples

        Returns
        -------
        float
            Loss value
        """
        x = np.copy(self.fg_coeffs)
        x[self.expression.ind_loc_coeffs] = x_loc
        params = self.expression.evaluate_params(x, pred_low)
        loc_low = params["loc"]
        params = self.expression.evaluate_params(x, pred_high)
        loc_high = params["loc"]

        derivative_loc = {
            p: _finite_difference(loc_high, loc_low, pred_high[p], pred_low[p])
            for p in self.data_pred
        }

        return (
            np.sum(
                [(derivative_loc[p] - derivative_targ[p]) ** 2 for p in self.data_pred]
                # ^ change of the location with the predictor should be similar to the change of the target with the predictor
            )
            + (0.5 * (loc_low + loc_high) - mean_targ) ** 2
            # ^ location should not be too far from the mean of the samples
        )

    def _fg_fun_loc(self, x_loc):
        r"""
        Loss function for the location coefficients. The objective is to get a location
        such that the center of the distribution is close to the location, thus we fit a mean
        squared error between the location and the smoothed target samples.

        The loss is computed as follows:

        .. math::

        mean((loc(x\_loc) - smooth\_target_{n})^2)

        Parameters
        ----------
        x_loc : numpy array
            Coefficients for the location

        Returns
        -------
        float
            Loss value
        """
        x = np.copy(self.fg_coeffs)
        x[self.expression.ind_loc_coeffs] = x_loc
        params = self.expression.evaluate_params(x, self.smooth_pred)
        if distrib_tests._test_evol_params(self.expression, params):
            loc = params["loc"]
            return np.mean((loc - self.smooth_targ) ** 2)

        else:
            # this coefficient on location causes problem
            return np.inf

    def _fg_fun_sca(self, x_sca):
        r"""
        Loss function for the scale coefficients. The objective is to get a scale such that
        the deviation of the distribution is close to the scale, thus we fit a mean squared
        error between the deviation of the (not smoothed) target samples from the location and the scale.

        The loss is computed as follows:

        .. math::

        mean(deviation_{n} - scale(x\_sca))^2)

        Parameters
        ----------
        x_sca : numpy array
            Coefficients for the scale

        Returns
        -------
        float
            Loss value

        """
        x = np.copy(self.fg_coeffs)
        x[self.expression.ind_sca_coeffs] = x_sca
        params = self.expression.evaluate_params(x, self.data_pred)

        if distrib_tests._test_evol_params(self.expression, params):
            if isinstance(params["scale"], np.ndarray):
                sca = params["scale"][self.l_smooth : -self.l_smooth]
            else:
                sca = params["scale"]
            return np.abs(np.mean(self.smooth_targ_dev**2 - sca**2))

        else:
            # this coefficient on scale causes problem
            return np.inf

    def _fg_fun_others(self, x_others, margin0=1.0e-3):
        """
        Loss function for other coefficients than loc and scale. Objective is to tune parameters such
        that target samples are within a likely range of the distribution. Instead of relying on the
        support, we use the cumulative distribution function (CDF) to penalize unlikely samples.
        """
        # prepare coefficients
        x = np.copy(self.fg_coeffs)
        x[self.expression.ind_others] = x_others

        # evaluate parameters
        params = self.expression.evaluate_params(x, self.data_pred)

        if distrib_tests._test_evol_params(self.expression, params):
            # compute CDF values for the target samples
            cdf_values = self.expression.distrib.cdf(self.data_targ, **params)

            if (np.min(cdf_values) < margin0) or (np.max(cdf_values) > 1 - margin0):
                return np.inf

            else:
                # penalize samples with CDF values close to 0 or 1 (unlikely samples)
                penalty_low = np.maximum(0, margin0 - cdf_values) ** 2
                penalty_high = np.maximum(0, cdf_values - (1 - margin0)) ** 2

                # sum penalties to compute the loss
                return np.sum(penalty_low + penalty_high)
        else:
            # the coefficients cause problems
            return np.inf

    def _fg_fun_nll_no_tests(self, coefficients):
        params = self.expression.evaluate_params(coefficients, self.data_pred)
        return distrib_optimizers.neg_loglike(
            self.expression, self.data_targ, params, self.data_weights
        )

    def _fg_fun_ll_n(self, x, n=4):
        params = self.expression.evaluate_params(x, self.data_pred)

        if self.expression.is_distrib_discrete:
            LL = np.sum(self.expression.distrib.logpmf(self.data_targ, **params) ** n)
        else:
            LL = np.sum(self.expression.distrib.logpdf(self.data_targ, **params) ** n)
        return LL

    def find_bound(self, i_c, x0, fact_coeff):
        """
        expand bound until coefficients are valid: starts  with x0
        and multiplies with fact_coeff until validate_coefficients returns all True.
        """
        # could be accelerated using dichotomy, but 100 iterations max are fast enough
        # not to require to make this part more complex.
        x, iter, itermax, test = np.copy(x0), 0, 100, True
        while test and (iter < itermax):
            test_c, test_p, test_d, test_v, _ = distrib_tests.validate_coefficients(
                self.expression,
                self.data_pred,
                self.data_targ,
                x,
                self.options.threshold_min_proba,
            )
            test = test_c and test_p and test_d and test_v
            x[i_c] += fact_coeff * x[i_c]
            iter += 1
        # TODO: returns value after test = False, but should it maybe be the last value before that?
        return x[i_c]
