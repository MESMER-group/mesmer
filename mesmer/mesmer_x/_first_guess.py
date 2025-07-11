# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import scipy as sp
from packaging.version import Version
from scipy.optimize import basinhopping

import mesmer.mesmer_x._distrib_checks as _distrib_checks
import mesmer.mesmer_x._optimizers as _optimizers
from mesmer.mesmer_x._expression import Expression
from mesmer.mesmer_x._optimizers import MinimizeOptions
from mesmer.mesmer_x._utils import _ignore_warnings

SEED_BASINHOPPING = 1931102249669598594


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
    out = np.convolve(data, np.ones(2 * length + 1) / (2 * length + 1), mode="valid")
    # removing the bias in the mean
    out += np.mean(data[length:-length]) - np.mean(out)
    return out


class _FirstGuess:
    def __init__(
        self,
        expression: Expression,
        minimize_options: MinimizeOptions,
        data_pred,
        predictor_names,
        data_targ,
        data_weights,
        first_guess,
        threshold_min_proba=1e-9,
    ):
        """
        First guess for the coefficients of the conditional distribution.

        Parameters
        ----------
        expression : Expression
            Expression for the conditional distribution we want to find the first guess of.

        minimizer_options : MinimizeOptions
            Options for the fit.

        data_pred : numpy array 1D or 2D of shape (n_samples, n_preds)
            Predictors for the training sample. If 2D, the first dimension must be the
            number of predictors, and the second dimension the number of samples, i.e. (n_samples, n_preds).

        predictor_names : list of str
            Names of the predictors as named in the `expression`.

        data_targ : numpy array 1D of shape (n_samples,)
            Target for the training sample. Must be 1D, i.e. (n_samples,).

        data_weights : numpy array 1D of shape (n_samples,)
            Weights for the training sample. Must be 1D, i.e. (n_samples,).

        first_guess : numpy array, default: None
            If provided, will use these values as first guess for the first guess, must be one value
            per coeff in expression.coefficients_list.

        """

        # initialization
        self.minimize_options = minimize_options
        self.expression = expression
        self.threshold_min_proba = threshold_min_proba

        if data_pred is not None:
            _distrib_checks._check_no_nan_no_inf(data_pred, "predictor data")
        _distrib_checks._check_no_nan_no_inf(data_targ, "target data")
        _distrib_checks._check_no_nan_no_inf(data_weights, "weights")

        if predictor_names is None:
            if data_pred is not None:
                raise ValueError(
                    "If data_pred is provided, predictor_names must be provided as well."
                )
            predictor_names = []
        elif data_pred is None:
            raise ValueError(
                "If predictor_names is provided, data_pred must be provided as well."
            )

        self.predictor_names = predictor_names
        n_preds = len(self.predictor_names)

        # ensuring format of numpy predictors
        if data_pred is not None:
            if data_pred.ndim == 1:
                data_pred = data_pred[:, np.newaxis]
            if data_pred.ndim > 2 or data_pred.shape[1] != n_preds:
                raise ValueError(
                    "data_pred must be 1D or a 2D array with shape (n_samples, n_preds), "
                    f"n_preds from `predictor_names` is {n_preds} but shape of data_pred is {data_pred.shape}."
                )

        # build dictionary
        # NOTE: extremely important that the order is the right one
        self.data_pred = {
            key: data_pred[:, i] for i, key in enumerate(self.predictor_names)
        }

        self.data_targ = data_targ

        # smooting to help with location & scale
        self.l_smooth = 5
        self.smooth_targ = _smooth_data(data_targ, length=self.l_smooth)
        self.smooth_targ_dev_sq = (
            data_targ[self.l_smooth : -self.l_smooth] - self.smooth_targ
        ) ** 2
        self.smooth_pred = {
            pp: _smooth_data(self.data_pred[pp], length=self.l_smooth)
            for pp in self.predictor_names
        }

        self.data_weights = data_weights

        # check first guess
        self.fg_coeffs = np.copy(first_guess)
        # make sure all values are floats bc if fg_coeff[ind] = type(int) we can only put ints in it too
        self.fg_coeffs = self.fg_coeffs.astype(float)

        # check if the first guess is valid
        if not len(self.fg_coeffs) == self.expression.n_coeffs:
            raise ValueError(
                "The provided first guess does not have the correct shape: "
                f"expected {self.expression.n_coeffs} (number of coeffs in expression)",
                f"got {len(self.fg_coeffs)}.",
            )

    # suppress nan & inf warnings
    @_ignore_warnings
    def _find_fg(self):
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

        # TODO split up into the several steps

        # Step 1: fit coefficients of location (objective: generate an adequate
        # first guess for the coefficients of location. proven to be necessary
        # in many situations, & accelerate step 2)
        fg_ind_loc = self.expression.ind_loc_coeffs

        # location might not be used (beta distribution) or set in the expression
        if len(fg_ind_loc) > 0 and len(self.predictor_names) > 0:

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

            derivative_targ = np.array(
                [
                    _finite_difference(
                        np.mean(self.smooth_targ[ind_targ_high]),
                        np.mean(self.smooth_targ[ind_targ_low]),
                        mean_high_preds[pp],
                        mean_low_preds[pp],
                    )
                    for pp in self.predictor_names
                ]
            )

            minimizer_kwargs = {
                "args": (
                    mean_high_preds,
                    mean_low_preds,
                    derivative_targ,
                    m_smooth_targ,
                )
            }

            if Version(sp.__version__) >= Version("1.15"):
                kwargs = {"rng": np.random.default_rng(SEED_BASINHOPPING)}
            else:
                kwargs = {"seed": np.random.default_rng(SEED_BASINHOPPING)}

            # TODO: do we move the basinhopping part to the class optimizers?
            globalfit_d01 = basinhopping(
                func=self._fg_fun_deriv01,
                x0=self.fg_coeffs[fg_ind_loc],
                niter=10,
                interval=100,
                minimizer_kwargs=minimizer_kwargs,
                **kwargs,
            )
            # warning, basinhopping tends to introduce non-reproductibility in fits,
            # reduced when using 2nd round of fits

            self.fg_coeffs[fg_ind_loc] = globalfit_d01.x

        # Step 2: fit coefficients of location (objective: improving the subset of
        # location coefficients)
        if len(fg_ind_loc) > 0:

            localfit_loc = _optimizers._minimize(
                func=self._fg_fun_loc,
                x0=self.fg_coeffs[fg_ind_loc],
                args=(),
                option_NelderMead="best_run",
                minimize_options=self.minimize_options,
            )
            self.fg_coeffs[fg_ind_loc] = localfit_loc.x

        # Step 3: fit coefficients of scale (objective: improving the subset of
        # scale coefficients)
        ind_scale = self.expression.ind_scale_coeffs
        # scale might not be used or set in the expression
        if len(ind_scale) > 0:
            x0 = self.fg_coeffs[ind_scale]

            localfit_scale = _optimizers._minimize(
                func=self._fg_fun_scale,
                x0=x0,
                args=(),
                minimize_options=self.minimize_options,
                option_NelderMead="best_run",
            )
            self.fg_coeffs[ind_scale] = localfit_scale.x

        # Step 4: fit other coefficients (objective: improving the subset of
        # other coefficients. May use multiple coefficients, eg beta distribution)
        # TODO: remove or find better loss function, see
        # - https://github.com/MESMER-group/mesmer/issues/582

        # if self.expression.ind_others.any():
        #     fg_ind_others = self.expression.ind_others

        #     localfit_others = _optimizers._minimize(
        #         func=self._fg_fun_others,
        #         x0=self.fg_coeffs[fg_ind_others],
        #         args=(),
        #         minimize_options=self.minimize_options,
        #         option_NelderMead="best_run",
        #     )
        #     self.fg_coeffs[fg_ind_others] = localfit_others.x

        # Step 5: fit coefficients using NLL (objective: improving all coefficients,
        # necessary to get good estimates for shape parameters, and avoid some local minima)
        localfit_nll = _optimizers._minimize(
            func=self._fg_fun_nll_no_tests,
            x0=self.fg_coeffs,
            args=(),
            minimize_options=self.minimize_options,
            option_NelderMead="best_run",
        )
        fg_coeffs = localfit_nll.x

        coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, _ = (
            _distrib_checks._validate_coefficients(
                self.expression,
                self.data_pred,
                self.data_targ,
                fg_coeffs,
                self.threshold_min_proba,
            )
        )

        # if any of validate_coefficients test fail (e.g. any of the coefficients are out of bounds)
        if not (
            coeffs_in_bounds and params_in_bounds and params_in_support and test_proba
        ):
            # Step 6: fit on LL^n (objective: improving all coefficients, necessary
            # to have all points within support. NB: NLL does not behave well enough here)

            localfit_opti = _optimizers._minimize(
                func=self._fg_fun_ll_n,
                x0=fg_coeffs,
                args=(),
                minimize_options=self.minimize_options,
                option_NelderMead="best_run",
            )
            if ~np.any(np.isnan(localfit_opti.x)):
                fg_coeffs = localfit_opti.x

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
        x_loc : numpy array
            Coefficients for the location
        pred_high : dict[str, numpy array]
            Predictors for the high samples of the targets
        pred_low : dict[str, numpy array]
            Predictors for the low samples of the targets
        derivative_targ : numpy array
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

        loc_low = self.expression._evaluate_one_param_fast(x, pred_low, "loc")
        loc_high = self.expression._evaluate_one_param_fast(x, pred_high, "loc")

        derivative_loc = np.array(
            [
                _finite_difference(loc_high, loc_low, pred_high[p], pred_low[p])
                for p in self.data_pred
            ]
        )
        diff = derivative_loc - derivative_targ
        return (
            # change of the location with the predictor should be similar to the change of the target with the predictor
            # np.sum((derivative_loc - derivative_targ) ** 2)
            np.dot(diff, diff)
            # location should not be too far from the mean of the samples
            + (0.5 * (loc_low + loc_high) - mean_targ) ** 2
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
        loc = self.expression._evaluate_one_param_fast(x, self.smooth_pred, "loc")

        if not _distrib_checks._param_in_bounds(self.expression, loc, "loc"):
            # this coefficient on location causes problem
            return np.inf

        # np.mean((loc - self.smooth_targ) ** 2)
        diff = loc - self.smooth_targ
        return np.dot(diff, diff)

    def _fg_fun_scale(self, x_scale):
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
        x[self.expression.ind_scale_coeffs] = x_scale
        scale = self.expression._evaluate_one_param_fast(x, self.data_pred, "scale")

        if not _distrib_checks._param_in_bounds(self.expression, scale, "scale"):
            # this coefficient on scale causes problem
            return np.inf

        return np.abs(np.mean(self.smooth_targ_dev_sq - scale**2))

    def _fg_fun_others(self, x_others, margin=1.0e-3):
        """
        Loss function for other coefficients than loc and scale. Objective is to tune parameters such
        that target samples are within a likely range of the distribution. Instead of relying on the
        support, we use the cumulative distribution function (CDF) to penalize unlikely samples.
        """
        # prepare coefficients
        x = np.copy(self.fg_coeffs)
        x[self.expression.ind_others] = x_others

        # evaluate parameters
        params = self.expression._evaluate_params_fast(x, self.data_pred)

        if not _distrib_checks._params_in_bounds(self.expression, params):
            # the coefficients cause problems
            return np.inf

        # compute CDF values for the target samples
        cdf_values = self.expression.distrib.cdf(self.data_targ, **params)

        if np.min(cdf_values) < margin or np.max(cdf_values) > 1 - margin:
            return np.inf

        # penalize samples with CDF values close to 0 or 1 (unlikely samples)
        penalty_low = np.maximum(0, margin - cdf_values) ** 2
        penalty_high = np.maximum(0, cdf_values - (1 - margin)) ** 2

        # sum penalties to compute the loss
        return penalty_low.sum() + penalty_high.sum()

    def _fg_fun_nll_no_tests(self, coefficients):
        params = self.expression._evaluate_params_fast(coefficients, self.data_pred)
        loss = _optimizers._neg_loglike(
            self.expression, self.data_targ, params, self.data_weights
        )
        return loss


    def _fg_fun_ll_n(self, x):
        # NOTE: n must be odd: https://github.com/MESMER-group/mesmer/issues/691
        n = 3
        params = self.expression._evaluate_params_fast(x, self.data_pred)

        if self.expression.is_distrib_discrete:
            loss = np.sum(self.expression.distrib.logpmf(self.data_targ, **params) ** n)
        else:
            loss = np.sum(self.expression.distrib.logpdf(self.data_targ, **params) ** n)

        if np.isnan(loss):
            return np.inf

        return -loss
