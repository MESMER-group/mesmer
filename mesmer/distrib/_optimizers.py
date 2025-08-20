# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/


import numpy as np
from scipy.optimize import minimize

from mesmer.distrib import _distrib_checks
from mesmer.distrib._expression import Expression


class MinimizeOptions:

    def __init__(
        self,
        method: str = "Nelder-Mead",
        tol: float | None = None,
        options: dict | None = None,
        # on_fail : Literal["error", "warn", "ignore"]="error",
    ):
        """options to pass to the minimizer - see scipy.optimize.minimize

        Parameters
        ----------
        method : str, default: "Nelder-Mead"
            Type of solver. See scipy.optimize.minimize
        tol : float | None, default: None
            Tolerance for termination. When tol is specified, the selected minimization
            algorithm sets some relevant solver-specific tolerance(s) equal to tol.
            For detailed control, use solver-specific options.
        options : dict | None, default None
            A dictionary of method-specific solver options. See scipy.optimize.minimize

        See also
        --------
        scipy.optimize.minimize
        """

        self.method = method
        self.tol = tol
        self.options = options


def _minimize(
    func,
    x0,
    args,
    minimize_options: MinimizeOptions,
    second_minimizer: MinimizeOptions | None = None,
):
    """custom minimize function."""

    fit = minimize(
        func,
        x0=x0,
        args=args,
        method=minimize_options.method,
        tol=minimize_options.tol,
        options=minimize_options.options,
    )

    # NOTE: observed that Powell solver is much faster, but less robust. May rarely
    # create directly NaN coefficients or wrong local optimum => Nelder-Mead can be used
    # at critical steps or when Powell fails.

    if second_minimizer is not None:
        second_fit = minimize(
            func,
            x0=x0,
            args=args,
            method=second_minimizer.method,
            tol=second_minimizer.tol,
            options=second_minimizer.options,
        )

        # if second_fit.success:
        # if the first fit failed or the second fit yields the better result
        if not fit.success or second_fit.fun < fit.fun:
            return second_fit

    return fit


# OPTIMIZATION FUNCTIONS & SCORES


class OptimizerNLL:
    """use negative log likelihood for optimization"""

    def _loss_function(self, expression, data_targ, params, data_weights):
        return _neg_loglike(expression, data_targ, params, data_weights)


def _optimization_function(
    optimizer: OptimizerNLL,
    data_pred,
    data_targ,
    data_weights,
    expression: Expression,
    threshold_min_proba,
):

    def _inner(coefficients):

        # check whether these coefficients respect all conditions: if so, can compute a
        # value for the optimization
        coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params = (
            _distrib_checks._validate_coefficients(
                expression, data_pred, data_targ, coefficients, threshold_min_proba
            )
        )

        if not (
            coeffs_in_bounds and params_in_bounds and params_in_support and test_proba
        ):
            # something wrong: returns a blocking value
            return np.inf

        return optimizer._loss_function(expression, data_targ, params, data_weights)

    return _inner


def _neg_loglike(expression: Expression, data_targ, params, data_weights):
    return -_loglike(expression, data_targ, params, data_weights)


def _loglike(expression: Expression, data_targ, params, data_weights):
    # compute loglikelihood
    if expression.is_distrib_discrete:
        LL = expression.distrib.logpmf(data_targ, **params)
    else:
        LL = expression.distrib.logpdf(data_targ, **params)

    # weighted sum of the loglikelihood
    # value = np.sum(data_weights * LL)
    value = np.dot(data_weights, LL)

    if np.isnan(value):
        return -np.inf

    return value


def _bic(expression, data_targ, params, data_weights):
    ll = _loglike(expression, data_targ, params, data_weights)
    n_coeffs = expression.n_coeffs
    return n_coeffs * np.log(len(data_targ)) - 2 * ll


def _crps(expression: Expression, data_targ, data_pred, data_weights, coeffs):

    try:
        import properscoring
    except ImportError:  # pragma: no cover
        msg = (
            "Computing the 'crps' metric requires the properscoring package to be"
            " installed"
        )
        raise ImportError(msg)

    # properscoring.crps_quadrature cannot be applied on conditional distributions, thus
    # calculating in each point of the sample, then averaging
    # NOTE: TAKES A VERY LONG TIME TO COMPUTE
    # TODO: find alternative way to compute this

    tmp_cprs = []
    for i in np.arange(len(data_targ)):
        distrib = expression.evaluate(coeffs, {p: data_pred[p][i] for p in data_pred})
        tmp_cprs.append(
            properscoring.crps_quadrature(
                x=data_targ[i],
                cdf_or_dist=distrib,
                xmin=-10 * np.abs(data_targ[i]),
                xmax=10 * np.abs(data_targ[i]),
                tol=1.0e-4,
            )
        )

    # averaging
    return np.sum(data_weights * np.array(tmp_cprs))
