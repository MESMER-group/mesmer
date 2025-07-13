# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

from typing import Literal

import numpy as np
import properscoring
from scipy.optimize import minimize

import mesmer.mesmer_x._distrib_checks as _distrib_checks
from mesmer.mesmer_x._expression import Expression


class MinimizeOptions:

    def __init__(
        self,
        method: str = "Powell",
        tol: float | None = None,
        options: dict | None = None,
        # on_fail : Literal["error", "warn", "ignore"]="error",
    ):
        """options to pass to the minimizer - see scipy.optimize.minimize

        Parameters
        ----------
        method : str
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
):
    """custom minimize function.
    """

    fit = minimize(
        func,
        x0=x0,
        args=args,
        method=minimize_options.method,
        tol=minimize_options.tol,
        options=minimize_options.options,
    )

    # NOTE: observed that Powell solver is much faster, but less robust. May rarely create
    # directly NaN coefficients or wrong local optimum => Nelder-Mead can be used at
    # critical steps or when Powell fails.

    # TODO: re-add second optimizer: https://github.com/MESMER-group/mesmer/issues/746

    return fit


# OPTIMIZATION FUNCTIONS & SCORES


class OptimizerNLL:
    """use negative log likelihood for optimization"""

    def _loss_function(self, expression, data_targ, params, data_weights):
        return _neg_loglike(expression, data_targ, params, data_weights)


class OptimizerFCNLL:
    """
    use full conditional negative log likelihood based on the stopping rule for
    optimization

    Parameters
    ----------
    * threshold_stopping_rule: float > 1
      Maximum return period, used to define the threshold of the stopping rule.

    * ind_year_thres: np.array
        Positions in the predictors where the thresholds have to be tested.

    * exclude_trigger: boolean
        Whether the threshold will be included or not in the stopping rule.

    Notes
    -----
    Source: https://doi.org/10.1016/j.wace.2023.100584
    """

    def __init__(self, threshold_stopping_rule, ind_year_thres, exclude_trigger):

        if threshold_stopping_rule < 1:
            msg = (
                "`threshold_stopping_rule` must be larger than 1, got"
                f" '{threshold_stopping_rule}'"
            )
            raise ValueError(msg)

        self.threshold_stopping_rule = threshold_stopping_rule
        self.ind_year_thres = ind_year_thres
        self.exclude_trigger = exclude_trigger

    def _loss_function(self, expression, data_targ, params, data_weights):
        # compute full conditioning
        # will apply the stopping rule: splitting data_fit into two sets of data
        # using the given threshold
        ind_data_ok, ind_data_stopped = _stopping_rule(
            expression,
            data_targ,
            params,
            self.threshold_stopping_rule,
            self.ind_year_thres,
            self.exclude_trigger,
        )

        params_ok, params_stopped = {}, {}
        for key in params:

            param = params[key]

            # if the parameter is a scalar, it is the same for all data points
            if params[key].shape != ind_data_ok.shape:
                param = np.full_like(ind_data_ok, fill_value=params[key], dtype=float)

            params_ok[key] = param[ind_data_ok]
            params_stopped[key] = param[ind_data_stopped]

        nll = _neg_loglike(
            expression,
            data_targ[ind_data_ok],
            params_ok,
            data_weights[ind_data_ok],
            # TODO: do we need different code depending on exclude_trigger?
            # https://github.com/MESMER-group/mesmer/issues/729
        )
        fc = _fullcond_thres(
            expression,
            data_targ[ind_data_stopped],
            params_stopped,
            data_weights[ind_data_stopped],
        )
        return nll + fc


def _optimization_function(
    optimizer: OptimizerNLL | OptimizerFCNLL,
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


def _stopping_rule(
    expression: Expression,
    data_targ,
    params,
    threshold_stopping_rule,
    ind_year_thres,
    exclude_trigger,
):
    """
    Identifies parts of the sample above the threshold for the stopping rule.

    Parameters
    ----------
    data_targ : numpy array 1D
        Target data to evaluate the stopping rule.
    params : dict
        Parameters of the distribution, evaluated for the target data.

    Returns
    -------
    ind_data_ok : numpy array 1D boolean
        Indices of the data that are below the threshold for the stopping rule.
    ind_data_stopped : numpy array 1D boolean
        Indices of the data that are above the threshold for the stopping rule.

    Notes
    -----
    Source: https://doi.org/10.1016/j.wace.2023.100584
    """

    # evaluating threshold over time
    thres_t = expression.distrib.isf(
        q=1 / threshold_stopping_rule, **params  # type: ignore
    )

    # selecting the minimum over the years to check
    thres = np.min(thres_t[ind_year_thres])

    # identifying where exceedances occur
    if exclude_trigger:
        ind_data_stopped = data_targ > thres
    else:
        ind_data_stopped = data_targ >= thres

    # identifying remaining positions
    ind_data_ok = ~ind_data_stopped
    return ind_data_ok, ind_data_stopped


def _fullcond_thres(expression: Expression, data_targ, params, data_weights):
    """
    Calculates the full conditional of the negative log likelihood, based on the
    stopping rule. This is the second term of the full conditional negative log
    likelihood (FCNLL) used in the optimization.

    Parameters
    ----------
    data_targ : numpy array 1D
        Target data to evaluate the full conditional.
    params : dict
        Parameters of the distribution, evaluated for the target data.
    data_weights : numpy array 1D
        Weights for the target data.

    Returns
    -------
    float
        The value of the full conditional of the negative log likelihood.

    Notes
    -----
    Source: https://doi.org/10.1016/j.wace.2023.100584
    """

    # calculating 2nd term for full conditional of the NLL
    # fc1 = distrib.logcdf(conditional_distrib.data_targ)
    fc2 = expression.distrib.sf(data_targ, **params)

    # TODO: can fc1 be left away?
    # TODO: it probably should be sum(log()) but the paper code uses log(sum())
    # https://github.com/MESMER-group/mesmer/issues/729

    # return np.sum(data_weights * np.log(fc2))
    return np.log(np.sum(data_weights * fc2))


def _bic(expression, data_targ, params, data_weights):
    ll = _loglike(expression, data_targ, params, data_weights)
    n_coeffs = expression.n_coeffs
    return n_coeffs * np.log(len(data_targ)) - 2 * ll


def _crps(expression: Expression, data_targ, data_pred, data_weights, coeffs):
    # properscoring.crps_quadrature cannot be applied on conditional distributions, thu
    # calculating in each point of the sample, then averaging
    # NOTE: WARNING, TAKES A VERY LONG TIME TO COMPUTE
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
