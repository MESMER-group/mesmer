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


def _minimize(
    func,
    x0,
    method_fit,
    args,
    options,
    option_NelderMead: Literal["dont_run", "fail_run", "best_run"] = "dont_run",
):
    """
    custom minimize function.

    First tries with method_fit.

    A second fit is performed either if
    - the first fit failed and option_NelderMead is set to "fail_run"
    - or if option_NelderMead is set to "best_run", regardless if the first fit succeded.

    The result of the second fit is only returned if it is better than the first fit.

    options_NelderMead: str
        * dont_run: would minimize only the chosen solver in method_fit
        * fail_run: would minimize using Nelder-Mead only if the chosen solver in
            method_fit fails
        * best_run: will minimize using Nelder-Mead and the chosen solver in
            method_fit, then select the best results
    """
    fit = minimize(
        func,
        x0=x0,
        args=args,
        method=method_fit,
        options=options,
    )

    # observed that Powell solver is much faster, but less robust. May rarely create
    # directly NaN coefficients or wrong local optimum => Nelder-Mead can be used at
    # critical steps or when Powell fails.

    if (option_NelderMead == "fail_run" and not fit.success) or (
        option_NelderMead == "best_run"
    ):
        fit_NM = minimize(
            func,
            x0=x0,
            args=args,
            method="Nelder-Mead",
            options={
                "maxfev": options["maxfev"],
                "maxiter": options["maxiter"],
                "xatol": list(options.values())[2],
                "fatol": list(options.values())[3],
            },
        )
        if (option_NelderMead == "fail_run") or (
            option_NelderMead == "best_run"
            and (fit_NM.fun < fit.fun or not fit.success)
        ):
            fit = fit_NM
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
