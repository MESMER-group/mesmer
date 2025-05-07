# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import properscoring
from scipy.optimize import minimize

from mesmer.mesmer_x._conditional_distribution import ConditionalDistribution, ConditionalDistributionOptions
from mesmer.mesmer_x._expression import Expression
import mesmer.mesmer_x._distrib_tests as distrib_tests

def _minimize(
    func, x0, conditional_distrib_options: ConditionalDistributionOptions, args=(), fact_maxfev_iter=1.0, option_NelderMead="dont_run",
):
    """
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
        method=conditional_distrib_options.method_fit,
        options={
            "maxfev": conditional_distrib_options.maxfev * fact_maxfev_iter,
            "maxiter": conditional_distrib_options.maxfev * fact_maxfev_iter,
            conditional_distrib_options.name_xtol: conditional_distrib_options.xtol_req,
            conditional_distrib_options.name_ftol: conditional_distrib_options.ftol_req,
        },
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
                "maxfev": conditional_distrib_options.maxfev * fact_maxfev_iter,
                "maxiter": conditional_distrib_options.maxiter * fact_maxfev_iter,
                "xatol": conditional_distrib_options.xtol_req,
                "fatol": conditional_distrib_options.ftol_req,
            },
        )
        if (option_NelderMead == "fail_run") or (
            option_NelderMead == "best_run"
            and (fit_NM.fun < fit.fun or not fit.success)
        ):
            fit = fit_NM
    return fit

# OPTIMIZATION FUNCTIONS & SCORES
def func_optim(conditional_distrib: ConditionalDistribution, coefficients, data_pred, data_targ, data_weights):
    # check whether these coefficients respect all conditions: if so, can compute a
    # value for the optimization
    distrib_options = conditional_distrib.options

    test_coeff, test_param, test_distrib, test_proba, params = (
        distrib_tests.validate_coefficients(conditional_distrib, data_pred, data_targ, coefficients)
    )

    if test_coeff and test_param and test_distrib and test_proba:
        if distrib_options.type_fun_optim == "fcnll":
            # compute full conditioning
            # will apply the stopping rule: splitting data_fit into two sets of data
            # using the given threshold
            ind_data_ok, ind_data_stopped = stopping_rule(conditional_distrib, data_targ, params)
            nll = neg_loglike(
                conditional_distrib.expression,
                data_targ[ind_data_ok],
                {pp: params[ind_data_ok] for pp in params},
                data_weights[ind_data_ok],
            )
            fc = fullcond_thres(
                conditional_distrib,
                data_targ[ind_data_stopped],
                {pp: params[ind_data_stopped] for pp in params},
                data_weights[ind_data_stopped],
            )
            return nll + fc
        elif distrib_options.type_fun_optim == "nll":
            # compute negative loglikelihood
            return neg_loglike(conditional_distrib.expression, data_targ, params, data_weights)

        else:
            raise Exception(
                f"Unknown type of optimization function: {distrib_options.type_fun_optim}"
            )
    else:
        # something wrong: returns a blocking value
        return np.inf

def neg_loglike(expression: Expression, data_targ, params, data_weights):
    return -loglike(expression, data_targ, params, data_weights)

def loglike(expression: Expression, data_targ, params, data_weights):
    # compute loglikelihood
    if expression.is_distrib_discrete:
        LL = expression.distrib.logpmf(data_targ, **params)
    else:
        LL = expression.distrib.logpdf(data_targ, **params)

    # weighted sum of the loglikelihood
    value = np.sum(data_weights * LL)

    if np.isnan(value):
        return -np.inf

    return value

def stopping_rule(conditional_distrib: ConditionalDistribution, data_targ, params):
    # evaluating threshold over time
    thres_t = conditional_distrib.expression.distrib.isf(
        q=1 / conditional_distrib.options.threshold_stopping_rule, **params # type: ignore
    )

    # selecting the minimum over the years to check
    thres = np.min(thres_t[conditional_distrib.options.ind_year_thres])

    # identifying where exceedances occur
    if conditional_distrib.options.exclude_trigger:
        ind_data_stopped = data_targ > thres
    else:
        ind_data_stopped = data_targ >= thres

    # identifying remaining positions
    ind_data_ok = ~ind_data_stopped
    return ind_data_ok, ind_data_stopped

def fullcond_thres(conditional_distrib: ConditionalDistribution, data_targ, params, data_weights):
    # calculating 2nd term for full conditional of the NLL
    # fc1 = distrib.logcdf(conditional_distrib.data_targ)
    fc2 = conditional_distrib.expression.distrib.sf(data_targ, **params)

    return np.log(np.sum((data_weights * fc2)[conditional_distrib.options.ind_stopped_data]))

def bic(conditional_distrib, data_targ, params, data_weights):
    loglike = conditional_distrib.options.loglike(data_targ, params, data_weights)
    n_coeffs = len(conditional_distrib.expr_fit.coefficients_list)
    return n_coeffs * np.log(len(data_targ)) - 2 * loglike

def crps(conditional_distrib, data_targ, data_pred, data_weights, coeffs):
    # properscoring.crps_quadrature cannot be applied on conditional distributions, thu
    # calculating in each point of the sample, then averaging
    # NOTE: WARNING, TAKES A VERY LONG TIME TO COMPUTE
    tmp_cprs = []
    for i in np.arange(len(data_targ)):
        distrib = conditional_distrib.expr_fit.evaluate(
            coeffs, {p: data_pred[p][i] for p in data_pred}
        )
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
