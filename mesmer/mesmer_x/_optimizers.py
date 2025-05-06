# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import properscoring
import xarray as xr
from scipy.optimize import minimize

from mesmer.mesmer_x.train_utils_mesmerx import (
    Expression,
)



class distrib_optimizer:
    def __init__(
        self,
        expr_fit: Expression,
        class_tests: distrib_tests,
        options_optim=None,  # TODO: replace by options class?
        options_solver=None,  # TODO: ditto?
    ):
        """Class to define optimizers used during first guess and training.

        Parameters
        ----------
        expr_fit : class 'expression'
            Expression to train. The string provided to the class can be found in
            'expr_fit.expression'.

        class_tests : class 'distrib_tests'
            Class defining the tests to perform during first guess and training

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
        """
        # initialization
        self.expr_fit = expr_fit
        self.class_tests = class_tests
        self.n_coeffs = len(self.expr_fit.coefficients_list)

        # preparing solver
        default_options_solver = {
            "method_fit": "Powell",
            "xtol_req": 1e-6,
            "ftol_req": 1.0e-6,
            "maxiter": 1000 * self.n_coeffs * (np.log(self.n_coeffs) + 1),
            "maxfev": 1000 * self.n_coeffs * (np.log(self.n_coeffs) + 1),
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

    # FLEXIBLE MINIMIZER
    def _minimize(
        self, func, x0, args=(), fact_maxfev_iter=1.0, option_NelderMead="dont_run"
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
            method=self.method_fit,
            options={
                "maxfev": self.maxfev * fact_maxfev_iter,
                "maxiter": self.maxfev * fact_maxfev_iter,
                self.name_xtol: self.xtol_req,
                self.name_ftol: self.ftol_req,
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
                    "maxfev": self.maxfev * fact_maxfev_iter,
                    "maxiter": self.maxiter * fact_maxfev_iter,
                    "xatol": self.xtol_req,
                    "fatol": self.ftol_req,
                },
            )
            if (option_NelderMead == "fail_run") or (
                option_NelderMead == "best_run"
                and (fit_NM.fun < fit.fun or not fit.success)
            ):
                fit = fit_NM
        return fit

    # OPTIMIZATION FUNCTIONS & SCORES
    def func_optim(self, coefficients, data_pred, data_targ, data_weights):
        # check whether these coefficients respect all conditions: if so, can compute a
        # value for the optimization
        test_coeff, test_param, test_distrib, test_proba, params = (
            self.class_tests.validate_coefficients(data_pred, data_targ, coefficients)
        )

        if test_coeff and test_param and test_distrib and test_proba:
            if self.type_fun_optim == "fcnll":
                # compute full conditioning
                # will apply the stopping rule: splitting data_fit into two sets of data
                # using the given threshold
                ind_data_ok, ind_data_stopped = self.stopping_rule(data_targ, params)
                nll = self.neg_loglike(
                    data_targ[ind_data_ok],
                    {pp: params[ind_data_ok] for pp in params},
                    data_weights[ind_data_ok],
                )
                fc = self.fullcond_thres(
                    data_targ[ind_data_stopped],
                    {pp: params[ind_data_stopped] for pp in params},
                    data_weights[ind_data_stopped],
                )
                return nll + fc
            elif self.type_fun_optim == "nll":
                # compute negative loglikelihood
                return self.neg_loglike(data_targ, params, data_weights)

            else:
                raise Exception(
                    f"Unknown type of optimization function: {self.type_fun_optim}"
                )
        else:
            # something wrong: returns a blocking value
            return np.inf

    def neg_loglike(self, data_targ, params, data_weights):
        return -self.loglike(data_targ, params, data_weights)

    def loglike(self, data_targ, params, data_weights):
        # compute loglikelihood
        if self.expr_fit.is_distrib_discrete:
            LL = self.expr_fit.distrib.logpmf(data_targ, **params)
        else:
            LL = self.expr_fit.distrib.logpdf(data_targ, **params)

        # weighted sum of the loglikelihood
        value = np.sum(data_weights * LL)

        if np.isnan(value):
            return -np.inf

        return value

    def stopping_rule(self, data_targ, params):
        # evaluating threshold over time
        thres_t = self.expr_fit.distrib.isf(
            q=1 / self.threshold_stopping_rule, **params
        )

        # selecting the minimum over the years to check
        thres = np.min(thres_t[self.ind_year_threshold])

        # identifying where exceedances occur
        if self.exclude_trigger:
            ind_data_stopped = data_targ > thres
        else:
            ind_data_stopped = data_targ >= thres

        # identifying remaining positions
        ind_data_ok = ~ind_data_stopped
        return ind_data_ok, ind_data_stopped

    def fullcond_thres(self, data_targ, params, data_weights):
        # calculating 2nd term for full conditional of the NLL
        # fc1 = distrib.logcdf(self.data_targ)
        fc2 = self.expr_fit.distrib.sf(data_targ, **params)

        return np.log(np.sum((data_weights * fc2)[self.ind_stopped_data]))

    def bic(self, data_targ, params, data_weights):
        loglike = self.loglike(data_targ, params, data_weights)
        n_coeffs = len(self.expr_fit.coefficients_list)
        return n_coeffs * np.log(len(data_targ)) - 2 * loglike

    def crps(self, data_targ, data_pred, data_weights, coeffs):
        # properscoring.crps_quadrature cannot be applied on conditional distributions, thu
        # calculating in each point of the sample, then averaging
        # NOTE: WARNING, TAKES A VERY LONG TIME TO COMPUTE
        tmp_cprs = []
        for i in np.arange(len(data_targ)):
            distrib = self.expr_fit.evaluate(
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
