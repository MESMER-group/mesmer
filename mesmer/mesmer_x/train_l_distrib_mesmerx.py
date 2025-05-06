# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Refactored code for the training of distributions

"""

import functools
import warnings

import numpy as np
import properscoring
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

# TODO: to replace with outputs from PR #607
from mesmer.core.datatree import collapse_datatree_into_dataset
from mesmer.core.geospatial import geodist_exact
from mesmer.mesmer_x.train_utils_mesmerx import (
    Expression,
    weighted_median,
)
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

def get_weights_uniform(targ_data, target, dims):
    """
    Generate uniform weights for the training sample.

    Parameters
    ----------
    targ_data : xr.DataTree | xr.Dataset | np.array
        Target for the training sample. Each branch must be a scenario,
        with a xarray dataset (time, member, gridpoint).

    target : str
        Name of the target. Must be the name in the datasets in targ_data.

    dims : list of str
        Dimensions of the data. Must be the same for all scenarios.

    Returns
    -------
    weights : DataTree
        Weights for the sample, uniform, summing to 1.

    Example
    -------
    TODO

    """
    if isinstance(targ_data, xr.DataTree):
        # preparing a datatree with ones everywhere
        factor_rescale = 0
        out = dict()
        for scen in targ_data:
            # identify the extra dimension
            extra_dims = [
                dim for dim in targ_data[scen][target].dims if dim not in dims
            ]
            locator_extra_dims = {dim: 0 for dim in extra_dims}

            # create a DataArray of ones with the required shape
            ones_array = xr.ones_like(
                targ_data[scen][target].loc[locator_extra_dims], dtype=float
            )
            out[scen] = xr.DataTree(xr.Dataset({"weight": ones_array}))

            # accumulate the total size for rescaling
            factor_rescale += ones_array.size

        # rescale
        return xr.DataTree.from_dict(out) / factor_rescale

    elif isinstance(targ_data, xr.Dataset):
        # identify the extra dimension
        extra_dims = [dim for dim in targ_data[target].dims if dim not in dims]
        locator_extra_dims = {dim: 0 for dim in extra_dims}

        # create a DataArray of ones with the required shape
        ones_array = xr.ones_like(
            targ_data[target].loc[locator_extra_dims], dtype=float
        )

        # rescale
        return xr.Dataset({"weight": ones_array / ones_array.size})

    elif isinstance(targ_data, np.ndarray):
        # create a DataArray of ones with the required shape
        # warning, it assumes that this is performed for a single gridpoint
        ones_array = np.ones(targ_data.shape, dtype=float)

        # rescale
        return ones_array / ones_array.size

    else:
        raise Exception(
            "The format for targ_data must be a xr.DataTree, xr.Dataset or a np.array."
        )


def get_weights_density(pred_data, predictor, targ_data, target, dims):
    """
    Generate weights for the sample, based on the inverse of the density of the
    predictors. More precisely, the density of the predictors is represented by a
    multidimensional kernel density estimate using gaussian kernels where each
    dimension is one of the predictors. Subsequently, the weights are the inverse
    of this density of the predictors. Consequently, samples in regions of this
    space with low density will have higher weights, this is, "unusual" samples
    will have more weight.

    Parameters
    ----------
    pred_data : xr.DataTree | xr.Dataset | np.array
        Predictors for the training sample. Each branch must be a scenario,
        with a xarray dataset (time, member). Each predictor is a variable.

    predictor : str
        Name of the predictor. Must be the name in the datasets in pred_data.

    targ_data : DataTree
        Target for the training sample. Each branch must be a scenario,
        with a xarray dataset (time, member, gridpoint).

    target : str
        Name of the target. Must be the name in the datasets in targ_data.

    dims : list of str
        Dimensions of the data. Must be the same for all scenarios.

    Returns
    -------
    weights : DataTree
        Weights for the sample, based on the inverse of the density of the
        predictors, summing to 1.

    Example
    -------
    TODO

    """

    # checking if predictors have been provided
    if len(pred_data) == 0:
        # NB: may use no predictors when training stationary distributions for bencharmking.
        print("no predictors provided, switching to uniform weights")
        return get_weights_uniform(targ_data, target, dims)

    elif isinstance(targ_data, xr.DataTree):
        # reshaping data for histogram
        tmp_pred = {}
        for var in pred_data:
            if var not in tmp_pred:
                tmp_pred[var] = np.array([])
            for scen in pred_data[var]:
                tmp_pred[var] = np.concatenate(
                    [tmp_pred[var], pred_data[var][scen][predictor].values.flatten()]
                )
        array_pred = np.array(list(tmp_pred.values()))

        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd that implies too many assumptions
        histo_kde = gaussian_kde(array_pred)

        # calculating density of points over the sample
        density = histo_kde.pdf(x=array_pred)

        # preparing the datatree
        weight, counter, factor_rescale = dict(), 0, 0
        # using former var, ensuring correct order on dimensions
        dims = pred_data[var][scen][predictor].dims
        for scen in pred_data[var]:
            # reshaping the weights for this scenario
            n_dims = {dim: pred_data[var][scen][dim].size for dim in dims}
            array_tmp = np.reshape(
                density[counter : counter + pred_data[var][scen][predictor].size],
                [n_dims[dim] for dim in dims],
            )
            tmp = xr.DataArray(
                data=array_tmp,
                dims=dims,
                coords={dim: pred_data[var][scen][dim] for dim in dims},
            )

            # inverse of density
            weight[scen] = xr.Dataset({"weight": 1 / tmp})
            factor_rescale += weight[scen]["weight"].sum()

            # preparing next scenario
            counter += pred_data[var][scen][predictor].size

        # preparing the output
        return xr.DataTree.from_dict(weight) / factor_rescale

    elif isinstance(targ_data, xr.Dataset):
        # reshaping data for histogram
        array_pred = pred_data.to_array().values

        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd that implies too many assumptions
        histo_kde = gaussian_kde(array_pred)

        # calculating density of points over the sample
        density = histo_kde.pdf(x=array_pred)

        # preparing the output
        return xr.Dataset({"weight": (1 / density) / np.sum(1 / density)})

    elif isinstance(targ_data, np.ndarray):
        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd that implies too many assumptions
        histo_kde = gaussian_kde(pred_data)

        # calculating density of points over the sample
        density = histo_kde.pdf(x=pred_data)

        # preparing the output
        return (1 / density) / np.sum(1 / density)

    else:
        raise Exception(
            "The format for targ_data must be a xr.DataTree, xr.Dataset or a np.array."
        )


class distrib_tests:
    def __init__(
        self,
        expr_fit: Expression,
        threshold_min_proba=1.0e-9,
        boundaries_params=None,
        boundaries_coeffs=None,
    ):
        """Class defining the tests to perform during first guess and training of distributions.

        Parameters
        ----------
        expr_fit : class 'expression'
            Expression to train. The string provided to the class can be found in
            'expr_fit.expression'.

        threshold_min_proba : float or None, default: 1e-9
            If numeric imposes a check during the fitting that every sample fulfills
            `cdf(sample) >= threshold_min_proba and 1 - cdf(sample) >= threshold_min_proba`,
            i.e. each sample lies within some confidence interval of the distribution.
            Note that it follows that threshold_min_proba math::\\in (0,0.5). Important to
            ensure that all points are feasible with the fitted distribution.
            If `None` this test is skipped.

        boundaries_params : dict, default: None
            Prescribed boundaries on the parameters of the expression. Some basic
            boundaries are already provided through 'expr_fit.boundaries_params'.

        boundaries_coeffs : dict, optional
            Prescribed boundaries on the coefficients of the expression. Default: None.

        """
        # initialization of expr_fit
        self.expr_fit = expr_fit

        # initialization and basic checks on threshold_min_proba
        self.threshold_min_proba = threshold_min_proba
        if threshold_min_proba is not None and (
            (threshold_min_proba <= 0) or (0.5 <= threshold_min_proba)
        ):
            raise ValueError("`threshold_min_proba` must be in (0, 0.5)")

        # initialization and basic checks on boundaries
        self.boundaries_params = self.expr_fit.boundaries_parameters
        if boundaries_params is not None:
            for param in boundaries_params:
                lower_bound = np.max(
                    [boundaries_params[param][0], self.boundaries_params[param][0]]
                )
                upper_bound = np.min(
                    [boundaries_params[param][1], self.boundaries_params[param][1]]
                )
                self.boundaries_params[param] = [lower_bound, upper_bound]
        self.boundaries_coeffs = {} if boundaries_coeffs is None else boundaries_coeffs

    def _test_coeffs_in_bounds(self, values_coeffs):

        # checking set boundaries on coefficients
        for coeff in self.boundaries_coeffs:
            bottom, top = self.boundaries_coeffs[coeff]

            # TODO: move this check to __init__ NOTE: also used in fg
            if coeff not in self.expr_fit.coefficients_list:
                raise ValueError(
                    f"Provided wrong boundaries on coefficient, {coeff}"
                    " does not exist in expr_fit"
                )

            values = values_coeffs[self.expr_fit.coefficients_list.index(coeff)]

            if np.any(values < bottom) or np.any(top < values):
                # out of boundaries
                return False

        return True

    def _test_evol_params(self, params):

        # checking set boundaries on parameters
        for param in self.boundaries_params:
            bottom, top = self.boundaries_params[param]

            param_values = params[param]

            # out of boundaries
            if np.any(param_values < bottom) or np.any(param_values > top):
                return False

        return True

    def _test_support(self, params, data):
        # test of the support of the distribution: is there any data out of the
        # corresponding support? dont try testing if there are issues on the parameters

        bottom, top = self.expr_fit.distrib.support(**params)

        # out of support
        if (
            np.any(np.isnan(bottom))
            or np.any(np.isnan(top))
            or np.any(data < bottom)
            or np.any(data > top)
        ):
            return False

        return True

    def _test_proba_value(self, params, data):
        """
        Test that all cdf(data) >= threshold_min_proba and 1 - cdf(data) >= threshold_min_proba
        Ensures that data lies within a confidence interval of threshold_min_proba for the tested
        distribution.
        """
        # NOTE: DONT write 'x=data', because 'x' may be called differently for some
        # distribution (eg 'k' for poisson).

        cdf = self.expr_fit.distrib.cdf(data, **params)
        thresh = self.threshold_min_proba
        return np.all(1 - cdf >= thresh) and np.all(cdf >= thresh)

    def validate_coefficients(self, data_pred, data_targ, coefficients):
        """validate coefficients

        Parameters
        ----------
        coefficients : numpy array 1D
            Coefficients to validate.

        data_pred : numpy array 1D
            Predictors for the training sample.

        data_targ : numpy array 1D
            Target for the training sample.

        Returns
        -------
        test_coeff : boolean
            True if the coefficients are within self.boundaries_coeffs. If
            False, all other tests will also be set to False and not tested.

        test_param : boolean
            True if parameters are within self.boundaries_params and within the support of the distribution.
            False if not or if test_coeff is False. If False, test_proba will be set to False and not tested.

        test_proba : boolean
            Only tested if self.threshold_min_proba is not None.
            True if the probability of the target samples for the given coefficients
            is above self.threshold_min_proba.
            False if not or if test_coeff or test_param or test_coeff is False.

        distrib : distrib_cov
            The distribution that has been evaluated for the given coefficients.

        """

        test_coeff = self._test_coeffs_in_bounds(coefficients)

        # tests on coeffs show already that it won't work: fill in the rest with False
        if not test_coeff:
            return test_coeff, False, False, False, False

        # evaluate the distribution for the predictors and this iteration of coeffs
        params = self.expr_fit.evaluate_params(coefficients, data_pred)
        # test for the validity of the parameters
        test_param = self._test_evol_params(params)

        # tests on params show already that it won't work: fill in the rest with False
        if not test_param:
            return test_coeff, test_param, False, False, False

        # test for the support of the distribution
        test_support = self._test_support(params, data_targ)

        # tests on params show already that it won't work: fill in the rest with False
        if not test_support:
            return test_coeff, test_param, test_support, False, False

        # test for the probability of the values
        if self.threshold_min_proba is None:
            return test_coeff, test_param, test_support, True, params

        else:
            test_proba = self._test_proba_value(params, data_targ)

            # return values for each test and the evaluated distribution
            return test_coeff, test_param, test_support, test_proba, params

    def get_var_data(self, data):
        if isinstance(data, np.ndarray):
            return data

        elif isinstance(data, xr.DataArray):
            return data

        elif isinstance(data, xr.Dataset):
            var_name = [var for var in data.variables][0]
            return data[var_name]

        elif isinstance(data, xr.DataTree):
            # TODO: useless, datatree uses datasets anyway, so it will become a dataarray
            new_data = xr.DataTree()
            for pred in data:
                var_name = [var for var in data[pred].variables][0]
                _ = xr.DataTree(name=pred, parent=new_data, data=data[pred][var_name])
            return new_data

        else:
            raise ValueError("data must be a DataArray, Dataset or DataTree")

    def validate_data(self, data_pred, data_targ, data_weights):
        """validate data

        Parameters
        ----------
        data_pred
            Predictors for the training sample.

        data_targ
            Target for the training sample.

        data_weights
            Weights for the training sample.
        -------
        """
        # basic checks on data_targ
        self.check_data(data_targ, "target")

        # basic checks on data_pred
        self.check_data(data_pred, "predictors")

        # basic checks on weights
        self.check_data(data_weights, "weights")

    def check_data(self, data, name):
        """
        basic check data
        """
        # getting variable. useful if calling _fit_np | _find_fg_np with wrong format
        data = self.get_var_data(data)

        # checking for NaN values
        if np.isnan(data).any():
            raise ValueError(f"nan values in {name}")

        # checking for infinite values
        if np.isinf(data).any():
            raise ValueError(f"infinite values in {name}")

    def prepare_data(self, predictors, target, weights):
        """
        shaping data for first guess, training or evaluation of scores.

        Parameters
        ----------
        predictors : dict of xr.DataArray or xr.Dataset | xr.Dataset | xr.DataTree
            Predictors for the first guess. Must either be a dictionary of xr.DataArray or
            xr.Dataset, each key/item being a predictor; a xr.Dataset with a coordinate
            being the list of predictors, and a variable that contains all predictors; or
            a xr.DataTree with one branch per predictor.
        target : xr.DataArray | xr.Dataset
            Target DataArray.
        weights : xr.DataArray | xr.Dataset
            Individual weights for each sample.

        Returns
        -------
        :data_pred:`xr.Dataset`
            shaped predictors for training (gridpoint, coefficient)
        :data_targ:`xr.Dataset`
            shaped sample for training (gridpoint, coefficient)
        :data_weights:`xr.Dataset`
            shaped weights for training (gridpoint, coefficient)
        """
        # check format of predictors
        if isinstance(predictors, dict):
            tmp = {
                key: self.class_tests.get_var_data(predictors[key])
                for key in predictors
            }
            ds_pred = xr.Dataset(tmp)

        elif isinstance(predictors, xr.Dataset):
            if "predictor" not in predictors.coords:
                raise Exception(
                    "If predictors are provided as xr.Dataset, it must contain a coordinate 'predictor'."
                )

        elif isinstance(predictors, xr.DataTree):
            # preparing predictors
            ds_pred = collapse_datatree_into_dataset(predictors, dim="predictor")

        else:
            raise Exception(
                "predictors is supposed to be a dict of xr.DataArray, xr.Dataset or xr.DataTree"
            )

        # check format of target
        if not (isinstance(target, xr.Dataset) or isinstance(target, xr.DataArray)):
            raise Exception("the target must be a xr.Dataset or xr.DataArray.")

        # check format of weights
        if not (isinstance(weights, xr.Dataset) or isinstance(weights, xr.DataArray)):
            raise Exception("the weights must be a xr.Dataset or xr.DataArray.")

        # getting just dataarray in the datasets
        data_pred = self.get_var_data(ds_pred)
        data_targ = self.get_var_data(target)
        data_weights = self.get_var_data(weights)

        return data_pred, data_targ, data_weights


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


class distrib_train:
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
