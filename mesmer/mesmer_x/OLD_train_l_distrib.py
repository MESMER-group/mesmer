# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train local distributions module of MESMER.
This script is adapted from the script of Mathias Hauser ("/net/cfc/landclim1/mathause/projects/Siberia_2020/code/utils/extreme_event.py")

NB: existence of this package: https://github.com/OpheliaMiralles/pykelihood
Very efficient for simple conditional distributions, but not as robust as here.
Demonstration:
from pykelihood.distributions import Normal
from pykelihood import kernels
loc_0, loc_T = -1, 1.5
scale_0, scale_T = 1, 0.1
size = 100
T = np.linspace( 0, 5, size )
data = ss.norm.rvs( loc=loc_0+loc_T*T, scale=scale_0+scale_T*T, size=size )
Normal.fit(data, loc=kernels.linear(T), scale=kernels.linear(T)) # --> fails.

# test speed:
from pykelihood.distributions import GEV
from pykelihood import kernels
GEV.fit(data, loc=kernels.linear(tmp_preds['cov_loc'][0][1])) # --> fails.
----> 0.20s for that, while my code runs in 0.27s

# test robustness:
with linear term on location, scale & shape of GEV, fails in getting correct coefficients
===================
Method for fitting a distribution with parameters varying with external drivers

Classes:
    distrib_cov()

Functions:
    train_l_distrib()
    transf_distrib2normal()

Functions REMOVED from the former script:
    glm_mcmc()

"""

from math import e

import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
from scipy.special import gamma, zeta
from statsmodels.regression.linear_model import OLS

from mesmer.utils import (
    eval_param_distrib,
    read_form_fit_distrib,
    sigmoid_backtransf,
    sigmoid_transf,
)


def train_l_distrib(preds, targs, cfg, form_fit_distrib, save_params=True, **kwargs):
    """Derive local parameters for a covariated distribution.

    Parameters
    ----------
    preds : dict
        dictionnary[target][name_covariant] = covariant. The covariants must be dictionnaries with scenarios as keys.

    targs : dict
        nested dictionary of targets with keys
        - [targ][scen] (3d array (run, time, gp) of target for specific scenario)

    cfg : module
        config file containing metadata

    form_fit_distrib : str
        string containing information on which fit to do: transformation and evolution of parameters
        Examples: transfo-_loc-gttasL-gthfdsL_scale-_shape-gttasS; transfo-logistic_loc-gttasL_scale-

    save_params : bool, optional
        determines if parameters are saved or not, default = True

    **kwargs : additional parameters that will be fed to the class 'distrib_cov' for fit of a distribution

    Returns
    -------
    params_l_distrib : dict
        nested dictionary of local variability paramters. For each target of targs,

        - ["parameters"] (parameters for the fit, dict)
        - ["distribution"] (fitted distribution, str)
        - ["cov_loc"] (covariants for location, list of strs)
        - ["cov_scale"] (covariants for scale, list of strs)
        - ["cov_shape"] (covariants for shape, list of strs)
        - ["scenarios_training"] (list of scenarios used for training, list of strs)
        - ["nruns_training"] (number of runs used for training, int)
        - ["quality_fit"] (value at each gridpoint of the minimization function of the fit (negative log likelihood), numpy array of float)

    Notes
    -----
    - Assumptions:
        If different number of runs are provided between target and covariants, the first ones are used, assuming they are the same.
    - Disclaimer:
    - TODO:
        Pass everything in xarray, especially to handle dimensions 'runs' and 'scenarios'
        Write more explicitely on typo of form_fit_distrib.
        Could simplify preds to have only one input for each variable, even if shared among parameters, handle explicitely things in _init_ of distrib_cov

    """

    # dictionary that will be filled in for every parameter
    params_out = {var_targ: {} for var_targ in targs}

    # looping on targets to fit a distrib on.
    for var_targ in targs:

        #  Reading form_fit_distrib to prepare complete description of the fit
        descrip_fit = read_form_fit_distrib(form_fit_distrib)

        # selecting aspects specific to this variable
        data = {scen: np.copy(targs[var_targ][scen][...]) for scen in targs[var_targ]}
        distr = cfg.methods[var_targ]["l_distrib"]

        # directly getting the list of the parameters of the distribution given:
        if distr in ["gaussian"]:
            tmp_list = ["loc", "scale"]
        elif distr in ["GEV"]:
            tmp_list = ["loc", "scale", "shape"]
        elif distr in ["poisson"]:
            tmp_list = ["loc", "mu"]
        else:
            raise Exception("This distribution has not been prepared here")
        # validating the list of provided covariants
        if set(descrip_fit.keys()) == set(tmp_list):
            params_list = ["cov_" + pp for pp in tmp_list]
        else:
            raise Exception(
                "Given the distribution ("
                + distr
                + "), should have information on {"
                + ", ".join(tmp_list)
                + "}, but got instead {"
                + ", ".join(list(set(descrip_fit.keys())))
                + "}"
            )

        # setting options in special cases --> for cfg?
        if ("transfo" in descrip_fit) and (descrip_fit["transfo"] != ""):
            boundaries_coeffs = {
                "transfo_asymptleft": [0, np.inf],
                "transfo_asymptright": [0, np.inf],
            }
            if transfo in ["generalizedlogistic", "generalizedalgebraic"]:
                boundaries_coeffs["transfo_alpha"] = [0, np.inf]

        elif var_targ in ["mrsomean"]:  # mrso? mrso_minmon?
            boundaries_coeffs = {"loc_0": [0, np.inf]}

        else:
            boundaries_coeffs = {}

        # selecting scenarios common to the target and the covariants
        common_scen = set(data.keys())
        for item in preds[var_targ]:
            common_scen = common_scen.intersection(list(preds[var_targ][item].keys()))
        common_scen = list(common_scen)
        common_scen.sort()
        scen0 = common_scen[0]  # just to shorten some lines

        # identiying number of gridpoints
        nr_gps = data[scen0].shape[-1]
        list_gp_nans = []

        # keeping only the same number of runs for covariants and target: this part COULD DEFINITELY BE IMPROVED IF RUNS WERE IDENTIFIED IN AN XARRAY. 8 months later, yup, this is getting out of hand.
        if data[common_scen[0]].ndim == 3:
            n_runs = {sc: data[sc].shape[0] for sc in data}
        else:
            raise Exception("Data must be Runs x Time x Gridpoints")

        # shaping covariants that will be used for all gridpoints.
        tmp_preds = {}
        for param in params_list:
            tmp_preds[param] = []
            # if info on a parameter is not provided in form_fit_distrib, next line will serve as a check.
            for name_covar in descrip_fit[param[len("cov_") :]]:
                form_fit = descrip_fit[param[len("cov_") :]][name_covar]
                _data = {
                    sc: np.repeat(
                        np.copy(preds[var_targ][name_covar][sc])[np.newaxis, :],
                        n_runs[sc],
                        axis=0,
                    )
                    for sc in preds[var_targ][name_covar]
                }
                tmp_preds[param].append(
                    [
                        name_covar,
                        np.hstack(
                            [
                                _data[scen][: n_runs[scen], :].flatten()
                                for scen in common_scen
                            ]
                        ),
                        form_fit,
                    ]
                )
        # adding the transformation if required
        if "transfo" in descrip_fit.keys():
            tmp_preds["transfo"] = descrip_fit["transfo"]

        # looping on every gridpoint
        quality_fit = np.nan * np.ones((nr_gps))
        sols = []
        for i_gp in np.arange(nr_gps):
            print(
                "Fitting a",
                distr,
                "on",
                var_targ,
                ":",
                str(np.round(100.0 * (i_gp + 1) / nr_gps, 1)),
                "%",
                end="\r",
            )
            # shaping inputs for covariated GEV (has checked before that it is Runs x Time x GridPoints)
            data_in = np.hstack(
                [data[scen][: n_runs[scen], :, i_gp].flatten() for scen in common_scen]
            )

            # fitting covariated GEV:@HERE, parameters to improve the fit --> class 'distrib_cov' using '**kwargs'
            tmp_cov = distrib_cov(
                data=data_in,
                cov_preds=tmp_preds,
                distrib=distr,
                boundaries_coeffs=boundaries_coeffs,
                **kwargs
            )
            sol = tmp_cov.fit()

            # saving
            sols.append(sol)
            if (
                np.all(np.isnan(list(sol.values()))) == False
            ):  # want to remove the case for missing points in data
                quality_fit[i_gp] = tmp_cov.neg_loglike(sol)
            else:
                quality_fit[i_gp] = np.nan

            # some solution may have bad results, especially if data_in had some nans
            if np.any(np.isnan(list(sol.values()))):
                list_gp_nans.append(i_gp)

            del sol

        # reshaping parameters for vector along gridpoints.
        # Warning, they are coefficients used to calculate parameters of the distribution. Yet, for the model, these coefficients are actually named parameters as well.
        params = {
            kk: np.array([sols[ii][kk] for ii in np.arange(len(sols))])
            for kk in sols[0]
        }
        # may include: loc, scale, shape AND transfo

        # end of loop, end of fit, archiving this training
        params_out[var_targ]["parameters"] = params
        params_out[var_targ]["distribution"] = distr
        params_out[var_targ]["form_fit_distrib"] = form_fit_distrib
        for cov_par in params_list:
            params_out[var_targ][cov_par] = [
                [item[0], item[2]] for item in tmp_preds[cov_par]
            ]
        params_out[var_targ]["scenarios_training"] = common_scen
        params_out[var_targ]["nruns_training"] = str(n_runs)
        params_out[var_targ]["quality_fit"] = quality_fit
        params_out[var_targ]["gridpoints_nan"] = list_gp_nans
        print("")  # because of the end='\r' in previous prints

    # end of loop on targets to fit a distribution on.
    return params_out


def transf_distrib2normal(preds, targs, params_l_distrib, threshold_sigma=6.0):
    """Probability integral transform of the data given parameters of a given distribution into their equivalent in a standard normal distribution.

    Parameters
    ----------
    preds : dict
        nested dictionary with 3 keys: cov_loc, cov_scale, cov_shape. Each one may be empty for no variation of the parameter of the distribution. If not empty, the variables will be used as covariants.
        - [targ][cov_...][covariant][scen]  (1d array (time) of predictor for specific scenario)

    targs : dict
        nested dictionary of targets with keys
        - [targ][scen] (3d array (run, time, gp) of target for specific scenario)

    params_l_distrib : dict
        nested dictionary of local variability paramters. Result of the function 'train_l_distrib'.

    threshold_sigma : float
        If a distribution is not correctly fitted, some values to transform may lead to unlikely values for a standard normal distribution. If above, they will be set to the threshold. Default : 6 (~happens once every 1e9 times)

    Returns
    -------
    transf_inputs : dict
        nested dictionary for transformed inputs
        - [targ][scen] (3d array (run, time, gp) of target for specific scenario)

    Notes
    -----
    - Assumptions:
    - Disclaimer:
    - TODO:

    """

    # creating the dictionary that will be filled in
    transf_inputs = {var_targ: {} for var_targ in targs}

    # looping over all targets to transform
    for var_targ in targs.keys():

        # preparing distribution
        distr = params_l_distrib[var_targ]["distribution"]
        if distr in ["gaussian"]:
            # this transformation must be performed even if it is from normal to normal, for there are evolution of the parameters.
            params_list = ["cov_loc", "cov_scale"]
            distr_cdf = ss.norm.cdf

        elif distr in ["GEV"]:
            params_list = ["cov_loc", "cov_scale", "cov_shape"]
            distr_cdf = ss.genextreme.cdf

        elif distr in ["poisson"]:
            params_list = ["cov_loc", "cov_mu"]
            distr_cdf = ss.poisson.cdf

        else:
            raise Exception("Distribution not prepared")

        # checking once that all required covariants are there for this target
        for par_var in params_list:
            # checking what type of inputs:
            req = [kk[0] for kk in params_l_distrib[var_targ][par_var]]
            for inp in req:
                if inp not in preds[var_targ]:
                    raise Exception("Missing input on " + par_var + ": " + inp)

        # evaluate the evolutions of the parameters for this variable
        params_all = eval_param_distrib(
            param=params_l_distrib[var_targ],
            cov=preds[var_targ],
            force_scen=params_l_distrib[var_targ]["scenarios_training"],
        )

        # checking if need to do a transformation
        descrip_fit = read_form_fit_distrib(
            params_l_distrib[var_targ]["form_fit_distrib"]
        )

        # transforming into a normal distribution all scenarios that can be transformed: need targs[var_targ] AND preds[var_targ]
        # objective: one timeserie at a time. It is a good tradeoff speed/RAM.
        test_unlikely_values = False
        for scen in params_l_distrib[var_targ]["scenarios_training"]:
            # preparing removal of NaN:
            ind_NoNaN = [
                ii
                for ii in range(targs[var_targ][scen].shape[2])
                if ii not in params_l_distrib[var_targ]["gridpoints_nan"]
            ]

            if "transfo" in descrip_fit.keys():
                left = params_l_distrib[var_targ]["parameters"]["transfo_asymptleft"][
                    ind_NoNaN
                ]
                right = params_l_distrib[var_targ]["parameters"]["transfo_asymptright"][
                    ind_NoNaN
                ]
                transfo = descrip_fit["transfo"]
                if transfo in ["generalizedlogistic", "generalizedalgebraic"]:
                    alpha = params_l_distrib[var_targ]["parameters"]["transfo_alpha"]
                    data_for_PIT = sigmoid_transf(
                        data=targs[var_targ][scen][..., ind_NoNaN],
                        left=left,
                        right=right,
                        type_sigm=transfo,
                        alpha=alpha,
                    )
                else:
                    data_for_PIT = sigmoid_transf(
                        data=targs[var_targ][scen][..., ind_NoNaN],
                        left=left,
                        right=right,
                        type_sigm=transfo,
                    )
            else:
                data_for_PIT = targs[var_targ][scen][..., ind_NoNaN]

            transf_inputs[var_targ][scen] = np.nan * np.ones(
                targs[var_targ][scen].shape
            )

            # looping on runs available for targs[var_targ] AND preds[var_targ]
            if False:
                if type(params_l_distrib[var_targ]["nruns_training"]) == str:
                    nrr = eval(params_l_distrib[var_targ]["nruns_training"])[scen]
                else:
                    nrr = params_l_distrib[var_targ]["nruns_training"][scen]
            else:
                nrr = targs[var_targ][scen].shape[0]
            for i_run in np.arange(nrr):

                # how likely are the observed values knowing the distribution at this year for this scenario?
                if distr in ["GEV"]:  # distribution that have location, scale AND shape
                    if (
                        params_all[scen]["loc_all"].ndim == 3
                    ):  # such a test could be avoided if using xarrays
                        tmp = distr_cdf(
                            x=data_for_PIT[i_run, :, :],
                            c=-params_all[scen]["shape_all"][i_run, ..., ind_NoNaN],
                            loc=params_all[scen]["loc_all"][i_run, ..., ind_NoNaN],
                            scale=params_all[scen]["scale_all"][i_run, ..., ind_NoNaN],
                        )
                    else:  # always (Time), even for constant terms, keep one axis for Time.
                        tmp = distr_cdf(
                            x=data_for_PIT[i_run, :, :],
                            c=-params_all[scen]["shape_all"][..., ind_NoNaN],
                            loc=params_all[scen]["loc_all"][..., ind_NoNaN],
                            scale=params_all[scen]["scale_all"][..., ind_NoNaN],
                        )

                elif distr in [
                    "gaussian"
                ]:  # distribution that have location, scale BUT no shape
                    if (
                        params_all[scen]["loc_all"].ndim == 3
                    ):  # such a test could be avoided if using xarrays
                        tmp = distr_cdf(
                            x=data_for_PIT[i_run, :, :],
                            loc=params_all[scen]["loc_all"][i_run, ..., ind_NoNaN],
                            scale=params_all[scen]["scale_all"][i_run, ..., ind_NoNaN],
                        )
                    else:  # always (Time), even for constant terms, keep one axis for Time.
                        tmp = distr_cdf(
                            x=data_for_PIT[i_run, :, :],
                            loc=params_all[scen]["loc_all"][..., ind_NoNaN],
                            scale=params_all[scen]["scale_all"][..., ind_NoNaN],
                        )

                elif distr in [
                    "poisson"
                ]:  # distribution that have location, scale BUT no shape
                    if (
                        params_all[scen]["loc_all"].ndim == 3
                    ):  # such a test could be avoided if using xarrays
                        tmp = distr_cdf(
                            k=data_for_PIT[i_run, :, :],
                            loc=params_all[scen]["loc_all"][i_run, ..., ind_NoNaN],
                            mu=params_all[scen]["mu_all"][i_run, ..., ind_NoNaN],
                        )
                    else:  # always (Time), even for constant terms, keep one axis for Time.
                        tmp = distr_cdf(
                            k=data_for_PIT[i_run, :, :],
                            loc=params_all[scen]["loc_all"][..., ind_NoNaN],
                            mu=params_all[scen]["mu_all"][..., ind_NoNaN],
                        )

                else:
                    raise Exception(
                        "Distribution not prepared, please make sure of its type."
                    )

                # to what values of a standard normal distribution would correspond these probabilities?
                transf_inputs[var_targ][scen][i_run, ...][..., ind_NoNaN] = ss.norm.ppf(
                    q=tmp, loc=0, scale=1
                )

            # Setting unlikely values to the threshold
            if np.any((transf_inputs[var_targ][scen] > threshold_sigma)) or np.any(
                (transf_inputs[var_targ][scen] < -threshold_sigma)
            ):
                test_unlikely_values = True
                transf_inputs[var_targ][scen][
                    np.where(transf_inputs[var_targ][scen] > threshold_sigma)
                ] = threshold_sigma
                transf_inputs[var_targ][scen][
                    np.where(transf_inputs[var_targ][scen] < -threshold_sigma)
                ] = -threshold_sigma
        if test_unlikely_values:
            print(
                "WARNING: some transformed values of "
                + var_targ
                + " are very unlikely, a possible cause is a fit missing strong signals. Action taken: blocking them at a limit."
            )

    return transf_inputs


class distrib_cov:
    """Class for the fit of a distribution with evolutions of the parameters with covariant variables. This class fits evolution of these covariants, with different functions possible. For now, only linear ('linear') and sigmoid ('logistic', 'arctan', 'gudermannian', 'errorfct', 'generalizedlogistic', 'generalizedalgebraic').
    The fit of this distribution embeds a semi-analytical optimization of the first guess.

    model for the evolutions of the parameters of the GEV, here only with linear terms:
    * mu = loc0 + sum_i loc_i * cov_loc_i
    * sigma = scale0 + sum_i scale_i * cov_scale_i
    * shape = shape0 + sum_i shape_i * cov_shape_i

    If logistic terms are asked over the same parameter, they are assumed to be under the same exponential, for instance:
    * mu = loc0 + delta / (1 + exp(beta1 * cov_loc1 + beta2 * cov_loc2 - epsilon )))
    instead of:
    * mu = loc0 + delta1 / (1 + exp(beta1 * (cov_loc1-epsilon1))) + delta2 / (1 + exp(beta2 * (cov_loc2-epsilon2)))
    This approximation helps in providing a better first guess for these terms.


    Parameters
    ----------
    data : numpy array 1D
        vector of observations for fit of a distribution with covariates on parameters

    cov_preds : dict
        dictionary of covariate variables (3 keys, (cov_loc, cov_scale, cov_shape)), for dependencies of the parameters of the distribution. Each item of the dictionary must have the same shape as data.

    distr : str
        type of distribution to fit. For now, only GEV, gaussian and poisson are prepared.

    method_fit : str
        Type of algorithm used during the optimization, using the function 'minimize'. Default: 'Nelder-Mead'. HIGHLY RECOMMENDED, for its stability.

    xtol_req : float
        Accurracy of the fit. Interpreted differently depending on 'method_fit'. Default: 1e-3

    maxiter : None or int
        Maximum number of iteration of the optimization. Default: 5000. Doubled if logistic asked.

    maxfev : None or int
        Maximum number of evaluation of the function during the optimization. Default: np.inf (important for relatively complex fits)

    fg_shape : float
        First guess for the semi-analytical optimization of the actual first guess of the distribution.

    boundaries_params : dictionnary{'loc', 'scale', 'shape'} of list of two values
        Interval for the parameters of the distribution to fit. Particulary important for positive scale and shape within ]-inf, 1/3]. Very little gridpoints have mean shape above 1/3. When using scipy, calculating an array of means of distributions with shapes > 1/3 causes an issue.

    prior_shape : float or None
        sets a gaussian prior for the shape parameter. prior_shape /2 = standard deviation of this gaussian prior. Default: 0, equivalent to None.

    option_silent : boolean
        just to avoid too many messages

    error_failedfit : boolean
        if True, will raise an issue if the fit failed

    Returns
    -------
    Once this class has been initialized, the go-to function is fit(). It returns the vector of solutions for the problem proposed.

    Notes
    -----
    - Assumptions:
    - Disclaimer:
    - TODO:
        - case of multiple logistic evolutions over the same parameter of the distribution: assumed to be under the same exponential, much easier to handle.
    """

    # --------------------
    # INITIALIZATION
    def __init__(
        self,
        data,
        cov_preds,
        distrib,
        fg_shape=-0.25,
        prior_shape=0,
        boundaries_params={},
        boundaries_coeffs={},
        option_silent=True,
        method_fit="Nelder-Mead",
        xtol_req=1e-3,
        maxiter=50000,
        maxfev=50000,
        error_failedfit=False,
    ):

        # data
        self.data = data

        # covariates:
        self.possible_sigmoid_forms = [
            "logistic",
            "arctan",
            "gudermannian",
            "errorfct",
            "generalizedlogistic",
            "generalizedalgebraic",
        ]
        tmp = {
            "params": [
                cov_type[len("cov_") :]
                for cov_type in cov_preds.keys()
                if cov_type != "transfo"
            ]
        }
        for typ in tmp["params"]:
            cov_type = "cov_" + typ

            # names of the covariates
            tmp[cov_type + "_names"] = [item[0] for item in cov_preds[cov_type]]

            # data for the covariates
            tmp[cov_type + "_data"] = [item[1] for item in cov_preds[cov_type]]

            # form of the fit for the covariates
            tmp[cov_type + "_form"] = [item[2] for item in cov_preds[cov_type]]
            # warning: assuming that if there are multiple logistic evolutions for the same parameter, they are under the same exponential.
            # coefficients associated
            tmp["coeffs_" + typ + "_names"] = [typ + "_0"]
            for ii in np.arange(len(tmp[cov_type + "_names"])):
                if tmp[cov_type + "_form"][ii] == "linear":
                    tmp["coeffs_" + typ + "_names"].append(
                        typ + "_linear_" + tmp[cov_type + "_names"][ii]
                    )

                elif tmp[cov_type + "_form"][ii] in self.possible_sigmoid_forms:
                    form_sigmoid = tmp[cov_type + "_form"][ii]
                    if (
                        typ + "_" + form_sigmoid + "_asymptleft"
                        not in tmp["coeffs_" + typ + "_names"]
                    ):
                        tmp["coeffs_" + typ + "_names"].append(
                            typ + "_" + form_sigmoid + "_asymptleft"
                        )
                        tmp["coeffs_" + typ + "_names"].append(
                            typ + "_" + form_sigmoid + "_asymptright"
                        )
                        tmp["coeffs_" + typ + "_names"].append(
                            typ + "_" + form_sigmoid + "_epsilon"
                        )
                        if form_sigmoid in [
                            "generalizedlogistic",
                            "generalizedalgebraic",
                        ]:
                            tmp["coeffs_" + typ + "_names"].append(
                                typ + "_" + form_sigmoid + "_alpha"
                            )
                    tmp["coeffs_" + typ + "_names"].append(
                        typ
                        + "_"
                        + form_sigmoid
                        + "_lambda_"
                        + tmp[cov_type + "_names"][ii]
                    )

                elif (type(tmp[cov_type + "_form"][ii]) == list) and (
                    tmp[cov_type + "_form"][ii][0] == "power"
                ):
                    pwr = tmp[cov_type + "_form"][ii][1]
                    tmp["coeffs_" + typ + "_names"].append(
                        typ + "_power" + str(pwr) + "_" + tmp[cov_type + "_names"][ii]
                    )

                else:
                    raise Exception("Unknown form of fit in " + type)

        # check for a case not handled: on one parameter, several sigmoids asked, but from different kinds: that would be a mess for the evaluation of the coefficients on the parameters.
        for typ in tmp["params"]:
            lst_forms = [
                form
                for form in tmp[cov_type + "_form"]
                if form in self.possible_sigmoid_forms
            ]
            if len(set(lst_forms)) > 1:
                raise Exception(
                    "Please avoid asking for different types of sigmoid on 1 parameter."
                )

        # adding terms if transformation asked.
        if "transfo" in cov_preds.keys():
            self.transfo = [True, cov_preds["transfo"]]
            tmp["coeffs_transfo_names"] = ["transfo_asymptleft", "transfo_asymptright"]
            if self.transfo[1] in ["generalizedlogistic", "generalizedalgebraic"]:
                tmp["coeffs_transfo_names"].append("transfo_alpha")
        else:
            self.transfo = [False, None]

        # saving that in a single variable
        self.cov = tmp

        # full list of coefficients
        if distrib in ["gaussian"]:
            tmp = self.cov["coeffs_loc_names"] + self.cov["coeffs_scale_names"]
        elif distrib in [
            "GEV"
        ]:  #  or (self.transfo[0] and str.split(distrib,'-')[0] in ['GEV'])
            tmp = (
                self.cov["coeffs_loc_names"]
                + self.cov["coeffs_scale_names"]
                + self.cov["coeffs_shape_names"]
            )
        elif distrib in ["poisson"]:
            tmp = self.cov["coeffs_loc_names"] + self.cov["coeffs_mu_names"]
        if self.transfo[0]:
            tmp = tmp + self.cov["coeffs_transfo_names"]
        self.coeffs_names = tmp

        # arguments
        self.distrib = distrib
        self.fg_shape = fg_shape
        self.method_fit = method_fit
        if len(boundaries_params) == 0:  # default
            if self.distrib in ["gaussian"]:
                self.boundaries_params = {
                    "loc": [-np.inf, np.inf],
                    "scale": [0, np.inf],
                }
            elif self.distrib in ["GEV"]:
                self.boundaries_params = {
                    "loc": [-np.inf, np.inf],
                    "scale": [0, np.inf],
                    "shape": [-np.inf, 1 / 3],
                }
            elif self.distrib in ["poisson"]:
                self.boundaries_params = {"loc": [-np.inf, np.inf], "mu": [0, np.inf]}
            else:
                raise Exception("Distribution not prepared here")
        else:
            self.boundaries_params = boundaries_params
        self.boundaries_coeffs = boundaries_coeffs  # this one is more technical
        self.xtol_req = xtol_req
        self.maxiter = maxiter  # used to have np.inf, but sometimes, the fit doesnt work... and it is meant to.
        self.error_failedfit = error_failedfit
        self.maxfev = maxfev  # used to have np.inf, but sometimes, the fit doesnt work... and it is meant to.
        if method_fit in [
            "CG",
            "BFGS",
            "L-BFGS-B",
        ]:  # TNC and trust-constr: better to use xtol
            self.name_xtol = "gtol"
        elif method_fit in ["Newton-CG", "Powell", "TNC", "trust-constr"]:
            self.name_xtol = "xtol"
        elif method_fit in ["Nelder-Mead"]:
            self.name_xtol = "xatol"
        elif method_fit in [
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
            "COBYLA",
            "SLSQP",
        ]:
            raise Exception("method for this fit not prepared, to avoid")

        # test on sizes of sample
        if self.data.ndim > 1:
            raise Exception(
                "input data must be a vector"
            )  # should do tests also for covariations.
        if (len(self.cov["cov_loc_data"]) > 0) and (
            self.data.shape[0] != self.cov["cov_loc_data"][0].shape[0]
        ):
            raise Exception("Sample of data and covariations have different sizes...")

        # prior for shape: normal distribution with sd=0 is not valid
        if type(prior_shape) in [tuple, list, np.ndarray]:
            # need to half the prior_shape to be consistent
            prior_shape[1] /= 2
            # return logpdf of a normal distribution
            self._prior_shape = ss.norm(loc=prior_shape[0], scale=prior_shape[1]).logpdf

        else:
            if np.isclose(prior_shape, 0):
                if option_silent == False:
                    print("setting prior_shape to None")
                prior_shape = None

            if prior_shape is None:
                # always return 0
                self._prior_shape = lambda x: 0.0
            else:
                # need to half the prior_shape to be consistent
                prior_shape /= 2
                # return logpdf of a normal distribution
                self._prior_shape = ss.norm(loc=0, scale=prior_shape).logpdf

    # --------------------

    # --------------------
    # FIRST GUESS
    @staticmethod
    def g(k, shape):
        return gamma(1 - k * shape)

    def GEV_mean(self, loc, scale, shape):
        # case: shape > 1
        out = np.inf * np.ones(loc.shape)
        # case: shape < 1
        ind = np.where(shape < 1)
        out[ind] = (loc + scale * (self.g(1, shape) - 1) / shape)[ind]
        # case: shape ~ 0
        ind = np.where(np.isclose(shape, 0))
        out[ind] = (loc + scale * e)[ind]
        return out

    def GEV_var(self, loc, scale, shape):
        # case: shape > 1/2
        out = np.inf * np.ones(loc.shape)
        # case: shape < 1/2
        ind = np.where(shape < 1 / 2)
        out[ind] = (
            scale**2 * (self.g(2, shape) - self.g(1, shape) ** 2) / shape**2
        )[ind]
        # case : shape ~ 0
        ind = np.where(np.isclose(shape, 0))
        out[ind] = (scale**2 * np.pi**2 / 6)[ind]
        return out

    def GEV_skew(self, loc, scale, shape):
        # case: shape > 1/3
        out = np.inf * np.ones(loc.shape)
        # case: shape < 1/3
        ind = np.where(shape < 1 / 3)
        out[ind] = (
            np.sign(shape)
            * (
                self.g(3, shape)
                - 3 * self.g(2, shape) * self.g(1, shape)
                + 2 * self.g(1, shape) ** 3
            )
            / (self.g(2, shape) - self.g(1, shape) ** 2) ** (3 / 2)
        )[ind]
        # case: shape ~ 0
        ind = np.where(np.isclose(shape, 0))
        out[ind] = 12 * np.sqrt(6) * zeta(3) / np.pi**3
        return out

    def eval_1_fg0(self, arr):
        deltaloc, deltascale, shape = arr

        # checking these values, by taking 'self.fg_x0', with first estimate of parameters, including covariants of loc
        x0_test = np.copy(self.fg_x0)
        x0_test[self.coeffs_names.index("loc_0")] += deltaloc  # careful with that one
        x0_test[self.coeffs_names.index("scale_0")] += deltascale
        if "shape_0" in self.coeffs_names:
            x0_test[self.coeffs_names.index("shape_0")] = shape
        args = self._parse_args(x0_test)
        test = self._test_coeffs(x0_test) * self._test_evol_params(self.data_tmp, args)

        if test:
            if self.distrib in ["GEV"]:
                # calculating mean, median, skew that a GEV would have with these parameters
                err_mean = (
                    self.dd_mean - np.mean(self.GEV_mean(args[0], args[1], args[2]))
                ) ** 2.0
                err_var = (
                    self.dd_var - np.mean(self.GEV_var(args[0], args[1], args[2]))
                ) ** 2.0
                err_skew = (
                    self.dd_skew - np.mean(self.GEV_skew(args[0], args[1], args[2]))
                ) ** 2.0

            else:
                raise Exception("This distribution has not been prepared.")

            if self.eval_fg0_with_dd_mean:
                out = err_mean + err_var + err_skew
            else:
                # trying without impact of dd_mean: for signals with high interannual variability, hard to detrend correctly data
                out = err_var + err_skew
        else:
            out = np.inf

        return out

    @staticmethod
    def find_m1_m2(dat, inds_sigm, data_sigm):
        # identifying the sigmoid term with the stronger variations on the sigmoid: used to identify the proper 'm1', which matters for the sigmoid transformation and loc_0
        tmp = []
        for ii in range(len(data_sigm)):
            ind = data_sigm[
                0
            ].argsort()  # here starts the moment where it is written as if a single sigmoid evolution
            ii = int(
                0.1 * len(ind)
            )  # taking average over 10% of higher and lower values to determine these two values
            tmp.append(np.abs(np.mean(dat[ind[:ii]]) - np.mean(dat[ind[-ii:]])))
        istrsigm = np.argmax(tmp)
        # identification of the overall evolution --> identification of min and max
        ind = data_sigm[
            istrsigm
        ].argsort()  # here starts the moment where it is written as if a single sigmoid evolution
        ii = int(
            0.1 * len(ind)
        )  # taking average over 10% of higher and lower values to determine these two values
        if np.mean(dat[ind[:ii]]) < np.mean(
            dat[ind[-ii:]]
        ):  # increasing sigmoid evolution
            m1 = np.min(dat)
            m2 = np.max(dat)
        else:  # decreasing sigmoid evolution
            m1 = np.max(dat)
            m2 = np.min(dat)
        # increasing range of (m1,m2). The 2 following lines account for both signs of the derivative
        m1 += 0.01 * (m1 - m2)
        m2 -= 0.01 * (m1 - m2)
        return m1, m2

    def reglin_fg(self, typ_cov, data):
        # initiating
        self.tmp_sol[typ_cov] = np.zeros(len(self.cov["coeffs_" + typ_cov + "_names"]))

        # ---------------------
        # linear contribution
        if (
            len(self.cov["cov_" + typ_cov + "_names"]) > 0
        ):  # checking that they are covariates
            inds_lin = [
                i
                for i, form in enumerate(self.cov["cov_" + typ_cov + "_form"])
                if form == "linear"
            ]
        else:
            inds_lin = []
        if (
            len(inds_lin) > 0
        ):  # checking that they are covariates on parameters with linear form
            data_lin = [self.cov["cov_" + typ_cov + "_data"][i] for i in inds_lin]
            ex = np.concatenate([np.ones((len(data), 1)), np.array(data_lin).T], axis=1)
        else:
            ex = np.ones(len(data))  # overkill, can simply remove mean.
        mod = OLS(exog=ex, endog=data)
        res = mod.fit()
        self.tmp_sol[typ_cov][0] = res.params[0]
        if len(inds_lin) > 0:  # just filling in linear coefficients
            for i in np.arange(1, len(res.params)):
                cf = (
                    typ_cov
                    + "_linear_"
                    + self.cov["cov_" + typ_cov + "_names"][inds_lin[i - 1]]
                )
                self.tmp_sol[typ_cov][
                    self.cov["coeffs_" + typ_cov + "_names"].index(cf)
                ] = res.params[i]

        # detrending with linear evolution
        data_detrended = data - res.predict()
        # ---------------------

        # ---------------------
        # power contribution
        if (
            len(self.cov["cov_" + typ_cov + "_names"]) > 0
        ):  # checking that they are covariates
            inds_pow = [
                i
                for i, form in enumerate(self.cov["cov_" + typ_cov + "_form"])
                if (type(self.cov["cov_" + typ_cov + "_form"][i]) == list)
                and (self.cov["cov_" + typ_cov + "_form"][i][0] == "power")
            ]
        else:
            inds_pow = []
        if (
            len(inds_pow) > 0
        ):  # checking that they are covariates on parameters with linear form
            data_pow = []
            for i in inds_pow:
                pwr = self.cov["cov_" + typ_cov + "_form"][i][1]
                data_pow.append(self.cov["cov_" + typ_cov + "_data"][i] ** pwr)
            ex = np.concatenate([np.ones((len(data), 1)), np.array(data_pow).T], axis=1)
        else:
            ex = np.ones(len(data))  # overkill, can simply remove mean.
        mod = OLS(exog=ex, endog=data_detrended)
        res = mod.fit()
        self.tmp_sol[typ_cov][0] += res.params[0]
        if len(inds_pow) > 0:  # just filling in power coefficients
            for i in np.arange(1, len(res.params)):
                pwr = self.cov["cov_" + typ_cov + "_form"][inds_pow[i - 1]][1]
                cf = (
                    typ_cov
                    + "_power"
                    + str(pwr)
                    + "_"
                    + self.cov["cov_" + typ_cov + "_names"][inds_pow[i - 1]]
                )
                self.tmp_sol[typ_cov][
                    self.cov["coeffs_" + typ_cov + "_names"].index(cf)
                ] = res.params[i]

        # detrending with power evolution
        data_detrended -= res.predict()
        # ---------------------

        # ---------------------
        # sigmoid contribution
        # principle: making a sigmoid transformation to evaluate parameters.
        if (
            len(self.cov["cov_" + typ_cov + "_names"]) > 0
        ):  # checking that they are covariates on this parameter
            inds_sigm = [
                i
                for i, form in enumerate(self.cov["cov_" + typ_cov + "_form"])
                if form in self.possible_sigmoid_forms
            ]
        else:
            inds_sigm = []
        if (
            len(inds_sigm) > 0
        ):  # checking that there are covariates on this parameter with lositic form
            # already made sure that only one sigmoid form is used on this parameter
            form_sigm = self.cov["cov_" + typ_cov + "_form"][inds_sigm[0]]
            # gathering data
            data_sigm = [self.cov["cov_" + typ_cov + "_data"][i] for i in inds_sigm]
            ex = np.concatenate(
                [np.ones((len(self.data), 1)), np.array(data_sigm).T], axis=1
            )

            # identifying boundaries of the sigmoid evolutions
            m1, m2 = self.find_m1_m2(data_detrended, inds_sigm, data_sigm)

            # sigmoid transformation
            if form_sigm in ["generalizedlogistic", "generalizedalgebraic"]:
                alpha = 1
                data_detrended_transf = sigmoid_transf(
                    data=data_detrended,
                    left=m1,
                    right=m2,
                    type_sigm=form_sigm,
                    detect_NaN=False,
                    alpha=alpha,
                )
                self.tmp_sol[typ_cov][
                    self.cov["coeffs_" + typ_cov + "_names"].index(
                        typ_cov + "_" + form_sigm + "_alpha"
                    )
                ] = alpha
            else:
                data_detrended_transf = sigmoid_transf(
                    data=data_detrended,
                    left=m1,
                    right=m2,
                    type_sigm=form_sigm,
                    detect_NaN=False,
                )
            # fitting
            mod = OLS(
                exog=ex, endog=data_detrended_transf
            )  # assuming linear variations on these terms!
            res = mod.fit()
            # filling in sigmoid coefficients
            self.tmp_sol[typ_cov][
                self.cov["coeffs_" + typ_cov + "_names"].index(
                    typ_cov + "_" + form_sigm + "_asymptleft"
                )
            ] = m1
            self.tmp_sol[typ_cov][
                self.cov["coeffs_" + typ_cov + "_names"].index(
                    typ_cov + "_" + form_sigm + "_asymptright"
                )
            ] = m2
            for ii in range(len(data_sigm)):
                cv_nm = self.cov["cov_" + typ_cov + "_names"][inds_sigm[ii]]
                self.tmp_sol[typ_cov][
                    self.cov["coeffs_" + typ_cov + "_names"].index(
                        typ_cov + "_" + form_sigm + "_lambda_" + cv_nm
                    )
                ] = res.params[1 + ii]
            self.tmp_sol[typ_cov][
                self.cov["coeffs_" + typ_cov + "_names"].index(
                    typ_cov + "_" + form_sigm + "_epsilon"
                )
            ] = -res.params[
                0
            ]  # better physical interpretation, we are using anomalies in global variables

            # detrending with sigmoid evolution
            if form_sigm in ["generalizedlogistic", "generalizedalgebraic"]:
                data_detrended -= sigmoid_backtransf(
                    res.predict(),
                    m1,
                    m2,
                    type_sigm=form_sigm,
                    detect_NaN=False,
                    alpha=alpha,
                )
            else:
                data_detrended -= sigmoid_backtransf(
                    res.predict(), m1, m2, type_sigm=form_sigm, detect_NaN=False
                )

        else:
            pass  # nothing to do
        # ---------------------

        return data_detrended

    def find_fg(self):
        # Objective: optimizing loc_0, scale_0, shape_0, so that it matches the expected mean, variance and skewness. These values may be used as first guess for the real fit, especially the shape.
        # The use of the parameter "eval_fg0_best" leads to an effective doubling of this step. Even though this step is very fast, it could be avoided by saving both solutions in a self.thingy

        # INITIATING
        self.tmp_sol = {}

        # TRANFORMATION?
        if self.transfo[0]:
            # using all inputs on location to identify correct boundaries. More statistical sense with only linear terms?
            inds_sigm = np.arange(len(self.cov["cov_loc_form"]))
            data_sigm = self.cov["cov_loc_data"]

            # identifying boundaries of the sigmoid evolution
            m1, m2 = self.find_m1_m2(self.data, inds_sigm, data_sigm)

            # transformation: new data that will be used
            if self.transfo[1] in ["generalizedlogistic", "generalizedalgebraic"]:
                # initiating alpha to 1
                alpha = 1
                data = sigmoid_transf(
                    data=self.data,
                    left=m1,
                    right=m2,
                    type_sigm=self.transfo[1],
                    alpha=alpha,
                )
                self.tmp_sol["transfo"] = [
                    m1,
                    m2,
                    1,
                ]  # transfo_asymptleft & right in coeffs_transfo_names
            else:
                data = sigmoid_transf(
                    data=self.data, left=m1, right=m2, type_sigm=self.transfo[1]
                )
                self.tmp_sol["transfo"] = [
                    m1,
                    m2,
                ]  # transfo_asymptleft & right in coeffs_transfo_names
        else:
            data = self.data

        # LOCATION:
        data_det = self.reglin_fg(typ_cov="loc", data=data)

        # SCALE:
        # evaluating running standard deviation of scale on a nn-values window: there will be peaks at change in scenarios, but it will be smoothen out by the linear regressions
        nn = 100  # with sma, seems to be better to have high values for proper evaluations of scale_0
        std_data_det = np.sqrt(
            np.max(
                [
                    np.convolve(data_det**2, np.ones(nn) / nn, mode="same")
                    - np.convolve(data_det, np.ones(nn) / nn, mode="same") ** 2,
                    np.zeros(data_det.shape),
                ],
                axis=0,
            )
        )
        if self.distrib in ["GEV", "gaussian"]:
            _ = self.reglin_fg(
                typ_cov="scale", data=std_data_det
            )  # coeffs are saved, useless to get the rest of the signal
        elif self.distrib in ["poisson"]:
            _ = self.reglin_fg(
                typ_cov="mu", data=std_data_det
            )  # coeffs are saved, useless to get the rest of the signal
        else:
            raise Exception("Distribution not prepared here")

        # preparing optimization of loc0, scale0 and shape0: calculating mean, median, skew kurtosis of detrended data
        self.dd_mean = np.mean(
            data_det
        )  # np.mean(data) #np.mean(data_det) why did i switch from data_det to data?
        self.dd_var = np.var(data_det)
        self.dd_skew = ss.skew(data_det)  # ss.skew(data_det)

        # finally creating the first guess
        if self.distrib in ["GEV"]:

            # initialize first guess 'x0'. the full x0 will be used to test validity of the first guesses while optimizing.
            self.fg_x0 = (
                list(self.tmp_sol["loc"])
                + list(self.tmp_sol["scale"])
                + [self.fg_shape]
                + list(np.zeros(len(self.cov["coeffs_shape_names"]) - 1))
            )
            # self.fg_x0[self.coeffs_names.index('loc_0')] += self.dd_mean
            self.fg_x0[self.coeffs_names.index("scale_0")] += np.sqrt(self.dd_var)
            if self.transfo[0]:
                self.fg_x0 = self.fg_x0 + list(self.tmp_sol["transfo"])

            # just avoiding zeros on sigmoid coefficients on shape parameter
            for param in self.cov["coeffs_shape_names"]:
                for form in self.possible_sigmoid_forms:
                    if (
                        form + "_lambda" in param
                    ):  # this parameter is a sigmoid lambda. Making a slow evolution
                        self.fg_x0[self.coeffs_names.index(param)] = 0.1
                    if (
                        form + "_asympt" in param
                    ):  # this parameter is a sigmoid difference. Making a small difference, relatively to its parameter_0
                        if "left" in param:
                            self.fg_x0[self.coeffs_names.index(param)] = 0.0
                        else:  # right
                            idcov = str.split(param, "_")[0] + "_0"
                            self.fg_x0[self.coeffs_names.index(param)] = (
                                0.01 * self.fg_x0[self.coeffs_names.index(idcov)]
                            )

            # preparing test of validity of starting domain
            x0_test = np.copy(self.fg_x0)
            # x0_test[self.coeffs_names.index('scale_0')] += np.sqrt(self.dd_var)
            # x0_test[self.coeffs_names.index('loc_0')] += self.dd_mean # careful with that one
            loc, scale, _ = self._parse_args(x0_test)

            # checking domain
            tmp = scale / (self.data - loc)
            bnds_c_limit = np.max(tmp[np.where(self.data - loc < 0)]), np.min(
                tmp[np.where(self.data - loc > 0)]
            )
            if bnds_c_limit[1] < bnds_c_limit[0]:
                raise Exception(
                    "No possible solution for shape there ("
                    + str(bnds_c_limit[0])
                    + " not < to "
                    + str(bnds_c_limit[1])
                    + "), please check."
                )

            # updating shape to confirm the validity of its domain, so that the fg_shape allows for a start within the support of the GEV
            correct_borders_shape = np.max(
                [-bnds_c_limit[1], self.boundaries_params["shape"][0]]
            ), np.min([-bnds_c_limit[0], self.boundaries_params["shape"][1]])
            self.safe_fg_shape = 0.5 * (
                correct_borders_shape[0] + correct_borders_shape[1]
            )  # may need to take min or max +/- 0.1 for shape with covariants
            self.fg_x0[self.coeffs_names.index("shape_0")] = self.safe_fg_shape
            # 0.1 is meant to add some margin in the domain. The definition of the support assumes that the shape is <0, which is no issue, for fg_shape is usually <0
            # self.safe_fg_shape = np.max( [self.fg_shape, np.sqrt(self.dd_var) / (self.dd_mean - np.max(self.data)) + 0.1 ] )

            # TEMPORARY DEACTIVATION OF THE OPTIMIZATION HERE, BECAUSE OF PROGRESS ON THE DEFINITION OF 'safe_fg_shape': MAY NOT BE NECESSARY ANYMORE, TRYING.
            x0 = self.fg_x0
            if False:
                # initializations of the values that will be optimized
                # xx0 = [self.dd_mean, np.sqrt(self.dd_var), self.safe_fg_shape]
                xx0 = [0, 0, self.safe_fg_shape]

                # looping over options for the first guess, breaking when have a correct one. Used to have more options, now only 2 possibilities, but more efficient.
                checks = {}
                # need to pass eventually transformed data to tests embedded within self.eval_1_fg_0
                self.data_tmp = data
                for self.eval_fg0_with_dd_mean in [True, False]:
                    # The first guess for this optimization is roughly done.
                    m = minimize(
                        self.eval_1_fg0,
                        x0=xx0,
                        method=self.method_fit,
                        options={self.name_xtol: self.xtol_req},
                    )
                    delta_loc0, delta_scale0, shape0 = m.x

                    # allocating values to x0, already containing information on initial first value of loc_0 and initial coefficients on linear covariates of location
                    x0 = np.copy(self.fg_x0)
                    x0[
                        self.coeffs_names.index("loc_0")
                    ] += delta_loc0  # the location computed out of this step is NOT a proper location_0!
                    x0[
                        self.coeffs_names.index("scale_0")
                    ] += delta_scale0  # better & safer results with += instead of =
                    if "shape_0" in self.coeffs_names:
                        x0[self.coeffs_names.index("shape_0")] = shape0

                    # checking the first guess
                    args = self._parse_args(x0)
                    test = self._test_coeffs(x0) * self._test_evol_params(
                        self.data_tmp, args
                    )
                    checks[self.eval_fg0_with_dd_mean] = [
                        np.copy(x0),
                        test,
                        self.neg_loglike(x0),
                    ]
                del self.data_tmp

                # checking if valid first guess
                valid_True = checks[True][1] and np.isinf(checks[True][1]) == False
                valid_False = checks[False][1] and np.isinf(checks[False][1]) == False
                # selecting one or the other:
                if valid_True and valid_False:  # both options are valid
                    if (
                        self.eval_fg0_best
                    ):  # default mode. Most of the time, will be here, taking the best first guess among the 2 options. For difficult fits, the 2nd may actually work better.
                        if checks[True][2] <= checks[False][2]:
                            x0 = checks[True][0]
                        else:
                            x0 = checks[False][0]
                    else:
                        if checks[True][2] <= checks[False][2]:
                            x0 = checks[False][0]
                        else:
                            x0 = checks[True][0]

                elif valid_True:  # only the first mode is valid
                    x0 = checks[True][0]

                elif valid_False:  # only the second mode is valid
                    x0 = checks[False][0]

        elif self.distrib in ["gaussian"]:
            # special case: the relevant parameters from the gaussian can directly be deduced here:
            x0 = list(self.tmp_sol["loc"]) + list(self.tmp_sol["scale"])
            x0[
                self.coeffs_names.index("loc_0")
            ] += (
                self.dd_mean
            )  # the location computed out of this step is NOT a proper location_0!
            if self.transfo[0]:
                x0 = x0 + list(self.tmp_sol["transfo"])

        elif self.distrib in ["poisson"]:
            # special case: the relevant parameters from the gaussian can directly be deduced here:
            x0 = list(self.tmp_sol["loc"]) + list(self.tmp_sol["mu"])
            x0[
                self.coeffs_names.index("loc_0")
            ] -= (
                self.dd_var
            )  # the location computed out of this step is NOT a proper location_0!
            x0[
                self.coeffs_names.index("mu_0")
            ] += (
                self.dd_var
            )  # the location computed out of this step is NOT a proper location_0!
            if self.transfo[0]:
                x0 = x0 + list(self.tmp_sol["transfo"])
            args = self._parse_args(x0)
            if np.min(args[1]) < 0:
                x0[self.coeffs_names.index("mu_0")] -= np.min(args[1]) * 1.1
            # some points in the sample may be 0, and for a location too high, it would cause the first guess to start in a wrong domain
            counter_modif_loc0 = 0
            while (
                np.isinf(self.loglike(x0))
                and np.any(np.isclose(self.data, 0))
                and (counter_modif_loc0 < 10)
            ):
                x0[self.coeffs_names.index("loc_0")] -= 1
                counter_modif_loc0 += 1

        else:
            raise Exception("Distribution not prepared here")

        # checking whether succeeded, or need more work on first guess
        if np.isinf(self.loglike(x0)):
            raise Exception("Could not find an improved first guess")
        else:
            return x0

    # --------------------

    # --------------------
    # TEST COEFFICIENTS
    def _parse_args(self, args):
        coeffs = np.asarray(args).T

        # looping on every available parameter: loc, scale, shape
        tmp = {}
        for typ in self.cov["params"]:

            # initializing the output
            tmp[typ] = coeffs[self.coeffs_names.index(typ + "_0")] * np.ones(
                self.data.shape
            )

            # looping on every covariate of this parameter
            for ii in np.arange(len(self.cov["cov_" + typ + "_names"])):
                # checking the form of its equations and using the relevant coefficients
                name_cov = self.cov["cov_" + typ + "_names"][ii]

                if self.cov["cov_" + typ + "_form"][ii] == "linear":
                    # THIS is the plain normal case.
                    tmp[typ] += (
                        coeffs[self.coeffs_names.index(typ + "_linear_" + name_cov)]
                        * self.cov["cov_" + typ + "_data"][ii]
                    )

                elif (
                    self.cov["cov_" + typ + "_form"][ii] in self.possible_sigmoid_forms
                ):
                    form_sigm = self.cov["cov_" + typ + "_form"][ii]
                    # all sigmoid terms are under the same exponential: they are dealt with the first time sigmoid is encountered on this parameter
                    if (
                        self.cov["cov_" + typ + "_form"].index(form_sigm) == ii
                    ):  # only the first one with sigmoid is returned here, checking if it is the one of the loop
                        # summing sigmoid terms
                        ind_sigm = np.where(
                            np.array(self.cov["cov_" + typ + "_form"]) == form_sigm
                        )[0]
                        var = 0
                        for i in ind_sigm:
                            L = coeffs[
                                self.coeffs_names.index(
                                    typ
                                    + "_"
                                    + form_sigm
                                    + "_lambda_"
                                    + self.cov["cov_" + typ + "_names"][i]
                                )
                            ]
                            var += L * self.cov["cov_" + typ + "_data"][i]
                        # dealing with the exponential
                        left = coeffs[
                            self.coeffs_names.index(
                                typ + "_" + form_sigm + "_asymptleft"
                            )
                        ]
                        right = coeffs[
                            self.coeffs_names.index(
                                typ + "_" + form_sigm + "_asymptright"
                            )
                        ]
                        eps = coeffs[
                            self.coeffs_names.index(typ + "_" + form_sigm + "_epsilon")
                        ]
                        if np.isclose(left, right):
                            tmp[typ] += np.zeros(
                                var.shape
                            )  # just for shape of tmp[typ]
                        else:
                            if form_sigm in [
                                "generalizedlogistic",
                                "generalizedalgebraic",
                            ]:
                                tmp[typ] += sigmoid_backtransf(
                                    data=var - eps,
                                    left=left,
                                    right=right,
                                    type_sigm=form_sigm,
                                    alpha=coeffs[
                                        self.coeffs_names.index(
                                            typ + "_" + form_sigm + "_alpha"
                                        )
                                    ],
                                )
                            else:
                                tmp[typ] += sigmoid_backtransf(
                                    data=var - eps,
                                    left=left,
                                    right=right,
                                    type_sigm=form_sigm,
                                )
                    else:  # not the first sigmoid term on this parameter, already accounted for.
                        pass

                elif (type(self.cov["cov_" + typ + "_form"][ii]) == list) and (
                    self.cov["cov_" + typ + "_form"][ii][0] == "power"
                ):
                    pwr = self.cov["cov_" + typ + "_form"][ii][1]
                    tmp[typ] += (
                        coeffs[
                            self.coeffs_names.index(
                                typ + "_power" + str(pwr) + "_" + name_cov
                            )
                        ]
                        * self.cov["cov_" + typ + "_data"][ii] ** pwr
                    )

                else:
                    raise Exception(
                        self.cov["cov_" + typ + "_form"][ii] + " is not prepared!"
                    )

        if self.distrib in ["GEV"]:
            # The parameter scale must be positive. CHOOSING to force it to zero, to avoid spurious fits
            pos_scale = np.max(
                [tmp["scale"], 1.0e-9 * np.ones(tmp["scale"].shape)], axis=0
            )
            # Warning, different sign convention than scipy: c = -shape!
            return (tmp["loc"], pos_scale, -tmp["shape"])

        elif self.distrib in ["gaussian"]:
            # The parameter scale must be positive. CHOOSING to force it to zero, to avoid spurious fits
            pos_scale = np.max(
                [tmp["scale"], 1.0e-9 * np.ones(tmp["scale"].shape)], axis=0
            )
            return (tmp["loc"], pos_scale)

        elif self.distrib in ["poisson"]:
            # the location must be integer values
            int_loc = np.array(np.round(tmp["loc"], 0), dtype=int)
            # The parameter mu must be positive. CHOOSING to force it to zero, to avoid spurious fits
            # Besides, when loc and mu are close to zero, the probability of obtaining the value 0 is ~1-mu. Having mu=0 makes any value != 0 infinitely unlikely => setting a threshold on mu at 1.e-9, ie 1 / 1e9 years.
            pos_mu = np.max([tmp["mu"], 1.0e-9 * np.ones(tmp["mu"].shape)], axis=0)
            return (int_loc, pos_mu)

    def _test_coeffs(self, args):
        # warning here, args are the coefficients

        # initialize test
        test = True

        # checking set boundaries on coeffs
        for coeff in self.boundaries_coeffs:
            low, top = self.boundaries_coeffs[coeff]
            cff = args[self.coeffs_names.index(coeff)]
            if (
                np.any(cff < low)
                or np.any(top < cff)
                or np.any(np.isclose(cff, low))
                or np.any(np.isclose(top, cff))
            ):
                test = (
                    False  # out of boundaries, strong signal to negative log likelyhood
                )

        # checking the transformation
        if self.transfo[0]:
            left = args[self.coeffs_names.index("transfo_asymptleft")]
            right = args[self.coeffs_names.index("transfo_asymptright")]
            if left < right:
                if (np.min(self.data) < left) or (right < np.max(self.data)):
                    test = False
            else:
                if (np.max(self.data) > left) or (right > np.min(self.data)):
                    test = False

        # checking that the coefficient in the exponential of the sigmoid evolution is positive
        for param in self.coeffs_names:
            if "_lambda" in param:  # on this parameter, there is a sigmoid evolution.
                pass
                # if args[self.coeffs_names.index(param)] < 0:
                #    test = False

        # has observed with sigmoid evolution that there may be compensation between coefficients, leading to a biased evolution of coefficients. This situation is rare for TXx, but happens often with SMA.
        for param in self.cov["params"]:
            for form_sigm in self.possible_sigmoid_forms:
                if form_sigm in self.cov["cov_" + param + "_form"]:
                    # checking for this parameter of the distribution if the fit of a sigmoid evolution leads to a drift in its param_0 and param_sigmoid_delta_covariate
                    # criteria: if the first guess provided to the optimization function has increased the constant term X times, it is unlikely and there may be a drift.
                    try:
                        if np.abs(
                            args[self.coeffs_names.index(param + "_0")]
                        ) > 10 * np.abs(self.x0[self.coeffs_names.index(param + "_0")]):
                            test = False
                    except AttributeError:
                        # test on coefficients but to define first guess, not yet during fit. Thus, has not & cannot do this test yet.
                        pass

        return test

    def _test_evol_params(self, data_fit, args):
        # warning, args are here the evolution of parameters. And args[2] is 'c', not 'shape'
        # warning n2: because of the option 'transfo', data_fit is not necessarily self.data

        if self.distrib in ["GEV"]:  # len(args) == 3:
            loc, scale, c = args
            do_c = True

        elif self.distrib in ["gaussian"]:  # len(args) == 2:
            loc, scale = args
            do_c = False

        elif self.distrib in ["poisson"]:
            loc, mu = args
            do_c = False

        else:
            raise Exception("Distribution not prepared here")

        # initialize test
        test = True

        if do_c:
            # test of the support of the GEV: is there any data out of the corresponding support?
            # The support of the GEV is: [ loc - scale/shape ; +inf [ if shape>0  and ] -inf ; loc - scale/shape ] if shape<0
            # NB: checking the support with only '<' is not enough, not even '<='. Has encountered situations where only '<' AND 'isclose' avoids data points to be too close from boundaries, leading to unrealistic values in the ensuing processes.
            if np.any(scale + c * (loc - data_fit) <= 0) or np.any(
                np.isclose(0, scale + c * (loc - data_fit))
            ):  # rewritten for simplicity as scale + c * (loc - data) > 0
                test = False
            # if type(c) == np.float64:# no covariants on shape
            #    if -c > 0:# support of this GEV is [ loc - scale/shape ; +inf [
            #        if np.any( data_fit < loc - scale/(-c) )  or  np.any(np.isclose( data_fit , loc - scale/(-c) )):
            #            test = False
            #    elif np.isclose( -c, 0):# support of this GEV is ] -inf ; +inf [
            #        pass
            #    elif -c < 0:# support of this GEV is ] -inf ; loc - scale/shape ]
            #        if np.any( loc - scale/(-c) < data_fit )  or  np.any(np.isclose( data_fit , loc - scale/(-c) )):
            #            test = False
            # else:# the parameter shape has the same size as inputs
            #    indInf, indSup = np.where(-c < 0)[0], np.where(-c > 0)[0]
            #    if np.any( data_fit[indSup] < (loc - scale/(-c))[indSup] )  or  np.any(np.isclose( data_fit[indSup] , (loc - scale/(-c))[indSup] )):
            #        test = False
            #    if np.any( (loc - scale/(-c))[indInf] < data_fit[indInf] )  or  np.any(np.isclose( data_fit[indInf] , (loc - scale/(-c))[indInf] )):
            #        test = False

            # comparing to prescribed borders for shape
            if test:  # if false, no need to test
                low, high = self.boundaries_params["shape"]
                if (
                    np.any(-c < low)
                    or np.any(high < -c)
                    or np.any(np.isclose(-c, low))
                    or np.any(np.isclose(high, -c))
                ):
                    test = False  # out of boundaries, strong signal to negative log likelyhood

        # scale should be strictly positive, or respect any other set boundaries
        if test:  # if false, no need to test
            if self.distrib in ["GEV", "gaussian"]:
                low, high = self.boundaries_params["scale"]
                if np.any(scale < low) or np.any(
                    high < scale
                ):  # or np.any(np.isclose(scale , low)) or np.any(np.isclose(high , scale)): # trying without the isclose, cases with no evolutions, ie scale~=0
                    test = False  # out of boundaries, strong signal to negative log likelyhood
            elif self.distrib in ["poisson"]:
                low, high = self.boundaries_params["mu"]
                if np.any(mu < low) or np.any(
                    high < mu
                ):  # or np.any(np.isclose(scale , low)) or np.any(np.isclose(high , scale)): # trying without the isclose, cases with no evolutions, ie scale~=0
                    test = False  # out of boundaries, strong signal to negative log likelyhood

        # location should respect set boundaries
        if test:  # if false, no need to test
            low, high = self.boundaries_params["loc"]
            if np.any(loc < low) or np.any(
                high < loc
            ):  # or np.any(np.isclose(loc , low)) or np.any(np.isclose(high , loc)):
                test = (
                    False  # out of boundaries, strong signal to negative log likelyhood
                )

        return test

    # --------------------

    # --------------------
    # FIT
    def loglike(self, args):

        if self._test_coeffs(args):
            # transformation?
            if self.transfo[0]:
                m1 = args[self.coeffs_names.index("transfo_asymptleft")]
                m2 = args[self.coeffs_names.index("transfo_asymptright")]
                # new data that will be used
                if self.transfo[1] in ["generalizedlogistic", "generalizedalgebraic"]:
                    alpha = args[self.coeffs_names.index("transfo_alpha")]
                    data_fit = sigmoid_transf(
                        self.data,
                        left=m1,
                        right=m2,
                        type_sigm=self.transfo[1],
                        alpha=alpha,
                    )
                else:
                    data_fit = sigmoid_transf(
                        self.data, left=m1, right=m2, type_sigm=self.transfo[1]
                    )
            else:
                data_fit = self.data

            # log-likelihood
            p_args = self._parse_args(args)

            # computing quality of the set of coefficients
            if self._test_evol_params(data_fit, p_args):
                # if here, then everything looks fine
                if self.distrib in ["GEV"]:
                    prior = self._prior_shape(p_args[2])  # .sum()
                    ll = ss.genextreme.logpdf(
                        data_fit, loc=p_args[0], scale=p_args[1], c=p_args[2]
                    ).sum()
                    out = ll + prior

                elif self.distrib in ["gaussian"]:
                    out = ss.norm.logpdf(data_fit, loc=p_args[0], scale=p_args[1]).sum()

                elif self.distrib in ["poisson"]:
                    out = ss.poisson.logpmf(data_fit, loc=p_args[0], mu=p_args[1]).sum()

                else:
                    raise Exception("This distribution has not been prepared.")
                # if ll+prior > 0, want to reduce output if quality transformation decrease
                # if ll+prior < 0, want to reduce further output if quality transformation decrease
                return out

            else:  # something wrong with the evolution of parameters
                return -np.inf

        else:  # something wrong with the coefficients
            return -np.inf

    def neg_loglike(self, args):
        # negative log likelihood (for fit)
        # just in case used out of the optimization function, to evaluate the quality of the fit
        if type(args) == dict:
            args = [args[kk] for kk in args]

        return -self.loglike(args)

    def translate_m_sol(self, mx):
        sol = {}
        for cf in self.coeffs_names:
            sol[cf] = mx[self.coeffs_names.index(cf)]
        return sol

    def fit(self):

        # checking if actually need to fit...
        if np.any(np.isnan(self.data)):
            return self.translate_m_sol(np.nan * np.ones(len(self.coeffs_names)))

        else:
            # Before fitting, need a good first guess, using 'find_fg'.

            # trying first with the best choice for a first guess.
            self.eval_fg0_best = True
            # making a correct first guess
            self.x0 = self.find_fg()

            # fitting
            m = minimize(
                self.neg_loglike,
                x0=self.x0,
                method=self.method_fit,
                options={
                    "maxfev": self.maxfev,
                    "maxiter": self.maxiter,
                    self.name_xtol: self.xtol_req,
                },
            )

            # Checking if failed. May have to choose the second option for difficult fits. (only "if self.distrib in [ 'GEV' ]", try if can remove it given all new improvements?)
            if m.success == False:
                self.eval_fg0_best = False
                # making a correct first guess
                self.x0 = self.find_fg()

                # make a first fit (to initialize walkers)
                m = minimize(
                    self.neg_loglike,
                    x0=self.x0,
                    method=self.method_fit,
                    options={
                        "maxfev": self.maxfev,
                        "maxiter": self.maxiter,
                        self.name_xtol: self.xtol_req,
                    },
                )

                # checking if that one failed as well
                if self.error_failedfit and (m.success == False):
                    raise Exception(
                        "The fast detrend provides with a valid first guess, but not good enough."
                    )

            return self.translate_m_sol(m.x)

    # --------------------
