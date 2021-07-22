# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train local variability module of MESMER.
"""


import os

import joblib
import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.tsa.ar_model import AutoReg

from .train_utils import train_l_prepare_X_y_wgteq


def train_lv(preds, targs, esm, cfg, save_params=True, aux={}, params_lv={}):
    """Derive local variability (i.e., natural variabiliy) parameters.

    Parameters
    ----------
    preds : dict
        empty dictionary if none, else nested dictionary of predictors with keys

        - [pred][scen]  (1d/ 2d arrays (time)/(run, time) of predictor for specific
        scenario)
    targs : dict
        nested dictionary of targets with keys

        - [targ][scen] (3d array (run, time, gp) of target for specific scenario)
    esm : str
        associated Earth System Model (e.g., "CanESM2" or "CanESM5")
    cfg : module
        config file containing metadata
    save_params : bool, optional
        determines if parameters are saved or not, default = True
    aux : dict, optional
        provides auxiliary variables needed for lv method at hand

        - [var] (Xd arrays of auxiliary variable)
    params_lv : dict, optional
        pass the params_lv dict, if it already exists so that builds upon that one

    Returns
    -------
    params_lv : dict
        dictionary of local variability paramters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - ["part_model_in_lt"] (states if part of the model is saved in params_lt, bool)
        - ["method_lt_each_gp_sep"] (states if local trends method is applied to each
          grid point separately, bool)
        - [xx] (additional params depend on employed lv method)

    Notes
    -----
    - Assumptions:
        - all targets use same approach and same predictors
        - each predictor and each target has the same scenarios as keys
        - all available scenarios are used for training
        - identified parameters are valid for all training scenarios
        - if historical data is used for training, it has its own scenario
        - need to pass the params_lv dict if it already exists so that can continue to
          build on it
    - Disclaimer:
        - currently no method with preds implemented; but already have in there for
          consistency
    - TODO:
        - add ability to weight samples differently than equal weight for each scenario
        in AR process

    """

    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # because same approach for each targ
    pred_names = list(preds.keys())

    # specify necessary variables from config file
    wgt_scen_tr_eq = cfg.wgt_scen_tr_eq

    preds_lv = []
    # check if any preds from pr
    if len(params_lv) > 0:
        [preds_lv.append(pred) for pred in params_lv["preds"]]
    # for now only gv implemented, but could easily extend to rv (regional) lv (local)
    # if wanted such preds
    for pred in pred_names:
        if "gv" in pred:
            preds_lv.append(pred)
    # add new predictors to params_lv
    if len(params_lv) > 0:
        params_lv["preds"] = preds_lv

    method_lv = cfg.methods[targ_name]["lv"]

    scenarios_tr = list(targs[targ_name].keys())

    # prepare predictors and targets
    X, y, wgt_scen_eq = train_l_prepare_X_y_wgteq(preds, targs)
    if wgt_scen_tr_eq is False:
        wgt_scen_eq[:] = 1  # each sample same weight

    if len(params_lv) == 0:
        print("Initialize params_lv dictionary")
        params_lv = {}
        params_lv["targs"] = targ_names
        params_lv["esm"] = esm
        params_lv["method"] = method_lv
        params_lv["preds"] = preds_lv
        params_lv["scenarios"] = scenarios_tr
        params_lv["part_model_in_lt"] = False

    if "AR1_sci" in method_lv and wgt_scen_tr_eq:

        # assumption: target values I feed in here is already ready for AR1_sci method
        # if were to add any other method before (ie introduce Link et al method for
        # large-scale teleconnections), would have to execute it first & fit this one on
        # residuals

        params_lv = train_lv_AR1_sci(params_lv, targs, y, wgt_scen_eq, aux, cfg)
    else:
        raise ValueError(
            "The chosen method and / or weighting approach is not implemented."
        )

    # overwrites lv module if already exists, i.e., assumption: always lt before lv
    if save_params:
        dir_mesmer_params = cfg.dir_mesmer_params
        dir_mesmer_params_lv = dir_mesmer_params + "local/local_variability/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_lv):
            os.makedirs(dir_mesmer_params_lv)
            print("created dir:", dir_mesmer_params_lv)
        filename_parts = [
            "params_lv",
            method_lv,
            *preds_lv,
            *targ_names,
            esm,
            *scenarios_tr,
        ]
        filename_params_lv = dir_mesmer_params_lv + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_lv, filename_params_lv)

    return params_lv


def train_lv_AR1_sci(params_lv, targs, y, wgt_scen_eq, aux, cfg):
    """Derive parameters for AR(1) process with spatially-correlated innovations.

    Parameters
    ----------
    params_lv : dict
        dictionary with the trained local variability parameters

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - [xx] (additional keys depend on employed method)
    targs : dict
        nested dictionary of targets with keys

        - [targ][scen] with 3d arrays (run, time, gp)
    y : np.ndarray
        3d array (sample, gp, targ) of targets
    wgt_scen_eq : np.ndarray
        1d array (sample) of sample weights
    aux : dict
        provides auxiliary variables needed for lv method at hand

        - ["phi_gc"] (Xd arrays of auxiliary variable)
    cfg : module
        config file containing metadata

    Returns
    -------
    emus_lv : dict
        local variability emulations dictionary with keys

        - [scen] (2d array  (emu, time, gp) of local variability in response to global
          variability emulation time series)

    Notes
    -----
    - Assumptions:
        - do for each target variable independently
        - the variability is Gaussian
        - each scenario receives the same weight during training
    - Potential TODO:
        - add possibility to account for cross-correlation between different variables
          (i.e., joint instead of independent emulation)

    """

    print("Derive parameters for AR(1) processes with spatially correlated innovations")
    # AR(1)
    params_lv["AR1_int"] = {}
    params_lv["AR1_coef"] = {}
    params_lv["AR1_std_innovs"] = {}
    params_lv["L"] = {}  # localisation radius
    # empirical cov matrix of the local variability trained on here
    params_lv["ecov"] = {}
    params_lv["loc_ecov"] = {}  # localized empirical cov matrix
    # localized empirical cov matrix of the innovations of the AR(1) process
    params_lv["loc_ecov_AR1_innovs"] = {}

    # largely ignore prepared targets and use original ones instead because in original
    # easier to loop over individ runs / scenarios
    targ_names = list(targs.keys())
    scenarios_tr = list(targs[targ_names[0]].keys())
    nr_scens = len(scenarios_tr)

    # fit parameters for each target individually
    for t, targ_name in enumerate(targ_names):
        targ = targs[targ_name]
        nr_gps = y.shape[1]
        y_targ = y[:, :, t]

        # AR(1)
        params_lv["AR1_int"][targ_name] = np.zeros(nr_gps)
        params_lv["AR1_coef"][targ_name] = np.zeros(nr_gps)
        params_lv["AR1_std_innovs"][targ_name] = np.zeros(nr_gps)

        for scen in scenarios_tr:
            nr_runs, nr_ts, nr_gps = targ[scen].shape
            AR1_int_runs = np.zeros(nr_gps)
            AR1_coef_runs = np.zeros(nr_gps)
            AR1_std_innovs_runs = np.zeros(nr_gps)

            for run in np.arange(nr_runs):
                for gp in np.arange(nr_gps):
                    AR1_model = AutoReg(
                        targ[scen][run, :, gp], lags=1, old_names=False
                    ).fit()
                    AR1_int_runs[gp] += AR1_model.params[0] / nr_runs
                    AR1_coef_runs[gp] += AR1_model.params[1] / nr_runs
                    # sqrt of variance = standard deviation
                    AR1_std_innovs_runs[gp] += np.sqrt(AR1_model.sigma2) / nr_runs

            params_lv["AR1_int"][targ_name] += AR1_int_runs / nr_scens
            params_lv["AR1_coef"][targ_name] += AR1_coef_runs / nr_scens
            params_lv["AR1_std_innovs"][targ_name] += AR1_std_innovs_runs / nr_scens

        # determine localization radius, empirical cov matrix, and localized ecov matrix
        (
            params_lv["L"][targ_name],
            params_lv["ecov"][targ_name],
            params_lv["loc_ecov"][targ_name],
        ) = train_lv_find_localized_ecov(y_targ, wgt_scen_eq, aux, cfg)

        # ATTENTION: STILL NEED TO CHECK IF THIS IS TRUE. I UNFORTUNATELY LEARNED THAT I
        # WROTE THIS FORMULA DIFFERENTLY IN THE ESD PAPER!!!!!!! (But I am pretty sure
        # that code is correct and the error is in the paper)
        # compute localized cov matrix of the innovations of the AR(1) process
        loc_ecov_AR1_innovs = np.zeros(params_lv["loc_ecov"][targ_name].shape)
        for i in np.arange(nr_gps):
            for j in np.arange(nr_gps):
                loc_ecov_AR1_innovs[i, j] = (
                    np.sqrt(1 - params_lv["AR1_coef"][targ_name][i] ** 2)
                    * np.sqrt(1 - params_lv["AR1_coef"][targ_name][j] ** 2)
                    * params_lv["loc_ecov"][targ_name][i, j]
                )

        params_lv["loc_ecov_AR1_innovs"][targ_name] = loc_ecov_AR1_innovs
        # derive the localized ecov of the innovations of the AR(1) process (ie the one
        # I will later draw innovs from)

    return params_lv


def train_lv_find_localized_ecov(y, wgt_scen_eq, aux, cfg):
    """
    Find suitable localization radius for empirical covariance matrix and derive
    localized empirical cov matrix.

    Parameters
    ----------
    y : np.ndarray
        2d array (sample, gp) of specific target
    wgt_scen_eq : np.ndarray
        1d array (sample) of sample weights
    aux : dict
        provides auxiliary variables needed for lv method at hand

        - ["phi_gc"] (dict with localisation radii as keys and each containing a 2d
        array (gp, gp) of of Gaspari-Cohn correlation matrix
    cfg : module
        config file containing metadata

    Returns
    -------
    L_sel : numpy.int64
        selected localization radius
    ecov : np.ndarray
        2d empirical covariance matrix array (gp, gp)
    loc_ecov : np.ndarray
        2d localized empirical covariance matrix array (gp, gp)

    Notes
    -----
    - Function could also handle determining ecov of several variables but would all
      have to be passed in same 2d y array (with corresponding wgt_scen_eq,
      aux["phi_gc"] shapes)

    """

    # derive the indices for the cross validation
    max_iter_cv = cfg.max_iter_cv
    nr_samples = y.shape[0]
    nr_it = np.min([nr_samples, max_iter_cv])
    idx_cv_out = np.zeros([nr_it, nr_samples], dtype=bool)
    for i in np.arange(nr_it):
        idx_cv_out[i, i::max_iter_cv] = True

    # spatial cross-correlations with specified cross val folds
    L_set = np.sort(list(aux["phi_gc"].keys()))  # the Ls to loop through

    llh_max = -10000
    llh_cv_sum = {}
    idx_L = 0
    L_sel = L_set[idx_L]
    idx_break = False

    while (idx_break is False) and (L_sel < L_set[-1]):
        # experience tells: once stop selecting larger loc radii, will not start again
        # better to stop once max is reached (to limit computational effort + amount of
        # singular matrices)
        L = L_set[idx_L]
        llh_cv_sum[L] = 0

        for it in np.arange(nr_it):
            # extract folds
            y_est = y[~idx_cv_out[it]]  # to estimate params
            y_cv = y[idx_cv_out[it]]  # to crossvalidate the estimate
            wgt_scen_eq_est = wgt_scen_eq[~idx_cv_out[it]]
            wgt_scen_eq_cv = wgt_scen_eq[idx_cv_out[it]]

            # compute ecov and likelihood of out fold to be drawn from it
            ecov = np.cov(y_est, rowvar=False, aweights=wgt_scen_eq_est)
            loc_ecov = aux["phi_gc"][L] * ecov
            # we want the mean of the res to be 0
            mean_0 = np.zeros(aux["phi_gc"][L].shape[0])

            llh_cv_each_sample = multivariate_normal.logpdf(
                y_cv, mean=mean_0, cov=loc_ecov, allow_singular=True
            )
            # allow_singular = True because stms ran into singular matrices
            # ESMs eg affected: CanESM2, CanESM5, IPSL-CM5A-LR, MCM-UA-1-0
            # -> reassuring that saw that in these ESMs L values where matrix
            # is not singular yet can end up being selected

            # each cv sample gets its own likelihood -> can sum them up for overall
            # likelihood
            # sum over all samples = wgt average * nr_samples
            llh_cv_fold_sum = np.average(
                llh_cv_each_sample, weights=wgt_scen_eq_cv
            ) * len(wgt_scen_eq_cv)

            # add to full sum over all folds
            llh_cv_sum[L] += llh_cv_fold_sum

        idx_L += 1

        if llh_cv_sum[L] > llh_max:
            L_sel = L
            llh_max = llh_cv_sum[L]
            print("Newly selected L =", L_sel)
        else:
            print("Final selected L =", L_sel)
            idx_break = True

    ecov = np.cov(y, rowvar=False, aweights=wgt_scen_eq)
    loc_ecov = aux["phi_gc"][L_sel] * ecov

    return L_sel, ecov, loc_ecov
