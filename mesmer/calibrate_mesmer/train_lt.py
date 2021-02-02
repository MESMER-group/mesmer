"""
mesmer.calibrate_mesmer.train_lt
===================
Functions to train local trends module of MESMER.


Functions:
    train_lt()
    train_lt_extract_additional_params_OLS()
    train_lt_prepare_X_y_wgteq()

"""


import copy
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


def train_lt(preds, targs, esm, cfg, save_params=True, res_lt=False):
    """Derive local trends (i.e., forced response) parameters for given ESM for given set of tragets and predictors.

    Args:
    - preds (dict): nested dictionary of predictors with keys
        [pred][scen] with 1d/2d arrays (time)/(run,time)
    - targs (dict): nested dictionary of targets with keys
        [targ][scen] with 3d arrays (run,time,gp)
    - esm (str): associated Earth System Model (e.g., 'CanESM2' or 'CanESM5')
    - cfg (module): config file containnig metadata
    - save_params (bool, optional): determines if parameters are saved or not, default = True
    - res_lt (bool, optional): determines if residual is fitted on or full time series

    Returns:
    - params_lt (dict): dictionary with the trained local trend parameters
        ['targs'] (emulated variables, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
        ['method_each_gp_sep'] (states if method is applied to each grid point separately,bool)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (emission scenarios used for training, list of strs)
        [xx] additional params depend on method employed, specified in train_gt_T_enstype_method() function
        ['full_model_contains_lv'] (whether the full model contains part of the local variability module, bool)
    - params_lv (dict, optional): dictionary of local variability paramters which are derived together with the local trend parameters
        ['targs'] (emulated variables, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (emission scenarios used for training, list of strs)
        ['part_model_in_lt'] (states if part of the model is saved in params_lt, bool)
        ['method_lt_each_gp_sep'] (states if local trends method is applied to each grid point separately, bool)

    General remarks:
    - Assumptions:  - all targets use same approach and same predictors
                    - each predictor and each target has the same scenarios as keys
                    - all available scenarios are used for training
                    - in predictor list: local trend predictors belong before local variability predictors (if there are any)
                    - identified parameters are valid for all training scenarios
                    - if historical data is used for training, it has its own scenario
    - Disclaimer:   - parameters must be saved in case also params_lv are created, otherwise train_lv() cannot find them
                    - not convinced yet whether I really need the res_lt variable

    """

    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # because same approach for each targ
    pred_names = list(preds.keys())

    # specify necessary variables from config file
    ens_type_tr = cfg.ens_type_tr
    hist_tr = cfg.hist_tr
    wgt_scen_tr_eq = cfg.wgt_scen_tr_eq

    preds_lt = []
    preds_lv = []
    # for now only gt/gv implemented, but could easily extend to rt/rv (regional) lt/lv (local)  if wanted such preds
    for pred in pred_names:
        if "gt" in pred:
            preds_lt.append(pred)
        elif "gv" in pred:
            preds_lv.append(pred)

    method_lt = cfg.methods[targ_name]["lt"]  # +"_" + "_".join(preds_lt)
    method_lv = cfg.methods[targ_name]["lv"]  # +"_" + "_".join(preds_lv)
    method_lt_each_gp_sep = cfg.method_lt_each_gp_sep
    dir_mesmer_params = cfg.dir_mesmer_params

    scenarios_tr = list(targs[targ_name].keys())
    scen_name_tr = "_".join(scenarios_tr)

    # initialize parameters dictionary and fill in the metadata which does not depend on the applied method
    params_lt = {}
    params_lt["targs"] = targ_names
    params_lt["esm"] = esm
    params_lt["ens_type"] = ens_type_tr
    params_lt["method"] = method_lt
    params_lt["method_each_gp_sep"] = method_lt_each_gp_sep
    params_lt["preds"] = preds_lt
    params_lt["scenarios"] = scenarios_tr

    # select the method from a dict of fcts
    training_method_func_mapping = {"OLS": LinearRegression().fit}

    # dict could be extended to contain other methods (even actual fcts that I write)
    training_method_lt = training_method_func_mapping[method_lt]

    # prepare predictors and targets such that they can be ingested into the training function
    X, y, wgt_scen_eq = train_lt_prepare_X_y_wgteq(preds, targs, method_lt_each_gp_sep)

    # prepare weights for individual runs

    # train the full model + save it (may also contain lv module parts)
    params_lt["full_model"] = {}
    if wgt_scen_tr_eq:
        for mod_key in y.keys():
            reg = training_method_lt(X, y[mod_key], wgt_scen_eq)
            params_lt["full_model"][mod_key] = copy.deepcopy(
                reg
            )  # needed because otherwise the last coef everywhere

    else:
        for mod_key in y.keys():
            reg = training_method_lt(X, y[mod_key])
            params_lt["full_model"][mod_key] = copy.deepcopy(
                reg
            )  # needed because otherwise the last coef everywhere
    # check if parameters for local variability module have been derived too, if yes, initialize params_lv
    if method_lt in method_lv:
        params_lt["full_model_contains_lv"] = True

        params_lv = {}
        params_lv["targs"] = targ_names
        params_lv["esm"] = esm
        params_lv["ens_type"] = ens_type_tr
        params_lv["method"] = method_lv
        params_lv["preds"] = preds_lv
        params_lv["scenarios"] = scenarios_tr
        params_lv["part_model_in_lt"] = True
        params_lv["method_lt_each_gp_sep"] = method_lt_each_gp_sep

    else:
        params_lt["full_model_contains_lv"] = False
        params_lv = {}  # only initiate empty dictionary

    # select the method to extract additional parameters from a dict of fcts
    extract_additonal_params_func_mapping = {
        "OLS": train_lt_extract_additional_params_OLS
    }
    # if additional lt methods are integrated, dict needs to be extended

    extract_additonal_params_lt = extract_additonal_params_func_mapping[method_lt]

    params_lt, params_lv = extract_additonal_params_lt(params_lt, params_lv)

    # save the local trend paramters if requested
    if save_params:
        dir_mesmer_params_lt = dir_mesmer_params + "local/local_trends/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_lt):
            os.makedirs(dir_mesmer_params_lt)
            print("created dir:", dir_mesmer_params_lt)
        filename_parts = [
            "params_lt",
            ens_type_tr,
            method_lt,
            *preds_lt,
            *targ_names,
            esm,
            scen_name_tr,
        ]
        filename_params_lt = dir_mesmer_params_lt + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_lt, filename_params_lt)

        # check if local variability parameters need to be saved too
        if (
            len(params_lv) > 0
        ):  # overwrites lv module if already exists, i.e., assumption: always lt before lv
            dir_mesmer_params_lv = dir_mesmer_params + "local/local_variability/"
            # check if folder to save params in exists, if not: make it
            if not os.path.exists(dir_mesmer_params_lv):
                os.makedirs(dir_mesmer_params_lv)
                print("created dir:", dir_mesmer_params_lv)
        filename_parts = [
            "params_lv",
            ens_type_tr,
            method_lv,
            *preds_lv,
            *targ_names,
            esm,
            scen_name_tr,
        ]
        filename_params_lv = dir_mesmer_params_lv + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_lv, filename_params_lv)

    if len(params_lv) > 0:
        return params_lt, params_lv
    else:
        return params_lt


def train_lt_extract_additional_params_OLS(params_lt, params_lv):
    """Extract additional parameters for the ordinary least squares (OLS) method.

    Args:
    - params_lt (dict): dictionary with the trained local trend parameters
        ['targs'] (emulated variables, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] ('OLS')
        ['method_each_gp_sep'] (states if method is applied to each grid point separately,bool)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (emission scenarios used for training, list of strs)
        ['full_model'] (full local trends model)
    - params_lv (dict): dictionary of local variability paramters which are derived together with the local trend parameters
        if len(params_lv)>0:
            ['targs'] (emulated variables, str)
            ['esm'] (Earth System Model, str)
            ['ens_type'] (ensemble type, str)
            ['method'] (applied method, str)
            ['preds'] (predictors, list of strs)
            ['scenarios'] (emission scenarios used for training, list of strs)
            ['part_model_in_lt'] (states if part of the model is saved in params_lt, bool)
            ['method_lt_each_gp_sep'] (states if local trends method is applied to each grid point separately, bool)

    Returns:
    - params_lt (dict): local trends parameters dictionary with added keys
        ['intercept'][targ] (1d array (gp) of intercept terms for each target)
        ['coef_'+pred][targ] (1d array (gp) of OLS regression coefficient terms for each predictor and each target)
    - params_lv (dict): local variability parameters dictionary with added keys
        if len(params_lv)>0:
            ['coef_'+pred][targ] (1d array (gp) of OLS regression coefficient terms for each predictor and each target)

    """
    nr_mod_keys = len(params_lt["full_model"])  # often grid points but not necessarily
    nr_targs = len(params_lt["targs"])

    # initialize the regression coefficient dictionaries
    params_lt["intercept"] = {}
    for pred in params_lt["preds"]:
        params_lt["coef_" + pred] = {}
    if len(params_lv) > 0:
        for pred in params_lv["preds"]:
            params_lv["coef_" + pred] = {}

    # fill the reg coef dictionaries for each target
    for targ_idx in np.arange(len(params_lt["targs"])):
        targ = params_lt["targs"][targ_idx]

        params_lt["intercept"][targ] = np.zeros(nr_mod_keys)
        for mod_key in params_lt["full_model"].keys():
            params_lt["intercept"][targ][mod_key] = params_lt["full_model"][
                mod_key
            ].intercept_[targ_idx]

        coef_idx = 0  # coefficient index
        for pred in params_lt["preds"]:
            params_lt["coef_" + pred][targ] = np.zeros(nr_mod_keys)
            for mod_key in params_lt["full_model"].keys():
                params_lt["coef_" + pred][targ][mod_key] = params_lt["full_model"][
                    mod_key
                ].coef_[targ_idx, coef_idx]
            coef_idx += 1

        if len(params_lv) > 0:  # assumption: coefs of lv are behind coefs of lt
            for pred in params_lv["preds"]:
                params_lv["coef_" + pred][targ] = np.zeros(nr_mod_keys)
                for mod_key in params_lt["full_model"].keys():
                    params_lv["coef_" + pred][targ][mod_key] = params_lt["full_model"][
                        mod_key
                    ].coef_[targ_idx, coef_idx]
                coef_idx += 1

    return params_lt, params_lv


def train_lt_prepare_X_y_wgteq(preds, targs, method_lt_each_gp_sep):
    """Create single array of predictors and single array of targets.

    Args:
    - preds (dict): nested dictionary of predictors with keys
        [pred][scen] with 1d/2d arrays (time)/(run,time)
    - targs (dict): nested dictionary of targets with keys
        [targ][scen] with 3d arrays (run,time,gp)
    - method_lt_each_gp_sep (bool): determines if method is applied to each grid point separately

    Returns:
    - X (np.ndarray): 2d array (sample,pred) of predictors
    - y (dict): target dictionary with keys
        if method_lt_each_gp_sep is True:
            [gp] 2d array (sample,targ) of targets at every grid point
    - wgt_scen_eq (np.ndarray): 1d array (sample) of sample weights based on equal treatment of each scenario (if scen has more ic member, each sample gets less weight)

    """
    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # because same approach for each targ
    pred_names = list(preds.keys())

    # identify characteristics of the predictors and the targets
    targ = targs[
        targ_name
    ]  # predictors are not influenced by whether there is a single or there are multiple targets
    scens = list(targ.keys())

    # assumption: nr_runs per scen and nr_ts for these runs can vary
    nr_samples = 0
    wgt_scen_eq = []
    for scen in scens:
        nr_runs, nr_ts, nr_gps = targ[scen].shape
        nr_samples_scen = nr_runs * nr_ts
        wgt_scen_eq = np.append(wgt_scen_eq, np.repeat(1 / nr_runs, nr_samples_scen))
        nr_samples += nr_samples_scen

    nr_preds = len(pred_names)
    nr_targs = len(targ_names)

    # derive X (ie array of predictors)
    X = np.zeros([nr_samples, nr_preds])
    for p in np.arange(nr_preds):  # index for predictors
        pred_name = pred_names[p]  # name of predictor p
        s = 0  # index for samples
        pred_raw = preds[pred_name]  # values of predictor p
        for scen in scens:
            if (
                len(pred_raw[scen].shape) == 2
            ):  # if 1 time series per run for predictor (e.g., gv)
                k = (
                    pred_raw[scen].shape[0] * pred_raw[scen].shape[1]
                )  # nr_runs*nr_ts for this specific scenario
                X[s : s + k, p] = pred_raw[scen].flatten()
                s += k
            elif (
                len(pred_raw[scen].shape) == 1
            ):  # if single time series as predictor (e.g. gt): repeat ts as many times as runs available
                nr_runs, nr_ts, nr_gps = targ[scen].shape
                nr_samples_scen = nr_runs * nr_ts
                X[s : s + nr_samples_scen, p] = np.tile(pred_raw[scen], nr_runs)
                s += nr_samples_scen
            else:
                print("Predictors in this shape cannot be processed.")

    # derive y (ie dictionary with arrays of targets)
    y = {}
    if method_lt_each_gp_sep:
        # initialize the target arrays
        for gp in np.arange(nr_gps):
            y[gp] = np.zeros(
                [nr_samples, nr_targs]
            )  # dict with key for each gp (ie key for each model)
        # loop through all target vars (often just a single one)
        for t in np.arange(nr_targs):
            targ_name = targ_names[t]
            targ = targs[targ_name]
            for gp in np.arange(nr_gps):
                s = 0
                for scen in scens:
                    k = (
                        targ[scen].shape[0] * targ[scen].shape[1]
                    )  # nr_runs*nr_ts for this specific scenario
                    y[gp][s : s + k, t] = targ[scen][:, :, gp].flatten()
                    s += k
    else:
        print(
            "No method for a single lt model is currently implemented. (If added, should also use a dict.)"
        )

    return X, y, wgt_scen_eq
