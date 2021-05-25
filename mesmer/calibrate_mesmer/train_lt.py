# Copyright (c) 2021 ETH Zurich, contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0; see LICENSE or https://www.gnu.org/licenses/
"""
Functions to train local trends module of MESMER.
"""


import copy
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from mesmer.calibrate_mesmer.train_utils import train_l_prepare_X_y_wgteq


def train_lt(preds, targs, esm, cfg, save_params=True):
    """
    Derive local trends (i.e., forced response) parameters for given ESM for given set
    of targets and predictors.

    Parameters
    ----------
    preds : dict
        nested dictionary of predictors with keys

        - [pred][scen]  (1d/ 2d arrays (time)/(run, time) of predictor for specific scenario)
    targs : dict
        nested dictionary of targets with keys

        - [targ][scen] (3d array (run, time, gp) of target for specific scenario)
    esm : str
        associated Earth System Model (e.g., "CanESM2" or "CanESM5")
    cfg : module
        config file containing metadata
    save_params : bool, optional
        determines if parameters are saved or not, default = True

    Returns
    -------
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["ens_type"] (ensemble type, str)
        - ["method"] (applied method, str)
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] additional params depend on method employed
        - ["full_model_contains_lv"] (whether the full model contains part of the local
          variability module, bool)
    params_lv : dict, optional
        dictionary of local variability paramters which are derived together with the
        local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["ens_type"] (ensemble type, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - ["part_model_in_lt"] (states if part of the model is saved in params_lt, bool)
        - ["method_lt_each_gp_sep"] (states if local trends method is applied to each
          grid point separately, bool)

    Notes
    -----
    - Assumptions:
        - all targets use same approach and same predictors
        - each predictor and each target has the same scenarios as keys
        - all available scenarios are used for training
        - in predictor list: local trend predictors belong before local variability
          predictors (if there are any)
        - identified parameters are valid for all training scenarios
        - if historical data is used for training, it has its own scenario
        - either each scenario is given the same weight or each time step
    - Disclaimer:
        - parameters must be saved in case also params_lv are created, otherwise
          train_lv() cannot find them

    """

    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # because same approach for each targ
    pred_names = list(preds.keys())

    # specify necessary variables from config file
    dir_mesmer_params = cfg.dir_mesmer_params
    ens_type_tr = cfg.ens_type_tr
    wgt_scen_tr_eq = cfg.wgt_scen_tr_eq

    preds_lt = []
    preds_lv = []
    # for now only gt/gv implemented, but could easily extend to rt/rv (regional) lt/lv (local)  if wanted such preds
    for pred in pred_names:
        if "gt" in pred:
            preds_lt.append(pred)
        elif "gv" in pred:
            preds_lv.append(pred)

    method_lt = cfg.methods[targ_name]["lt"]
    method_lv = cfg.methods[targ_name]["lv"]
    method_lt_each_gp_sep = cfg.method_lt_each_gp_sep

    scenarios_tr = list(targs[targ_name].keys())

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
    X, y, wgt_scen_eq = train_l_prepare_X_y_wgteq(preds, targs)

    # prepare weights for individual runs

    # train the full model + save it (may also contain lv module parts)
    params_lt["full_model"] = {}
    if wgt_scen_tr_eq is False:
        wgt_scen_eq[
            :
        ] = 1  # if each scen does not get the same weight, each sample gets it instead

    if method_lt_each_gp_sep:
        nr_gps = y.shape[1]
        for gp in np.arange(nr_gps):
            reg = training_method_lt(X, y[:, gp, :], wgt_scen_eq)
            params_lt["full_model"][gp] = copy.deepcopy(
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
            *scenarios_tr,
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
            *scenarios_tr,
        ]
        filename_params_lv = dir_mesmer_params_lv + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_lv, filename_params_lv)

    if len(params_lv) > 0:
        return params_lt, params_lv
    else:
        return params_lt


def train_lt_extract_additional_params_OLS(params_lt, params_lv):
    """Extract additional parameters for the ordinary least squares (OLS) method.

    Parameters
    ----------
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["ens_type"] (ensemble type, str)
        - ["method"] ("OLS")
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - ["full_model"] (full local trends model)
    params_lv : dict
        dictionary of local variability paramters which are derived together with the
        local trend parameters. if len(params_lv)>0 requires

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["ens_type"] (ensemble type, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - ["part_model_in_lt"] (states if part of the model is saved in params_lt, bool)
        - ["method_lt_each_gp_sep"] (states if local trends method is applied to each
          grid point separately, bool)

    Returns
    -------
    params_lt : dict
        local trends parameters dictionary with added keys

        - ["intercept"][targ] (1d array (gp) of intercept terms for each target)
        - ["coef\\_" + pred][targ] (1d array (gp) of OLS regression coefficient terms
          for each predictor and each target)
    params_lv : dict
        local variability parameters dictionary with added keys. if len(params_lv)>0
        returns

        - ["coef\\_" + pred][targ] (1d array (gp) of OLS regression coefficient terms
          for each predictor and each target)
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
    for targ_idx in np.arange(nr_targs):
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
