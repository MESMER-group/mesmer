# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train local trends module of MESMER.
"""


import xarray as xr

from mesmer.calibrate_mesmer.train_utils import (
    get_scenario_weights,
    stack_predictors_and_targets,
)
from mesmer.io.save_mesmer_bundle import save_mesmer_data
from mesmer.stats import LinearRegression


def train_lt(preds, targs, esm, cfg, save_params=True):
    """
    Derive local trends (i.e., forced response) parameters for given ESM for given set
    of targets and predictors.

    Parameters
    ----------
    preds : dict
        nested dictionary of predictors with keys

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

    Returns
    -------
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] additional params depend on method employed
        - ["full_model_contains_lv"] (whether the full model contains part of the local
          variability module, bool)
    params_lv : dict, optional
        dictionary of local variability parameters which are derived together with the
        local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
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
    - TODO:
        - find better way to deal with the assumption that local trend predictors belong
          before local variability predictors (e.g., add check on whether this
          assumption is fulfilled or rewrite code such that no longer necessary)

    """

    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # because same approach for each targ
    pred_names = list(preds.keys())

    # specify necessary variables from config file
    wgt_scen_tr_eq = cfg.wgt_scen_tr_eq
    # This code will ever only work with a single target variable
    method_lt = cfg.methods[targ_name]["lt"]
    method_lv = cfg.methods[targ_name]["lv"]
    method_lt_each_gp_sep = cfg.method_lt_each_gp_sep

    preds_lt = []
    preds_lv = []
    # for now only gt/gv implemented, but could easily extend to rt/rv (regional) lt/lv
    # (local)  if wanted such preds
    for pred in pred_names:
        if "gt" in pred:
            preds_lt.append(pred)
        elif "gv" in pred:
            preds_lv.append(pred)

    scenarios_tr = list(targs[targ_name].keys())

    # initialize parameters dictionary and fill in the metadata which does not depend on
    # the applied method
    params_lt = {}
    params_lt["targs"] = targ_names
    params_lt["esm"] = esm
    params_lt["method"] = method_lt
    params_lt["method_each_gp_sep"] = method_lt_each_gp_sep
    params_lt["preds"] = preds_lt
    params_lt["scenarios"] = scenarios_tr

    # check if parameters for local variability module have been derived too, if yes,
    # initialize params_lv
    if method_lt in method_lv:
        params_lt["full_model_contains_lv"] = True

        params_lv = {}
        params_lv["targs"] = targ_names
        params_lv["esm"] = esm
        params_lv["method"] = method_lv
        params_lv["preds"] = preds_lv
        params_lv["scenarios"] = scenarios_tr
        params_lv["part_model_in_lt"] = True
        params_lv["method_lt_each_gp_sep"] = method_lt_each_gp_sep

    else:
        params_lt["full_model_contains_lv"] = False
        params_lv = {}  # only initiate empty dictionary

    # prepare predictors and targets such that they can be ingested into the training
    # function
    X, y = stack_predictors_and_targets(preds, targs)
    # TODO: use DataArray objects throughout the code
    X = {key: xr.DataArray(value, dims="sample") for key, value in X.items()}
    y = {key: xr.DataArray(value, dims=("sample", "cell")) for key, value in y.items()}

    wgt_scen_eq = get_scenario_weights(targs[targ_name])
    # TODO: use DataArray objects throughout the code
    wgt_scen_eq = xr.DataArray(wgt_scen_eq, dims="sample")

    # prepare weights for individual runs
    if wgt_scen_tr_eq is False:
        wgt_scen_eq[:] = 1
        # if each scen does not get the same weight, each sample gets it instead

    # train the full model + save it (may also contain lv module parts)
    if method_lt_each_gp_sep and method_lt == "OLS":

        # initialize the regression coefficient dictionaries
        params_lt["intercept"] = {}
        for pred in params_lt["preds"]:
            params_lt[f"coef_{pred}"] = {}
        for pred in params_lv["preds"]:
            params_lv[f"coef_{pred}"] = {}

        # NOTE: atm only one target can be and is present
        for targ in params_lt["targs"]:
            lr = LinearRegression()
            lr.fit(predictors=X, target=y[targ], dim="sample", weights=wgt_scen_eq)
            params = lr.params

            params_lt["intercept"][targ] = params.intercept.values

            for pred in params_lt["preds"]:
                params_lt[f"coef_{pred}"][targ] = params[pred].values

            for pred in params_lv["preds"]:
                params_lv[f"coef_{pred}"][targ] = params[pred].values
    else:
        raise NotImplementedError()

    # save the local trend parameters if requested
    if save_params:
        save_mesmer_data(
            params_lt,
            cfg.dir_mesmer_params,
            "local",
            "local_trends",
            filename_parts=[
                "params_lt",
                method_lt,
                *preds_lt,
                *targ_names,
                esm,
                *scenarios_tr,
            ],
        )

        # check if local variability parameters need to be saved too
        # overwrites lv module if already exists, i.e., assumption: lt before lv
        if len(params_lv) > 0:
            save_mesmer_data(
                params_lv,
                cfg.dir_mesmer_params,
                "local",
                "local_variability",
                filename_parts=[
                    "params_lv",
                    method_lv,
                    *preds_lv,
                    *targ_names,
                    esm,
                    *scenarios_tr,
                ],
            )

    if len(params_lv) > 0:
        return params_lt, params_lv
    else:
        return params_lt
