# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to create local trends emulations with MESMER.
"""


import os

import joblib
import numpy as np


def create_emus_lt(params_lt, preds_lt, cfg, concat_h_f=False, save_emus=True):
    """
    Create local trends (i.e., forced response) emulations for given parameter set and
    predictors.

    Parameters
    ----------
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] (additional params depend on method employed)
        - ["full_model_contains_lv"] (whether the full model contains part of the local
          variability module, bool)
    preds_lt : dict
        nested dictionary of predictors for local trends with keys

        - [pred][scen] (1d/2d arrays (time)/(run, time) of predictor for specific scenario)
    cfg : module
        config file containing metadata
    concat_h_f : bool, optional
        determines if historical and future time period is concatenated into a single
        emulation or not, default = False (must be set to False if no historical data
        provided)
    save_emus : bool, optional
        determines if parameters are saved or not, default = True

    Returns
    -------
    emus_lt : dict
        local trend emulations nested dictionary with keys

        - [scen]["targ"] (2d array (time, gp) of local trend emulations)

    Notes
    -----
    - Assumptions:
        - same predictors for each target
        - if historical time period is included in predictors, it has its own dictionary
          key
        - if historical time period was included in training, it has its own scenario
        - either historical period is included for every scenario or for no scenario
    - Potential TODO:
        - evaluate if really need / want concat_h_f or if I want output to be determined
          by shape predictors

    """
    # specify necessary variables from config file
    if save_emus:
        dir_mesmer_emus = cfg.dir_mesmer_emus

    # derive necessary scenario names
    pred_names = list(preds_lt.keys())
    scenarios_emus = list(preds_lt[pred_names[0]].keys())

    if concat_h_f:
        if scenarios_emus[0] == "hist":
            scens_out_f = scenarios_emus[1:]
            scens_out = ["h-" + s for s in scens_out_f]
        else:
            raise ValueError("This combination is not supported.")
    else:
        if "h-" in scenarios_emus[0]:
            scens_out_f = list(map(lambda x: x.replace("h-", ""), scenarios_emus))
            scens_out = ["hist"] + scens_out_f
        else:
            scens_out = scens_out_f = scenarios_emus

    # check predictors
    if pred_names != params_lt["preds"]:  # check if correct predictors
        raise ValueError(
            "Wrong predictors were passed. The emulations cannot be created."
        )

    # select the method from a dict of fcts
    create_emus_method_func_mapping = {"OLS_each_gp_sep": create_emus_OLS_each_gp_sep}
    # extend dict if add more methods

    # dict could be extended to contain other methods (even actual fcts that I write)
    method_lt = params_lt["method"]
    if params_lt["method_each_gp_sep"]:
        method_lt = method_lt + "_each_gp_sep"
    else:
        raise ValueError(
            f"No such method ({params_lt['method_each_gp_sep']}) is currently implemented."
        )

    create_emus_method_lt = create_emus_method_func_mapping[method_lt]

    # create emulations
    emus_lt = {}
    if concat_h_f:
        lt_hist = create_emus_method_lt(params_lt, preds_lt, "hist")
        for scen_out, scen_out_f in zip(scens_out, scens_out_f):
            lt_scen_f = create_emus_method_lt(params_lt, preds_lt, scen_out_f)
            emus_lt[scen_out] = {}
            for targ in params_lt["targs"]:
                emus_lt[scen_out][targ] = np.concatenate(
                    [lt_hist[targ], lt_scen_f[targ]]
                )
    else:
        for scen_out in scens_out:
            emus_lt[scen_out] = create_emus_method_lt(params_lt, preds_lt, scen_out)

    # save the local trends emulation if requested
    if save_emus:
        dir_mesmer_emus_lt = dir_mesmer_emus + "local/local_trends/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_lt):
            os.makedirs(dir_mesmer_emus_lt)
            print("created dir:", dir_mesmer_emus_lt)
        filename_parts = [
            "emus_lt",
            params_lt["method"],
            *params_lt["preds"],
            *params_lt["targs"],
            params_lt["esm"],
            *scens_out,
        ]
        filename_emus_lt = dir_mesmer_emus_lt + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_lt, filename_emus_lt)

    return emus_lt


def create_emus_OLS_each_gp_sep(params_lt, preds_lt, scen):
    """Create local trends with OLS with grid-point-specific predictors

    Parameters
    ----------
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] (additional params depend on method employed)
        - ["full_model_contains_lv"] (whether the full model contains part of the local
          variability module, bool)
    preds_lt : dict
        nested dictionary of predictors for local trends with keys

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific scenario)
    scen : str
        emulated scenario

    Returns
    -------
    emus_lt : dict
        local trend emulations dictionary with keys

        - ["targ"] (2d array (time, gp) of local trend emulations)

    Notes
    -----
    - Assumptions:
        - Coefficients are the same for every scenario

    """

    pred_names = list(preds_lt.keys())
    nr_ts = len(
        preds_lt[pred_names[0]][scen]
    )  # nr_ts could vary for different scenarios but is the same for all predictors

    emus_lt = {}
    for targ in params_lt["targs"]:
        nr_gps = len(params_lt["intercept"][targ])
        emus_lt[targ] = np.zeros([nr_ts, nr_gps])
        for gp in np.arange(nr_gps):
            pred_vals = [
                params_lt["coef_" + pred][targ][gp] * preds_lt[pred][scen]
                for pred in params_lt["preds"]
            ]

            emus_lt[targ][:, gp] = sum(pred_vals) + params_lt["intercept"][targ][gp]

    return emus_lt
