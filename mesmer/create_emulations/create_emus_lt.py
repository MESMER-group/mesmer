"""
mesmer.create_emulations.create_emus_lt
===================
Functions to create local trends emulations with MESMER.


Functions:
    create_emus_lt()

"""


import os

import joblib
import numpy as np


def create_emus_lt(params_lt, preds_lt, cfg, scenarios="emus", save_emus=True):
    """Create local trends (i.e., forced response) emulations for given parameter set and predictors.

    Args:
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
    - preds_lt (dict): nested dictionary of predictors for local trends with keys
        [pred][scen] with 1d/2d arrays (time)/(run,time)
    - cfg (module): config file containnig metadata
    - scenarios (str), optional: determines if local trends are created for the emulation or training scenarios
    - save_emus (bool, optional): determines if parameters are saved or not, default = True

    Returns:
    - emus_lt (dict): local trend emulations nested dictionary with keys
        [scen]['targ'] (2d array (time,gp) of local trend emulations)

    General remarks:
    - Assumption:   - same predictors for each target
    - TODO: split into subfunctions to increase readability once more approaches to choose from are added (see e.g. train_lt())

    """

    # specify necessary variables from config file
    if scenarios == "emus":
        scenarios_emus = cfg.scenarios_emus
        scen_name_emus = cfg.scen_name_emus
    elif scenarios == "tr":
        scenarios_emus = params_lt["scenarios"]
        if cfg.hist_tr:  # check whether historical data was used in training
            scen_name_emus = "hist_" + "_".join(scenarios_emus)
        else:
            scen_name_emus = "_".join(scenarios_emus)

    dir_mesmer_emus = cfg.dir_mesmer_emus

    # check predictors
    pred_names = list(preds_lt.keys())
    if pred_names != params_lt["preds"]:  # check if correct predictors
        print("Wrong predictors were passed. The emulations cannot be created.")

    emus_lt = {}
    for scen in scenarios_emus:
        emus_lt[scen] = {}
        nr_ts = len(
            preds_lt[pred_names[0]][scen]
        )  # nr_ts could vary for different scenarios but is the same for all predictors
        for targ in params_lt["targs"]:
            if (
                params_lt["method"] == "OLS" and params_lt["method_each_gp_sep"]
            ):  # assumption: coefs are the same across scens
                nr_gp = len(params_lt["full_model"])
                emus_lt[scen][targ] = np.zeros([nr_ts, nr_gp])  # nr_ts x nr_gp
                for gp in params_lt["full_model"].keys():
                    emus_lt[scen][targ][:, gp] = (
                        sum(
                            [
                                params_lt["coef_" + pred][targ][gp]
                                * preds_lt[pred][scen]
                                for pred in params_lt["preds"]
                            ]
                        )
                        + params_lt["intercept"][targ][gp]
                    )

    # save the local trends emulation if requested
    if save_emus:
        dir_mesmer_emus_lt = dir_mesmer_emus + "local/local_trends/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_lt):
            os.makedirs(dir_mesmer_emus_lt)
            print("created dir:", dir_mesmer_emus_lt)
        filename_parts = [
            "emus_lt",
            params_lt["ens_type"],
            params_lt["method"],
            *params_lt["preds"],
            *params_lt["targs"],
            params_lt["esm"],
            scen_name_emus,
        ]
        filename_emus_lt = dir_mesmer_emus_lt + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_lt, filename_emus_lt)

    return emus_lt
