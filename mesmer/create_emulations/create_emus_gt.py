"""
mesmer.create_emulations.create_emus_gt
===================
Functions to create global trend emulations with MESMER.


Functions:
    create_emus_gt()

"""

import os

import joblib


def create_emus_gt(params_gt, cfg, scenarios="emus", save_emus=True):
    """Create global trend (emissions + volcanoes) emulations for specified ensemble type and method.

    Args:
    - params_gt (dict):
        ['targ'] (emulated variable, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (type of ensemble which is emulated, str)
        ['method'] (applied method, str)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (scenarios which are used for training, list of strs)
        ['time'] (1d array of years, np.ndarray)
        [xx] (additional keys depend on employed method and are listed in train_gt_T_ens_type_method() function)
    - cfg (module): config file containnig metadata
    - scenarios (str,optional): determines if training or emulations scenarios are emulated, default = 'emus'
    - save_emus (bool,optional): determines if emulation is saved or not, default = True

    Returns:
    - emus_gt (dict): global trend emulations dictionary with keys
        [scen] (1d array of global trend emulation time series)

    """

    # specify necessary variables from config file
    if scenarios == "emus":
        scenarios_emus = cfg.scenarios_emus
        scen_name_emus = cfg.scen_name_emus
    elif scenarios == "tr":
        scenarios_emus = params_gt["scenarios"]
        if cfg.hist_tr:  # check whether historical data was used in training
            scen_name_emus = "hist_" + "_".join(scenarios_emus)
        else:
            scen_name_emus = "_".join(scenarios_emus)

    dir_mesmer_emus = cfg.dir_mesmer_emus

    # initialize global trend emulations dictionary with scenarios as keys
    emus_gt = {}

    # apply the chosen method
    if "LOWESS" in params_gt["method"]:
        for scen in scenarios_emus:
            emus_gt[scen] = params_gt[scen]["gt"]
    else:
        print("No alternative method is currently implemented.")

    # save the global trend emulation if requested
    if save_emus:
        dir_mesmer_emus_gt = dir_mesmer_emus + "global/global_trend/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_gt):
            os.makedirs(dir_mesmer_emus_gt)
            print("created dir:", dir_mesmer_emus_gt)
        filename_parts = [
            "emus_gt",
            params_gt["ens_type"],
            params_gt["method"],
            *params_gt["preds"],
            params_gt["targ"],
            params_gt["esm"],
            scen_name_emus,
        ]
        filename_emus_gt = dir_mesmer_emus_gt + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_gt, filename_emus_gt)

    return emus_gt
