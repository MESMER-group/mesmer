"""
mesmer.create_emulations.create_emus_gv
===================
Functions to create global variability emulations with MESMER.


Functions:
    create_emus_gv()
    create_emus_gv_AR()

"""


import os

import joblib
import numpy as np


def create_emus_gv(params_gv, cfg, save_emus=True):
    """Create global variablity emulations for specified ensemble type and method.

    Args:
    - params_gv (dict):
        ['targ'] (variable which is emulated, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (type of ensemble which is emulated, str)
        ['method'] (applied method, str)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (scenarios which are used for training, list of strs)
        [xx] (additional keys depend on employed method and are listed in train_gv_T_ens_type_method() function)
    - cfg (module): config file containnig metadata
    - save_emus (bool,optional): determines if emulation is saved or not, default = True

    Returns:
    - emus_gv (dict): global variability emulations dictionary with keys
        [scen] (2d array  (emus x time) of global trend emulation time series)

    """

    # specify necessary variables from config file
    scenarios_emus = cfg.scenarios_emus_v
    scen_name_emus = cfg.scen_name_emus_v
    dir_mesmer_emus = cfg.dir_mesmer_emus

    # set up dictionary for emulations of global variability with emulated scenarios as keys
    emus_gv = {}

    for scen in scenarios_emus:
        # apply the chosen method
        if (
            params_gv["method"] == "AR"
        ):  # for now irrespective of ens_type and scenario. Could still be adapted later if necessary
            emus_gv[scen] = create_emus_gv_AR(params_gv, scen, cfg)
        else:
            print("No alternative method is currently implemented")
            # if the emulations should depend on the scenario, scen needs to be passed to the fct

    # save the global variability emus if requested
    if save_emus:
        dir_mesmer_emus_gv = dir_mesmer_emus + "global/global_variability/"
        # check if folder to save emus in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_gv):
            os.makedirs(dir_mesmer_emus_gv)
            print("created dir:", dir_mesmer_emus_gv)
        filename_parts = [
            "emus_gv",
            params_gv["ens_type"],
            params_gv["method"],
            *params_gv["preds"],
            params_gv["targ"],
            params_gv["esm"],
            scen_name_emus,
        ]
        filename_emus_gv = dir_mesmer_emus_gv + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_gv, filename_emus_gv)

    return emus_gv


def create_emus_gv_AR(params_gv, scen, cfg):
    """Draw global variablity emulations from an AR process.

    Args:
    - params_gv (dict):
        ['targ'] (variable which is emulated, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (type of ensemble which is emulated, str)
        ['method'] (applied method, str)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (scenarios which are used for training, list of strs)
        ['max_lag'] (maximum lag considered when finding suitable AR model, int)
        ['sel_crit'] (selection criterion applied to find suitable AR model, str)
        ['AR_int'] (intercept of the AR model, float)
        ['AR_coefs'] (coefficients of the AR model for the lags which are contained in the selected AR model, list of floats)
        ['AR_lags'] (AR lags which are contained in the selected AR model, list of ints)
        ['AR_std_innovs'] (standard deviation of the innovations of the selected AR model, float)
    - scen (str): emulated scenario
    - cfg (module): config file containnig metadata
    - seed_offset (int): offset for the model-specific seed listed in the cfg file, used if different emulations for each scenario

    Returns:
    - emus_gv (dict): global variability emulations dictionary with keys
        [scen] (2d array  (emus x time) of global trend emulation time series)

    """

    # specify necessary variables from config file
    esm = params_gv["esm"]
    seed = cfg.seed[esm][scen]["gv"]

    nr_ts_emus_v = cfg.nr_ts_emus_v[esm][scen]  # how long emulations
    nr_emus = cfg.nr_emus[esm][scen]  # how many emulations

    # ensure reproducibility
    np.random.seed(seed)

    # buffer so that initial start at 0 does not influence overall result
    buffer = 50

    # re-name params for easier reading of code below
    ar_int = params_gv["AR_int"]
    ar_coefs = params_gv["AR_coefs"]
    ar_lags = params_gv["AR_lags"]

    innovs_emus_gv = np.random.normal(
        loc=0, scale=params_gv["AR_std_innovs"], size=(nr_emus, nr_ts_emus_v + buffer)
    )

    # initialize global variability emulations (dim array: nr_emus x nr_ts)
    emus_gv = np.zeros([nr_emus, nr_ts_emus_v + buffer])

    for i in np.arange(nr_emus):
        # simulate from AR process
        for t in np.arange(ar_lags[-1], len(emus_gv[i])):  # avoid misleading indices
            emus_gv[i, t] = (
                ar_int
                + sum(
                    [
                        ar_coefs[k] * emus_gv[i, t - ar_lags[k]]
                        for k in np.arange(len(ar_lags))
                    ]
                )
                + innovs_emus_gv[i, t]
            )
    # remove buffer
    emus_gv = emus_gv[:, buffer:]

    return emus_gv
