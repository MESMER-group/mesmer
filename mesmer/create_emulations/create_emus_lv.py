# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to create local variability emulations with MESMER.
"""


import os

import joblib
import numpy as np


def create_emus_lv(params_lv, preds_lv, cfg, save_emus=True, submethod=""):
    """Create local variablity emulations.

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
    preds_lv : dict
        nested dictionary of predictors for local variability with keys

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific scenario)
    cfg : module
        config file containing metadata
    save_emus : bool, optional
        determines if emulation is saved or not, default = True
    submethod : str, optional
        determines if only submethod should be used, default = "" indicating using the
        full method

    Returns
    -------
    emus_lv : dict
        local variability emulations dictionary with keys

        - [scen] (3d array  (emu, time, gp) of local variability emulation time series)

    Notes
    -----
    - Assumptions:
        - if stochastic realizations are drawn, preds_lv must contain concatenated
          hist + future scenarios or only future scenarios
        - if no preds_lv needed, pass time as predictor instead such that can get info
          about how many scenarios / ts per scenario should be drawn for stochastic part
        - submethod specific assumptions are listed in the submethod description
    - Long-term TODO:
        - improve consistency with actual esms by sharing same realizations in
          historical time period and diff one for each scen in future
        - improve this function in terms of generalization properties + design
          (+ consistency with rest of code)
        - improve function to also work if only part of predictors are used for OLS and
          others for other methods
    """

    # specify necessary variables from config file
    if save_emus:
        dir_mesmer_emus = cfg.dir_mesmer_emus

    pred_names = list(preds_lv.keys())
    scens_out = list(preds_lv[pred_names[0]].keys())

    # if no submethod is specified, the actual method contained in params_lv is employed
    if len(submethod) == 0:
        submethod = params_lv["method"]

    # carry out emulations
    emus_lv = {}

    if "OLS" in submethod and params_lv["method_lt_each_gp_sep"]:
        emus_lv = create_emus_lv_OLS(params_lv, preds_lv)
        # overwrites empty emus_lv dict, i.e., must always be executed as first method

    # HERE ALTERNATIVE METHODS COULD BE PLACED WHICH ARE EXECUTED BEFORE AR1_sci
    # e.g., my idea to do hybrid between Link et al and my things

    if "AR1_sci" in submethod:
        emus_lv = create_emus_lv_AR1_sci(emus_lv, params_lv, preds_lv, cfg)

    # save the local trends emulation if requested
    if save_emus:
        dir_mesmer_emus_lv = dir_mesmer_emus + "local/local_variability/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_lv):
            os.makedirs(dir_mesmer_emus_lv)
            print("created dir:", dir_mesmer_emus_lv)
        filename_parts = [
            "emus_lv",
            submethod,
            *params_lv["preds"],
            *params_lv["targs"],
            params_lv["esm"],
            *scens_out,
        ]
        filename_emus_lv = dir_mesmer_emus_lv + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_lv, filename_emus_lv)

    return emus_lv


def create_emus_lv_AR1_sci(emus_lv, params_lv, preds_lv, cfg):
    """
    Create local variablity emulations with AR(1) process with spatially-correlated
    innovations.

    Parameters
    ----------
    emus_lv : dict
        local variability emulations dictionary with keys

        - [scen] (3d array (emu, time, gp) of local variability from previous submethods)
        - empty dict if no previous submethod
    params_lv : dict
        dictionary with the trained local variability parameters

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - [xx] (additional keys depend on employed method)
    preds_lv : dict
        nested dictionary of predictors for local variability with keys

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific scenario)
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
        - do for each target variable independently.
        - the variability is Gaussian
    - Long-term TODO:
        - add possibility to account for cross-correlation between different variables

    """

    print("Start with AR(1) with spatially correlated innovations.")
    pred_names = list(preds_lv.keys())
    scens_out = list(preds_lv[pred_names[0]].keys())
    nr_emus_v = cfg.nr_emus_v
    seed_all_scens = cfg.seed[params_lv["esm"]]

    for scen in scens_out:
        # if 1-d array, time = 1st dim, else time = 2nd dim
        if len(preds_lv[pred_names[0]][scen].shape) > 1:
            nr_ts_emus_stoch_v = preds_lv[pred_names[0]][scen].shape[1]
        else:
            nr_ts_emus_stoch_v = preds_lv[pred_names[0]][scen].shape[0]

        if scen not in emus_lv:
            emus_lv[scen] = {}

        for targ in params_lv["targs"]:

            seed = seed_all_scens[scen]["lv"]
            nr_gps = len(params_lv["AR1_int"][targ])

            # in case no emus_lv[scen] exist yet, initialize it. Otherwise build up on existing one
            if len(emus_lv[scen]) == 0:

                emus_lv[scen][targ] = np.zeros(nr_emus_v, nr_ts_emus_stoch_v, nr_gps)

            # ensure reproducibility
            np.random.seed(seed)

            # buffer so that initial start at 0 does not influence overall result
            buffer = 20

            print("Draw the innovations")
            # draw the innovations
            innovs = np.random.multivariate_normal(
                np.zeros(nr_gps),
                params_lv["loc_ecov_AR1_innovs"][targ],
                size=[nr_emus_v, nr_ts_emus_stoch_v + buffer],
            )

            print(
                "Compute the contribution to emus_lv by the AR(1) process with the spatially correlated innovations"
            )
            emus_lv_tmp = np.zeros([nr_emus_v, nr_ts_emus_stoch_v + buffer, nr_gps])
            for t in np.arange(1, nr_ts_emus_stoch_v + buffer):
                emus_lv_tmp[:, t, :] = (
                    params_lv["AR1_int"][targ]
                    + params_lv["AR1_coef"][targ] * emus_lv_tmp[:, t - 1, :]
                    + innovs[:, t, :]
                )
            emus_lv_tmp = emus_lv_tmp[:, buffer:, :]

            print("Create the full local variability emulations")
            emus_lv[scen][targ] += emus_lv_tmp

    return emus_lv


def create_emus_lv_OLS(params_lv, preds_lv):
    """Create local variablity emulations with OLS.

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
    preds_lv : dict
        nested dictionary of predictors for local variability with keys

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific scenario)

    Returns
    -------
    emus_lv : dict
        local variability emulations dictionary with keys

        - [scen] (3d array  (emu, time, gp) of local variability in response to global
          variability emulation time series)

    Notes
    -----
    - Assumptions:
        - first submethod that gets executed (i.e., assumes can make a new emus_lv dict
          within this function)
        - all predictors in preds_lv are being used (ie no other part of method is
          allowed to have predictors)
        - OLS coefs are the same for each scenario
    """

    print("Start with OLS")
    pred_names = list(preds_lv.keys())
    if pred_names != params_lv["preds"]:
        raise ValueError("Wrong list of predictors was passed.")

    scens_OLS = list(preds_lv[pred_names[0]].keys())
    emus_lv = {}
    for scen in scens_OLS:
        emus_lv[scen] = {}

        for targ in params_lv["targs"]:
            nr_emus_v, nr_ts_emus_v = preds_lv[pred_names[0]][scen].shape
            nr_gps = len(params_lv["coef_" + params_lv["preds"][0]][targ])
            emus_lv[scen][targ] = np.zeros([nr_emus_v, nr_ts_emus_v, nr_gps])
            for run in np.arange(nr_emus_v):
                for gp in np.arange(nr_gps):
                    emus_lv[scen][targ][run, :, gp] = sum(
                        [
                            params_lv["coef_" + pred][targ][gp]
                            * preds_lv[pred][scen][run]
                            for pred in params_lv["preds"]
                        ]
                    )
    return emus_lv
