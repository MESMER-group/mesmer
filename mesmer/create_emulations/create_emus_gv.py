# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to create global variability emulations with MESMER.
"""


import os

import joblib
import numpy as np


def create_emus_gv(params_gv, preds_gv, cfg, save_emus=True):
    """Create global variablity emulations for specified method.

    Parameters
    ----------
    params_gv : dict
        Parameters dictionary.

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - [xx] (additional keys depend on employed method and are listed in
          train_gv_T_method() function)
    preds_gv : dict
        nested dictionary of predictors for global variability with keys

        - [pred][scen]  (1d/2d arrays (time)/(run, time) of predictor for specific
        scenario)
    cfg : module
        config file containing metadata
    save_emus : bool, optional
        determines if emulation is saved or not, default = True

    Returns
    -------
    emus_gv : dict
        global variability emulations dictionary with keys

        - [scen] (2d array  (emus, time) of global trend emulation time series)

    Notes
    -----
    - Assumptions:
        - if no preds_gv needed, pass time as predictor instead such that can get info
          about how many scenarios / ts per scenario should be drawn for stochastic part

    """

    # specify necessary variables from config file
    if save_emus:
        dir_mesmer_emus = cfg.dir_mesmer_emus

    nr_emus_v = cfg.nr_emus_v
    seed_all_scens = cfg.seed[params_gv["esm"]]

    pred_names = list(preds_gv.keys())
    scens_out = list(preds_gv[pred_names[0]].keys())

    if scens_out != list(seed_all_scens.keys()):
        raise ValueError(
            "The scenarios which should be emulated do not have a seed assigned in the"
            " config file. The emulations cannot be created."
        )

    # set up dict for emulations of global variability with emulated scenarios as keys
    emus_gv = {}

    for scen in scens_out:
        if len(preds_gv[pred_names[0]][scen].shape) > 1:
            nr_ts_emus_v = preds_gv[pred_names[0]][scen].shape[1]
        else:
            nr_ts_emus_v = preds_gv[pred_names[0]][scen].shape[0]

        # apply the chosen method
        if params_gv["method"] == "AR":
            emus_gv[scen] = create_emus_gv_AR(
                params_gv, nr_emus_v, nr_ts_emus_v, seed_all_scens[scen]["gv"]
            )
        else:
            raise ValueError("The chosen method is currently not implemented.")
            # if the emus should depend on the scen, scen needs to be passed to the fct

    # save the global variability emus if requested
    if save_emus:
        dir_mesmer_emus_gv = dir_mesmer_emus + "global/global_variability/"
        # check if folder to save emus in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_gv):
            os.makedirs(dir_mesmer_emus_gv)
            print("created dir:", dir_mesmer_emus_gv)
        filename_parts = [
            "emus_gv",
            params_gv["method"],
            *params_gv["preds"],
            params_gv["targ"],
            params_gv["esm"],
            *scens_out,
        ]
        filename_emus_gv = dir_mesmer_emus_gv + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_gv, filename_emus_gv)

    return emus_gv


def create_emus_gv_AR(params_gv, nr_emus_v, nr_ts_emus_v, seed):
    """Draw global variablity emulations from an AR process.

    Parameters
    ----------
    params_gv : dict
        Parameters dictionary.

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - ["max_lag"] (maximum lag considered when finding suitable AR model, int)
        - ["sel_crit"] (selection criterion applied to find suitable AR model, str)
        - ["AR_int"] (intercept of the AR model, float)
        - ["AR_coefs"] (coefficients of the AR model for the lags which are contained in
          the selected AR model, list of floats)
        - ["AR_order_sel"] (selected AR order, int)
        - ["AR_std_innovs"] (standard deviation of the innovations of the selected AR
          model, float)
    nr_emus_v : int
        number of global variability emulations
    nr_ts_emus_v : int
        number of time steps in each global variability emulation
    seed : int
        esm and scenario specific seed for gv module to ensure reproducability of
        results

    Returns
    -------
    emus_gv : dict
        global variability emulations dictionary with keys

        - [scen] (2d array  (emus, time) of global variability emulation time series)
    """

    # ensure reproducibility
    np.random.seed(seed)

    # buffer so that initial start at 0 does not influence overall result
    buffer = 50

    # re-name params for easier reading of code below
    ar_int = params_gv["AR_int"]
    ar_coefs = params_gv["AR_coefs"]
    ar_lags = np.arange(1, params_gv["AR_order_sel"] + 1, dtype=int)

    # if AR(0) process chosen, no AR_coefs are available -> to have code run
    # nevertheless ar_coefs and ar_lags are set to 0 (-> emus are created with
    # ar_int + innovs)
    if len(ar_coefs) == 0:
        ar_coefs = [0]
        ar_lags = [0]

    innovs_emus_gv = np.random.normal(
        loc=0, scale=params_gv["AR_std_innovs"], size=(nr_emus_v, nr_ts_emus_v + buffer)
    )

    # initialize global variability emulations (dim array: nr_emus x nr_ts)
    emus_gv = np.zeros([nr_emus_v, nr_ts_emus_v + buffer])

    for i in np.arange(nr_emus_v):
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
