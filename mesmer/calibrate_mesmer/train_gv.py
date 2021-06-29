# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train global variability module of MESMER.
"""


import os

import joblib
import numpy as np
from statsmodels.tsa.ar_model import ar_select_order


def train_gv(gv, targ, esm, cfg, save_params=True):
    """
    Derive global variability parameters for a specified method from a specifie
    ensemble type.

    Parameters
    ----------
    gv : dict
        Nested global mean variability dictionary with keys

        - [scen] (2d array (run, time) of globally-averaged variability time series)
    targ : str
        target variable (e.g., "tas")
    esm : str
        associated Earth System Model (e.g., "CanESM2" or "CanESM5")
    cfg : config module
        config file containing metadata
    save_params : bool, optional
        determines if parameters are saved or not, default = True

    Returns
    -------
    params_gv : dict
        dictionary containing the trained parameters for the chosen method / ensemble
        type

        - ["targ"] (emulated variable, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] additional params depend on method employed, specified in
          ``train_gv_T_method()`` function

    Notes
    -----
    - Assumption

        - If historical data is used for training, it has its own scenario.

    """

    # specify necessary variables from config file
    method_gv = cfg.methods[targ]["gv"]
    preds_gv = cfg.preds[targ]["gv"]

    scenarios_tr = list(gv.keys())

    # initialize parameters dictionary and fill in the metadata which does not depend on the applied method
    params_gv = {}
    params_gv["targ"] = targ
    params_gv["esm"] = esm
    params_gv["method"] = method_gv
    params_gv["preds"] = preds_gv
    params_gv["scenarios"] = scenarios_tr

    # apply the chosen method
    if params_gv["method"] == "AR":
        params_gv = train_gv_AR(params_gv, gv)
    else:
        raise ValueError("The chosen method is currently not implemented.")

    # save the global variability paramters if requested
    if save_params:
        dir_mesmer_params = cfg.dir_mesmer_params
        dir_mesmer_params_gv = dir_mesmer_params + "global/global_variability/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_gv):
            os.makedirs(dir_mesmer_params_gv)
            print("created dir:", dir_mesmer_params_gv)
        filename_parts = [
            "params_gv",
            method_gv,
            *preds_gv,
            targ,
            esm,
            *scenarios_tr,
        ]
        filename_params_gv = dir_mesmer_params_gv + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_gv, filename_params_gv)

    return params_gv


def train_gv_AR(params_gv, gv):
    """
    Derive AR parameters of global variability under the assumption that gv does not
    depend on the scenario.

    Parameters
    ----------
    params_gv : dict
        parameter dictionary containing keys which do not depend on applied method

        - ["targ"] (variable, i.e., tas or tblend, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, i.e., AR, str)
        - ["scenarios"] (emission scenarios used for training, list of strs)
    gv : dict
        nested global mean temperature variability (volcanic influence removed)
        dictionary with keys

        - [scen] (2d array (nr_runs, nr_ts) of globally-averaged temperature variability
          time series)

    Returns
    -------
    params : dict
        parameter dictionary containing original keys plus

        - ["max_lag"] (maximum lag considered when finding suitable AR model, hardcoded
          to 15 here, int)
        - ["sel_crit"] (selection criterion applied to find suitable AR model, hardcoded
          to Bayesian Information Criterion bic here, str)
        - ["AR_int"] (intercept of the AR model, float)
        - ["AR_coefs"] (coefficients of the AR model for the lags which are contained in
          the selected AR model, list of floats)
        - ["AR_lags"] (AR lags which are contained in the selected AR model, list of
          ints)
        - ["AR_std_innovs"] (standard deviation of the innovations of the selected AR
          model, float)

    Notes
    -----
    - TODO
        - change fct to 1) train AR params on each run individually -> 2) average across
          all runs of specific scen -> 3) all scens
        - learn proper way for bic selection -> eg same process as described above but
          2x: 1x to select order (always take median) 1x to fit params (take mean or
          median again?)

    """

    # put all the global variability time series together in single array
    scenarios = params_gv["scenarios"]  # single entry if ic ensemble, otherwise more

    # assumption: nr_runs per scen and nr_ts for these runs can vary
    nr_samples = 0
    for scen in scenarios:
        nr_runs, nr_ts = gv[scen].shape
        nr_samples += nr_runs * nr_ts

    gv_all = np.zeros(nr_samples)
    i = 0
    for scen in scenarios:
        k = (
            gv[scen].shape[0] * gv[scen].shape[1]
        )  # = nr_runs*nr_ts for this specific scenario
        gv_all[i : i + k] = gv[scen].flatten()
        i += k

    # specifiy parameters employed for AR process fitting
    max_lag = 12  # rather arbitrarily chosen in trade off between allowing enough lags and computational time
    # open to change
    sel_crit = "bic"  # selection criterion

    # select AR order and fit the AR model
    AR_order_sel = ar_select_order(gv_all, maxlag=max_lag, ic=sel_crit)
    AR_model = AR_order_sel.model.fit()

    # fill in the parameter dictionary
    params_gv["max_lag"] = max_lag
    params_gv["sel_crit"] = sel_crit
    params_gv["AR_int"] = AR_model.params[0]  # intercept
    params_gv["AR_coefs"] = AR_model.params[
        1:
    ]  # all coefs which are linked to lags, sorted in same way as ar lags
    params_gv["AR_lags"] = AR_model.ar_lags
    params_gv["AR_std_innovs"] = np.sqrt(
        AR_model.sigma2
    )  # sqrt of variance = standard deviation

    return params_gv
