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
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


def train_gv(gv, targ, esm, cfg, save_params=True, **kwargs):
    """
    Derive global variability parameters for a specified method.

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
    **kwargs:
        additional arguments, passed through to the training function

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
        - if historical data is used for training, it has its own scenario

    - TODO:
        - add ability to weight samples differently than equal weight for each scenario

    """

    # specify necessary variables from config file
    method_gv = cfg.methods[targ]["gv"]
    preds_gv = cfg.preds[targ]["gv"]
    wgt_scen_tr_eq = cfg.wgt_scen_tr_eq

    scenarios_tr = list(gv.keys())

    # initialize parameters dictionary and fill in the metadata which does not depend on
    # the applied method
    params_gv = {}
    params_gv["targ"] = targ
    params_gv["esm"] = esm
    params_gv["method"] = method_gv
    params_gv["preds"] = preds_gv
    params_gv["scenarios"] = scenarios_tr

    # apply the chosen method
    if params_gv["method"] == "AR" and wgt_scen_tr_eq:
        # specifiy parameters employed for AR process fitting
        if "max_lag" not in kwargs:
            kwargs["max_lag"] = 12
        if "sel_crit" not in kwargs:
            kwargs["sel_crit"] = "bic"
        params_gv = train_gv_AR(params_gv, gv, kwargs["max_lag"], kwargs["sel_crit"])
    else:
        raise ValueError(
            "The chosen method and / or weighting approach is currently not implemented."
        )

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


def train_gv_AR(params_gv, gv, max_lag, sel_crit):
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
    max_lag: int
        maximum number of lags considered during fitting
    sel_crit: str
        selection criterion for the AR process order, e.g., 'bic' or 'aic'

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
        - ["AR_order_sel"] (selected AR order, int)
        - ["AR_std_innovs"] (standard deviation of the innovations of the selected AR
          model, float)

    Notes
    -----
    - Assumptions
        - number of runs per scenario and the number of time steps in each scenario can
        vary
        - each scenario receives equal weight during training

    """

    params_gv["max_lag"] = max_lag
    params_gv["sel_crit"] = sel_crit

    # select the AR Order
    nr_scens = len(gv.keys())
    AR_order_scens_tmp = np.zeros(nr_scens)

    for scen_idx, scen in enumerate(gv.keys()):
        nr_runs = gv[scen].shape[0]
        AR_order_runs_tmp = np.zeros(nr_runs)

        for run in np.arange(nr_runs):
            run_ar_lags = ar_select_order(
                gv[scen][run], maxlag=max_lag, ic=sel_crit, old_names=False
            ).ar_lags
            # if order > 0 is selected,add selected order to vector
            if len(run_ar_lags) > 0:
                AR_order_runs_tmp[run] = run_ar_lags[-1]

        AR_order_scens_tmp[scen_idx] = np.percentile(
            AR_order_runs_tmp, q=50, interpolation="nearest"
        )
        # interpolation is not a good way to go here because it could lead to an AR
        # order that wasn't chosen by run -> avoid it by just taking nearest

    AR_order_sel = int(np.percentile(AR_order_scens_tmp, q=50, interpolation="nearest"))

    # determine the AR params for the selected AR order
    params_gv["AR_int"] = 0
    params_gv["AR_coefs"] = np.zeros(AR_order_sel)
    params_gv["AR_order_sel"] = AR_order_sel
    params_gv["AR_std_innovs"] = 0

    for scen_idx, scen in enumerate(gv.keys()):
        nr_runs = gv[scen].shape[0]
        AR_order_runs_tmp = np.zeros(nr_runs)
        AR_int_tmp = 0
        AR_coefs_tmp = np.zeros(AR_order_sel)
        AR_std_innovs_tmp = 0

        for run in np.arange(nr_runs):
            AR_model_tmp = AutoReg(
                gv[scen][run], lags=AR_order_sel, old_names=False
            ).fit()
            AR_int_tmp += AR_model_tmp.params[0] / nr_runs
            AR_coefs_tmp += AR_model_tmp.params[1:] / nr_runs
            AR_std_innovs_tmp += np.sqrt(AR_model_tmp.sigma2) / nr_runs

        params_gv["AR_int"] += AR_int_tmp / nr_scens
        params_gv["AR_coefs"] += AR_coefs_tmp / nr_scens
        params_gv["AR_std_innovs"] += AR_std_innovs_tmp / nr_scens

    # check if fitted AR process is stationary
    # (highly unlikely this test will ever fail but better safe than sorry)
    ar = np.r_[1, -params_gv["AR_coefs"]]  # add zero-lag and negate
    ma = np.r_[1]  # add zero-lag
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    if not arma_process.isstationary:
        raise ValueError(
            "The fitted AR process is not stationary. Another solution is needed."
        )

    return params_gv
