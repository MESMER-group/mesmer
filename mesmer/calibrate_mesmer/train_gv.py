"""
mesmer.calibrate_mesmer.train_gv
===================
Functions to train global variability module of MESMER.


Functions:
    train_gv_T()
    train_gv_T_AR()

"""


import os

import joblib
import numpy as np
from statsmodels.tsa.ar_model import ar_select_order


def train_gv_T(gv_novolc_T, targ, esm, cfg, save_params=True):
    """Derive global variability parameters for a specified method from a specified ensemble type.

    Args:
    - gv_novolc_T(dict): nested global mean temperature variability (volcanic influence removed) dictionary with keys
        [scen] (2d array (run,time) of globally-averaged temperature variability time series)
    - targ (str): target variable (e.g., 'tas')
    - esm (str): associated Earth System Model (e.g., 'CanESM2' or 'CanESM5')
    - cfg (module): config file containnig metadata
    - save_params (bool, optional): determines if parameters are saved or not, default = True

    Returns:
    - params_gv_T (dict): dictionary containing the trained parameters for the chosen method / ensemble type
        ['targ'] (emulated variable, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (emission scenarios used for training, list of strs)
        [xx] additional params depend on method employed, specified in train_gv_T_ens_type_method() function

    """

    # specify necessary variables from config file
    ens_type_tr = cfg.ens_type_tr
    hist_tr = cfg.hist_tr
    method_gv = cfg.methods[targ]["gv"]
    preds_gv = cfg.preds[targ]["gv"]
    scenarios_tr = cfg.scenarios_tr
    scen_name_tr = cfg.scen_name_tr
    dir_mesmer_params = cfg.dir_mesmer_params

    scenarios_tr = list(gv_novolc_T.keys())
    if hist_tr:  # check whether historical data is used during training too
        scen_name_tr = "hist_" + "_".join(scenarios_tr)
    else:
        scen_name_tr = "_".join(scenarios_tr)

    # initialize parameters dictionary and fill in the metadata which does not depend on the applied method
    params_gv_T = {}
    params_gv_T["targ"] = targ
    params_gv_T["esm"] = esm
    params_gv_T["ens_type"] = ens_type_tr
    params_gv_T["method"] = method_gv
    params_gv_T["preds"] = preds_gv
    params_gv_T["scenarios"] = scenarios_tr

    # apply the chosen method
    if (
        params_gv_T["method"] == "AR"
    ):  # for now irrespective of ens_type. Could still be adapted later if necessary
        params_gv_T = train_gv_T_AR(params_gv_T, gv_novolc_T)
    else:
        print("No alternative method is currently implemented")

    # save the global variability paramters if requested
    if save_params:
        dir_mesmer_params_gv = dir_mesmer_params + "global/global_variability/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_gv):
            os.makedirs(dir_mesmer_params_gv)
            print("created dir:", dir_mesmer_params_gv)
        joblib.dump(
            params_gv_T,
            dir_mesmer_params_gv
            + "params_gv_"
            + ens_type_tr
            + "_"
            + method_gv
            + "_"
            + "_".join(preds_gv)
            + "_"
            + targ
            + "_"
            + esm
            + "_"
            + scen_name_tr
            + ".pkl",
        )

    return params_gv_T


def train_gv_T_AR(params_gv_T, gv_novolc_T):
    """Derive AR parameters of global variability under the assumption that gv does not depend on the scenario.

    Args:
    - params_gv_T (dict): parameter dictionary containing keys which do not depend on applied method
        ['targ'] (variable, i.e., tas or tblend, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, i.e., ic or ms, str)
        ['method'] (applied method, i.e., AR, str)
        ['scenarios'] (emission scenarios used for training, list of strs)
    - gv_novolc(dict): nested global mean temperature variability (volcanic influence removed) dictionary with keys
        [scen] (2d array (nr_runs, nr_ts) of globally-averaged temperature variability time series)

    Returns:
    - params_gv_T (dict): parameter dictionary containing original keys plus
        ['max_lag'] (maximum lag considered when finding suitable AR model, hardcoded to 15 here, int)
        ['sel_crit'] (selection criterion applied to find suitable AR model, hardcoded to Bayesian Information Criterion bic here, str)
        ['AR_int'] (intercept of the AR model, float)
        ['AR_coefs'] (coefficients of the AR model for the lags which are contained in the selected AR model, list of floats)
        ['AR_lags'] (AR lags which are contained in the selected AR model, list of ints)
        ['AR_std_innovs'] (standard deviation of the innovations of the selected AR model, float)

    """

    # put all the global variability time series together in single array
    scenarios = params_gv_T["scenarios"]  # single entry if ic ensemble, otherwise more

    # assumption: nr_runs per scen and nr_ts for these runs can vary
    nr_samples = 0
    for scen in scenarios:
        nr_runs, nr_ts = gv_novolc_T[scen].shape
        nr_samples += nr_runs * nr_ts

    gv_novolc_T_all = np.zeros(nr_samples)
    i = 0
    for scen in scenarios:
        k = (
            gv_novolc_T[scen].shape[0] * gv_novolc_T[scen].shape[1]
        )  # = nr_runs*nr_ts for this specific scenario
        gv_novolc_T_all[i : i + k] = gv_novolc_T[scen].flatten()
        i += k

    # specifiy parameters employed for AR process fitting
    max_lag = 12  # rather arbitrarily chosen in trade off between allowing enough lags and computational time
    # open to change
    sel_crit = "bic"  # selection criterion

    # select AR order and fit the AR model
    AR_order_sel = ar_select_order(
        gv_novolc_T_all, maxlag=max_lag, ic=sel_crit, glob=True
    )
    AR_model = AR_order_sel.model.fit()

    # fill in the parameter dictionary
    params_gv_T["max_lag"] = max_lag
    params_gv_T["sel_crit"] = sel_crit
    params_gv_T["AR_int"] = AR_model.params[0]  # intercept
    params_gv_T["AR_coefs"] = AR_model.params[
        1:
    ]  # all coefs which are linked to lags, sorted in same way as ar lags
    params_gv_T["AR_lags"] = AR_model.ar_lags
    params_gv_T["AR_std_innovs"] = np.sqrt(
        AR_model.sigma2
    )  # sqrt of variance = standard deviation

    return params_gv_T
