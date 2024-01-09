# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to create local variability emulations with MESMER.
"""


import numpy as np
import xarray as xr

from mesmer.create_emulations.utils import _gather_lr_params, _gather_lr_preds
from mesmer.io.save_mesmer_bundle import save_mesmer_data
from mesmer.stats import LinearRegression, draw_auto_regression_correlated


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

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific
          scenario)

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
        save_mesmer_data(
            emus_lv,
            cfg.dir_mesmer_emus,
            "local",
            "local_variability",
            filename_parts=[
                "emus_lv",
                submethod,
                *params_lv["preds"],
                *params_lv["targs"],
                params_lv["esm"],
                *scens_out,
            ],
        )

    return emus_lv


def create_emus_lv_AR1_sci(emus_lv, params_lv, preds_lv, cfg):
    """
    Create local variablity emulations with AR(1) process with spatially-correlated
    innovations.

    Parameters
    ----------
    emus_lv : dict
        local variability emulations dictionary with keys

        - [scen] 3d array (emu, time, gp) of local variability from previous submethods
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

        - [pred][scen] 1d/ 2d arrays (time)/(run, time) of predictor for specific
          scenario

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
    pred_name = list(preds_lv.keys())[0]
    scens_out = list(preds_lv[pred_name].keys())
    nr_emus_v = cfg.nr_emus_v
    seed_all_scens = cfg.seed[params_lv["esm"]]

    for scen in scens_out:

        time_axis = 1 if preds_lv[pred_name][scen].ndim > 1 else 0

        nr_ts_emus_stoch_v = preds_lv[pred_name][scen].shape[time_axis]

        if scen not in emus_lv:
            emus_lv[scen] = {}

        for targ in params_lv["targs"]:

            seed = seed_all_scens[scen]["lv"]

            # in case no emus_lv[scen] exist yet, initialize it. Otherwise build up on
            # existing one
            if len(emus_lv[scen]) == 0:
                emus_lv[scen][targ] = 0

            # create intermediate ar_params & covariance Dataset / DataArray
            intercept = xr.DataArray(params_lv["AR1_int"][targ], dims="gridpoint")

            dims = ("lags", "gridpoint")
            coeffs = xr.DataArray(params_lv["AR1_coef"][targ][np.newaxis, :], dims=dims)

            ar_params = xr.Dataset({"intercept": intercept, "coeffs": coeffs})

            dims = ("gridpoint_i", "gridpoint_j")
            covariance = xr.DataArray(params_lv["loc_ecov_AR1_innovs"][targ], dims=dims)

            emus_ar = draw_auto_regression_correlated(
                ar_params,
                covariance,
                time=nr_ts_emus_stoch_v,
                realisation=nr_emus_v,
                seed=seed,
                buffer=20,
            )

            # get back the old order of the emus
            emus_ar = emus_ar.values.transpose((2, 0, 1))

            emus_lv[scen][targ] += emus_ar.squeeze()
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

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific
          scenario)

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

    pred_names = list(preds_lv.keys())
    if pred_names != params_lv["preds"]:
        raise ValueError("Wrong list of predictors was passed.")

    scens_OLS = list(preds_lv[pred_names[0]].keys())
    emus_lv = {}
    for scen in scens_OLS:
        emus_lv[scen] = {}

        preds = _gather_lr_preds(
            preds_lv, params_lv["preds"], scen, dims=("scen", "time")
        )

        for targ in params_lv["targs"]:

            params = _gather_lr_params(params_lv, targ, dims="gridpoint")

            lr = LinearRegression()
            lr.params = params
            prediction = lr.predict(predictors=preds)

            emus_lv[scen][targ] = prediction.values

    return emus_lv
