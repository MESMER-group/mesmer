# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train global trend module of MESMER.
"""


import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

from mesmer.io import load_strat_aod


def train_gt(var, targ, esm, time, cfg, save_params=True):
    """
    Derive global trend (emissions + volcanoes) parameters from specified ensemble type
    with specified method.

    Parameters
    ----------
    var : dict
        nested global mean variable dictionary with keys for each scenario employed for
        training

        - [scen] (2d array (run, time) of globally-averaged variable time series)
    targ : str
        target variable (e.g., "tas")
    esm : str
        associated Earth System Model (e.g., "CanESM2" or "CanESM5")
    time : np.ndarray
        [scen] (1d array of years)
    cfg : module
        config file containing metadata
    save_params : bool, default True
        determines if parameters are saved or not, default = True

    Returns
    -------
    params_gt : dict
        dictionary containing the trained parameters for the chosen method / ensemble
        type

        - ["targ"] (emulated variable, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] (additional params depend on method employed)

    Notes
    -----
    - Assumptions:
        - All scens start at the same point in time
        - If historical data is present, historical data and future scenarios are
          transmitted as single time series
    - No perfect smoothness enforced at transition from historical to future scenario
    - No perfect overlap between future scenarios which share the same forcing in the
      beginning is enforced

    """

    # specify necessary variables from config file
    gen = cfg.gen
    method_gt = cfg.methods[targ]["gt"]
    preds_gt = cfg.preds[targ]["gt"]

    scenarios_tr = list(var.keys())

    # initialize parameters dictionary and fill in the metadata which does not depend on the applied method
    params_gt = {}
    params_gt["targ"] = targ
    params_gt["esm"] = esm
    params_gt["method"] = method_gt
    params_gt["preds"] = preds_gt
    params_gt["scenarios"] = scenarios_tr  # single entry in case of ic ensemble

    # apply the chosen method to the type of ensenble
    gt = {}
    if "LOWESS" in params_gt["method"]:
        for scen in scenarios_tr:  # ie derive gt for each scen individually
            gt[scen], frac_lowess_name = train_gt_ic_LOWESS(var[scen])
        params_gt["frac_lowess"] = frac_lowess_name
    else:
        raise ValueError("No alternative method to LOWESS is implemented for now.")

    params_gt["time"] = {}

    if scenarios_tr[0][:2] == "h-":  # ie if hist included

        if gen == 5:
            start_year_fut = 2005
        elif gen == 6:
            start_year_fut = 2014

        idx_start_year_fut = np.where(time[scen] == start_year_fut)[0][0] + 1

        params_gt["time"]["hist"] = time[scen][:idx_start_year_fut]

        # compute median LOWESS estimate of historical part across all scenarios
        gt_lowess_hist_all = np.zeros([len(gt.keys()), len(params_gt["time"]["hist"])])
        for i, scen in enumerate(gt.keys()):
            gt_lowess_hist_all[i] = gt[scen][:idx_start_year_fut]
        gt_lowess_hist = np.median(gt_lowess_hist_all, axis=0)

        if params_gt["method"] == "LOWESS_OLSVOLC":
            scen = scenarios_tr[0]
            var_all = var[scen][:, :idx_start_year_fut]
            for scen in scenarios_tr[1:]:
                var_tmp = var[scen][:, :idx_start_year_fut]
                var_all = np.vstack([var_all, var_tmp])

                # check for duplicates & exclude those runs
                var_all = np.unique(var_all, axis=0)

            params_gt["saod"], params_gt["hist"] = train_gt_ic_OLSVOLC(
                var_all, gt_lowess_hist, params_gt["time"]["hist"], cfg
            )
        elif params_gt["method"] == "LOWESS":
            params_gt["hist"] = gt_lowess_hist

        scenarios_tr_f = list(
            map(lambda x: x.replace("h-", ""), scenarios_tr)
        )  # isolte future scen names

    else:
        idx_start_year_fut = 0  # because first year would be already in future
        scenarios_tr_f = scenarios_tr  # because only future covered anyways

    for scen_f, scen in zip(scenarios_tr_f, scenarios_tr):
        params_gt["time"][scen_f] = time[scen][idx_start_year_fut:]
        params_gt[scen_f] = gt[scen][idx_start_year_fut:]

    # save the global trend paramters if requested
    if save_params:
        dir_mesmer_params = cfg.dir_mesmer_params
        dir_mesmer_params_gt = dir_mesmer_params + "global/global_trend/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_gt):
            os.makedirs(dir_mesmer_params_gt)
            print("created dir:", dir_mesmer_params_gt)
        filename_parts = [
            "params_gt",
            method_gt,
            *preds_gt,
            targ,
            esm,
            *scenarios_tr,
        ]
        filename_params_gt = dir_mesmer_params_gt + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_gt, filename_params_gt)

    return params_gt


def train_gt_ic_LOWESS(var):
    """
    Derive smooth global trend of variable from single ESM ic ensemble with LOWESS
    smoother.

    Parameters
    ----------
    var : np.ndarray
        2d array (run, time) of globally-averaged time series

    Returns
    -------
    gt_lowess : np.ndarray
        1d array of smooth global trend of variable
    frac_lowess : float
        fraction of the data used when estimating each y-value
    """

    # number time steps
    nr_ts = var.shape[1]

    # average across all runs to get a first smoothing
    av_var = np.mean(var, axis=0)

    # apply lowess smoother to further smooth the Tglob time series
    frac_lowess = (
        50 / nr_ts
    )  # rather arbitrarily chosen value that gives a smooth enough trend,
    # open to changes but if much smaller, var trend ends up very wiggly
    frac_lowess_name = "50/nr_ts"

    gt_lowess = lowess(
        av_var, np.arange(nr_ts), return_sorted=False, frac=frac_lowess, it=0
    )

    return gt_lowess, frac_lowess_name


def train_gt_ic_OLSVOLC(var, gt_lowess, time, cfg):
    """
    Derive global trend (emissions + volcanoes) parameters from single ESM ic ensemble
    by adding volcanic spikes to LOWESS trend.

    Parameters
    ----------
    var : np.ndarray
        2d array (run, time) of globally-averaged time series
    gt_lowess : np.ndarray
        1d array of smooth global trend of variable
    time : np.ndarray
        1d array of years
    cfg : module
        config file containing metadata needed to load in stratospheric AOD time series

    Returns
    -------
    coef_saod : float
        stratospheric AOD OLS coefficient for variable variability
    gt : np.ndarray
        1d array of global temperature trend with volcanic spikes

    Notes
    -----
    - Assumptions:
        - only historical time period data is passed

    """

    # specify necessary variables from cfg file
    dir_obs = cfg.dir_obs

    nr_runs, nr_ts = var.shape

    # account for volcanic eruptions in historical time period
    # load in observed stratospheric aerosol optical depth
    aod_obs = load_strat_aod(time, dir_obs).reshape(
        -1, 1
    )  # bring in correct format for sklearn linear regression
    aod_obs_all = np.tile(
        aod_obs, (nr_runs, 1)
    )  # repeat aod time series as many times as runs available
    nr_aod_obs = len(aod_obs)

    if nr_ts != nr_aod_obs:
        raise ValueError(
            f"The number of time steps of the variable ({nr_ts}) and the saod ({nr_aod_obs}) do not match."
        )
    # extract global variability (which still includes volc eruptions) by removing smooth trend from Tglob in historic period
    gv_all_for_aod = np.zeros(nr_runs * nr_aod_obs)
    i = 0
    for run in np.arange(nr_runs):
        gv_all_for_aod[i : i + nr_aod_obs] = var[run] - gt_lowess
        i += nr_aod_obs
    # fit linear regression of gv to aod (because some ESMs react very strongly to volcanoes)
    linreg_gv_volc = LinearRegression(fit_intercept=False).fit(
        aod_obs_all, gv_all_for_aod
    )  # no intercept to not artifically
    # move the ts

    # extract the saod coefficient
    coef_saod = linreg_gv_volc.coef_[0]

    # apply linear regression model to obtain volcanic spikes
    contrib_volc = linreg_gv_volc.predict(aod_obs)

    # merge the lowess trend wit the volc contribution
    gt = gt_lowess + contrib_volc

    return coef_saod, gt
