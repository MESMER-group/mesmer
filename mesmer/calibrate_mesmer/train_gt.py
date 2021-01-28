"""
mesmer.calibrate_mesmer.train_gt
===================
Functions to train global trend module of MESMER.


Functions:
    train_gt()
    train_gt_ic_LOWESS()
    train_gt_ic_OLSVOLC()

"""


import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

from mesmer.io import load_strat_aod


def train_gt(var, targ, esm, time, cfg, save_params=True):
    """ Derive global trend (emissions + volcanoes) parameters from specified ensemble type with specified method.

    Args:
    - var (dict): nested global mean variable dictionary with keys for each scenario employed for training
        [scen] (2d array (run,time) of globally-averaged variable time series)
    - targ (str): target variable (e.g., 'tas')
    - esm (str): associated Earth System Model (e.g., 'CanESM2' or 'CanESM5')
    - time (np.ndarray): 
        [scen] (1d array of years)
    - cfg (module): config file containnig metadata
    - save_params (bool, optional): determines if parameters are saved or not, default = True

    Returns:
    - params_gt (dict): dictionary containing the trained parameters for the chosen method / ensemble type
        ['targ'] (emulated variable, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (emission scenarios used for training, list of strs)
        [xx] (additional params depend on method employed)

    """

    # specify necessary variables from config file
    ens_type_tr = cfg.ens_type_tr
    hist_tr = cfg.hist_tr
    method_gt = cfg.methods[targ]["gt"]
    preds_gt = cfg.preds[targ]["gt"]
    dir_mesmer_params = cfg.dir_mesmer_params

    scenarios_tr = list(var.keys())
    if hist_tr:  # check whether historical data is used in training
        scen_name_tr = "hist_" + "_".join(scenarios_tr)
    else:
        scen_name_tr = "_".join(scenarios_tr)

    # initialize parameters dictionary and fill in the metadata which does not depend on the applied method
    params_gt = {}
    params_gt["targ"] = targ
    params_gt["esm"] = esm
    params_gt["ens_type"] = ens_type_tr
    params_gt["method"] = method_gt
    params_gt["preds"] = preds_gt
    params_gt["scenarios"] = scenarios_tr  # single entry in case of ic ensemble

    for scen in params_gt["scenarios"]:
        params_gt[scen] = {}
        params_gt[scen]["time"] = time[scen]

    # apply the chosen method to the type of ensenble
    if "LOWESS" in params_gt["method"]:
        for scen in params_gt["scenarios"]:  # ie derive gt for each scen individually
            params_gt[scen]["gt"], params_gt[scen]["frac_lowess"] = train_gt_ic_LOWESS(
                var[scen]
            )

    if params_gt["method"] == "LOWESS_OLSVOLC":
        for scen in params_gt["scenarios"]:  # ie derive gt for each scen individually
            if (
                time[scen].min() < 2000
            ):  # check if historical period is part of the training runs
                # overwrites existing smooth trend with smooth trend + volcanic spikes
                params_gt[scen]["saod"], params_gt[scen]["gt"] = train_gt_ic_OLSVOLC(
                    var[scen], params_gt[scen]["gt"], time[scen], cfg
                )
            else:  # no volcanic eruptions in the scenario time period
                params_gt[scen]["saod"] = None
                params_gt[scen]["gt"] = gt_lowess[scen]

    # save the global trend paramters if requested
    if save_params:
        dir_mesmer_params_gt = dir_mesmer_params + "global/global_trend/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_gt):
            os.makedirs(dir_mesmer_params_gt)
            print("created dir:", dir_mesmer_params_gt)
        joblib.dump(
            params_gt,
            dir_mesmer_params_gt
            + "params_gt_"
            + ens_type_tr
            + "_"
            + method_gt
            + "_"
            + "_".join(preds_gt)
            + "_"
            + targ
            + "_"
            + esm
            + "_"
            + scen_name_tr
            + ".pkl",
        )

    return params_gt


def train_gt_ic_LOWESS(var):
    """ Derive smooth global trend of variable from single ESM ic ensemble with LOWESS smoother.

    Args:
    - var (np.ndarray): 2d array (run, time) of globally-averaged time series

    Returns:
    - gt_lowess (np.ndarray): 1d array of smooth global trend of variable
    - frac_lowess (float): fraction of the data used when estimating each y-value

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

    gt_lowess = lowess(
        av_var, np.arange(nr_ts), return_sorted=False, frac=frac_lowess, it=0
    )

    return gt_lowess, frac_lowess


def train_gt_ic_OLSVOLC(var, gt_lowess, time, cfg):
    """ Derive global trend (emissions + volcanoes) parameters from single ESM ic ensemble by adding volcanic spikes to LOWESS trend.

    Args:
    - var (np.ndarray): 2d array (run, time) of globally-averaged time series
    - gt_lowess (np.ndarray): 1d array of smooth global trend of variable
    - time (np.ndarray): 1d array of years
    - cfg (module): config file containnig metadata needed to load in stratospheric AOD time series

    Returns:
    - coef_saod (float): stratospheric AOD OLS coefficient for variable variability
    - gt (np.ndarray): 1d array of global temperature trend with volcanic spikes

    """

    # specify necessary variables from cfg file
    gen = cfg.gen
    dir_obs = cfg.dir_obs

    nr_runs, nr_ts = var.shape

    # account for volcanic eruptions in historical time period
    # load in observed stratospheric aerosol optical depth
    aod_obs = load_strat_aod(time, gen, dir_obs).reshape(
        -1, 1
    )  # bring in correct format for sklearn linear regression
    aod_obs_all = np.tile(
        aod_obs, (nr_runs, 1)
    )  # repeat aod time series as many times as runs available
    nr_aod_obs = len(aod_obs)

    # extract global variability (which still includes volc eruptions) by removing smooth trend from Tglob in historic period
    gv_all_for_aod = np.zeros(nr_runs * nr_aod_obs)
    i = 0
    for run in np.arange(nr_runs):
        gv_all_for_aod[i : i + nr_aod_obs] = (var[run] - gt_lowess)[:nr_aod_obs]
        i += nr_aod_obs
    # fit linear regression of gv to aod (because some ESMs react very strongly to volcanoes)
    linreg_gv_volc = LinearRegression(fit_intercept=False).fit(
        aod_obs_all, gv_all_for_aod
    )  # no intercept to not artifically
    # move the ts
    # apply linear regression model to obtain volcanic spikes
    contrib_volc = np.concatenate(
        (linreg_gv_volc.predict(aod_obs), np.zeros(nr_ts - nr_aod_obs))
    )  # aod contribution in past, 0 in future
    coef_saod = linreg_gv_volc.coef_[0]
    gt = contrib_volc + gt_lowess

    return coef_saod, gt
