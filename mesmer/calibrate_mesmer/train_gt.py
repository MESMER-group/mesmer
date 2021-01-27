"""
mesmer.calibrate_mesmer.train_gt
===================
Functions to train global trend module of MESMER.


Functions:
    train_gt_T()
    train_gt_T_ic_LOWESS_OLS()

"""


import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

from mesmer.io import load_strat_aod


def train_gt_T(Tglob, targ, esm, time, cfg, save_params=True):
    """ Derive global trend (emissions + volcanoes) parameters from specified ensemble type with specified method.

    Args:
    - Tglob (dict): nested global mean temperature dictionary with keys for each scenario employed for training
        [scen] (2d array (run,time) of globally-averaged temperature anomaly time series)
    - targ (str): target variable (e.g., 'tas')
    - esm (str): associated Earth System Model (e.g., 'CanESM2' or 'CanESM5')
    - time (np.ndarray): 
        [scen] (1d array of years)
    - cfg (module): config file containnig metadata
    - save_params (bool, optional): determines if parameters are saved or not, default = True

    Returns:
    - params_gt_T (dict): dictionary containing the trained parameters for the chosen method / ensemble type
        ['targ'] (emulated variable, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
        ['preds'] (predictors, list of strs)
        ['scenarios'] (emission scenarios used for training, list of strs)
        ['time'] (1d array of years, np.ndarray)
        [xx] additional params depend on method employed, specified in train_gt_T_enstype_method() function

    General remarks:
    - Assumption: all training scenarios span the same time period and thus a single time vector suffices

    """

    # specify necessary variables from config file
    ens_type_tr = cfg.ens_type_tr
    hist_tr = cfg.hist_tr
    method_gt = cfg.methods[targ]["gt"]
    preds_gt = cfg.preds[targ]["gt"]
    dir_mesmer_params = cfg.dir_mesmer_params

    scenarios_tr = list(Tglob.keys())
    if hist_tr:  # check whether historical data is used in training
        scen_name_tr = "hist_" + "_".join(scenarios_tr)
    else:
        scen_name_tr = "_".join(scenarios_tr)

    # initialize parameters dictionary and fill in the metadata which does not depend on the applied method
    params_gt_T = {}
    params_gt_T["targ"] = targ
    params_gt_T["esm"] = esm
    params_gt_T["ens_type"] = ens_type_tr
    params_gt_T["method"] = method_gt
    params_gt_T["preds"] = preds_gt
    params_gt_T["scenarios"] = scenarios_tr  # single entry in case of ic ensemble

    # apply the chosen method to the type of ensenble
    if params_gt_T["method"] == "LOWESS_OLS":
        for scen in params_gt_T["scenarios"]:  # ie derive gt for each scen individually
            params_gt_T = train_gt_T_ic_LOWESS_OLS(params_gt_T, Tglob[scen], scen, time[scen], cfg)
    else:
        print("No alternative method is currenty implemented.")

    # save the global trend paramters if requested
    if save_params:
        dir_mesmer_params_gt = dir_mesmer_params + "global/global_trend/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_gt):
            os.makedirs(dir_mesmer_params_gt)
            print("created dir:", dir_mesmer_params_gt)
        joblib.dump(
            params_gt_T,
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

    return params_gt_T


def train_gt_T_ic_LOWESS_OLS(params_gt_T, Tglob, scen, time, cfg):
    """ Derive global trend (emissions + volcanoes) parameters from single ESM ic ensemble with LOWESS smoother for ghg trend and OLS for volcanic spikes.

    Args:
    - params_gt_T (dict): parameter dictionary containing keys which do not depend on applied method
        ['targ'] (emulated variable, i.e., tas or tblend, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, i.e., ic, str)
        ['method'] (applied method, i.e., LOWESS_OLS_saod, str)
        ['scenarios_tr'] (emission scenarios used for training, list of strs with 1 entry because ic ensemble)
    - Tglob (np.ndarray): 2d global mean temperature array (run, time) of globally-averaged temperature anomaly time series
    - scen (str): scenario to derive gt and associated parameters for
    - time (np.ndarray): 1d array of years
    - cfg (module): config file containnig metadata needed to load in stratospheric AOD time series

    Returns:
    - params_gt_T (dict): parameter dictionary containing original keys plus scenario-specific
        [scen]['frac_lowess'] (fraction of the data used when estimating each y-value, float)
        [scen]['coef_saod'] (stratospheric AOD OLS coefficient for global temperature variability, float)
        [scen]['gt'] (1d array of global temperature trend with volcanic spikes)

    """

    # specify necessary variables from cfg file
    gen = cfg.gen
    dir_obs = cfg.dir_obs

    # derive smooth trend with volcano spikes with simple statistical model

    # smooth trend
    nr_runs = Tglob.shape[
        0
    ]  # nr runs, because dim Tglob = (run,time)
    nr_ts = len(time)  # number time steps
    # average across all runs to get a first smoothing
    av_Tglob = np.zeros(nr_ts)
    for run in np.arange(nr_runs):
        av_Tglob += Tglob[run]
    av_Tglob /= nr_runs
    # apply lowess smoother to further smooth the Tglob time series
    frac_lowess = (
        50 / nr_ts
    )  # rather arbitrarily chosen value that gives a smooth enough trend,
    # open to changes but if much smaller, Tglob trend ends up very wiggly
    gt_lowess = lowess(
        av_Tglob, np.arange(nr_ts), return_sorted=False, frac=frac_lowess, it=0
    )

    # start setting up dict for the necessary params
    params_gt_T[scen] = {}
    params_gt_T[scen]["frac_lowess"] = frac_lowess
    
    if time.min()<2000: # check if historical period is part of the training runs
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
            gv_all_for_aod[i : i + nr_aod_obs] = (Tglob[run] - gt_lowess)[:nr_aod_obs]
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
        params_gt_T[scen]["coef_saod"] = linreg_gv_volc.coef_[0]
        gt = contrib_volc + gt_lowess

    else:
        # no volcanic eruptions in the scenario time period
        params_gt_T[scen]["coef_saod"] = None # not sure if needed
        gt = gt_lowess

    params_gt_T[scen]["time"] = time
    params_gt_T[scen]["gt"] = gt
    

    return params_gt_T
