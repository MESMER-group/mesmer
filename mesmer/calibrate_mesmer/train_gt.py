# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train global trend module of MESMER.
"""

import warnings

import numpy as np
import xarray as xr

from mesmer.io import load_strat_aod
from mesmer.io.save_mesmer_bundle import save_mesmer_data
from mesmer.stats.linear_regression import LinearRegression
from mesmer.stats.smoothing import lowess
from mesmer.utils import separate_hist_future


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

    # specify necessary variables from config
    method_gt = cfg.methods[targ]["gt"]
    preds_gt = cfg.preds[targ]["gt"]

    scenarios = list(var.keys())

    # initialize param dict and fill in the metadata which does not depend on the method
    params_gt = {}
    params_gt["targ"] = targ
    params_gt["esm"] = esm
    params_gt["method"] = method_gt
    params_gt["preds"] = preds_gt
    params_gt["scenarios"] = scenarios  # single entry in case of ic ensemble

    # apply the chosen method to the type of ensenble
    gt = {}
    if "LOWESS" in params_gt["method"]:
        # derive gt for each scen individually
        for scen in scenarios:
            gt[scen], frac_lowess = train_gt_ic_LOWESS(var[scen])
        params_gt["frac_lowess"] = frac_lowess
    else:
        raise ValueError("No alternative method to LOWESS is implemented for now.")

    # if hist included
    if scenarios[0].startswith("h-"):

        gt_s, time_s = separate_hist_future(gt, time, cfg)

        # compute median LOWESS estimate of historical part across all scenarios
        gt_lowess_hist_all_new = gt_s.pop("hist")

        gt_hist_median = np.median(gt_lowess_hist_all_new, axis=0)

        if params_gt["method"] == "LOWESS_OLSVOLC":
            var_s, time_s = separate_hist_future(var, time, cfg)

            params_gt["saod"], params_gt["hist"] = train_gt_ic_OLSVOLC(
                var_s["hist"], gt_hist_median, time_s["hist"]
            )

        elif params_gt["method"] == "LOWESS":
            params_gt["hist"] = gt_hist_median

        gt_to_distribute = gt_s
        params_gt["time"] = time_s

    else:
        gt_to_distribute = gt
        params_gt["time"] = time

    for scen, data in gt_to_distribute.items():
        params_gt[scen] = data.squeeze()

    # save the global trend paramters if requested
    if save_params:
        save_mesmer_data(
            params_gt,
            cfg.dir_mesmer_params,
            "global",
            "global_trend",
            filename_parts=(
                "params_gt",
                method_gt,
                *preds_gt,
                targ,
                esm,
                *scenarios,
            ),
        )

    return params_gt


def train_gt_ic_LOWESS(data):
    """
    Derive smooth global trend of variable from single ESM ic ensemble with LOWESS
    smoother.

    Parameters
    ----------
    data : np.ndarray
        2d array (run, time) of globally-averaged time series

    Returns
    -------
    gt_lowess : np.ndarray
        1d array of smooth global trend of variable
    frac_lowess : float
        fraction of the data used when estimating each y-value
    """

    data = xr.DataArray(data, dims=("ensemble", "time"))

    # average across all runs to get a first smoothing
    data = data.mean("ensemble")

    dim = "time"

    # apply lowess smoother to further smooth the Tglob time series
    # rather arbitrarily chosen value that gives a smooth enough trend,
    frac = 50 / data.sizes[dim]

    # open to changes but if much smaller, var trend ends up very wiggly
    frac_lowess_name = "50/nr_ts"

    gt_lowess = lowess(data, dim=dim, frac=frac).values

    return gt_lowess, frac_lowess_name


def train_gt_ic_OLSVOLC(var, gt_lowess, time, cfg=None):
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
    cfg : None
        Passing cfg is no longer required.

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

    if cfg is not None:
        warnings.warn(
            "Passing ``cfg`` to ``train_gt_ic_OLSVOLC`` is no longer necessary",
            FutureWarning,
        )

    nr_runs, nr_ts = var.shape

    # account for volcanic eruptions in historical time period
    # load in observed stratospheric aerosol optical depth
    aod_obs = load_strat_aod(time)
    # drop "year" coords - aod_obs does not have coords (currently)
    aod_obs = aod_obs.drop_vars("year")

    # repeat aod time series as many times as runs available
    aod_obs_all = xr.concat([aod_obs] * nr_runs, dim="year")

    nr_aod_obs = aod_obs.shape[0]
    if nr_ts != nr_aod_obs:
        raise ValueError(
            f"The number of time steps of the variable ({nr_ts}) and the saod "
            f"({nr_aod_obs}) do not match."
        )

    # extract global variability (which still includes volc eruptions) by removing
    # smooth trend from Tglob in historic period
    # (should broadcast, and flatten the correct way - hopefully)
    gv_all_for_aod = (var - gt_lowess).ravel()

    gv_all_for_aod = xr.DataArray(gv_all_for_aod, dims="year").expand_dims("x")

    lr = LinearRegression()

    # fit linear regression of gt to aod (because some ESMs react very strongly to
    # volcanoes)
    # no intercept to not artifically move the ts
    lr.fit(
        predictors={"aod_obs": aod_obs_all},
        target=gv_all_for_aod,
        dim="year",
        fit_intercept=False,
    )

    # extract the saod coefficient
    coef_saod = lr.params["aod_obs"].values

    # apply linear regression model to obtain volcanic spikes
    contrib_volc = lr.predict(predictors={"aod_obs": aod_obs})

    # merge the lowess trend wit the volc contribution
    gt = gt_lowess + contrib_volc.values.squeeze()

    return coef_saod, gt
