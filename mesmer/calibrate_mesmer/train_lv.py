# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train local variability module of MESMER.
"""


import os

import joblib
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal

from mesmer.core.auto_regression import _fit_auto_regression_xr

from .train_utils import get_scenario_weights, stack_predictors_and_targets


def train_lv(preds, targs, esm, cfg, save_params=True, aux={}, params_lv={}):
    """Derive local variability (i.e., natural variabiliy) parameters.

    Parameters
    ----------
    preds : dict
        empty dictionary if none, else nested dictionary of predictors with keys

        - [pred][scen]  (1d/ 2d arrays (time)/(run, time) of predictor for specific
          scenario)
    targs : dict
        nested dictionary of targets with keys

        - [targ][scen] (3d array (run, time, gp) of target for specific scenario)
    esm : str
        associated Earth System Model (e.g., "CanESM2" or "CanESM5")
    cfg : module
        config file containing metadata
    save_params : bool, optional
        determines if parameters are saved or not, default = True
    aux : dict, optional
        provides auxiliary variables needed for lv method at hand

        - [var] (Xd arrays of auxiliary variable)
    params_lv : dict, optional
        pass the params_lv dict, if it already exists so that builds upon that one

    Returns
    -------
    params_lv : dict
        dictionary of local variability paramters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - ["part_model_in_lt"] (states if part of the model is saved in params_lt, bool)
        - ["method_lt_each_gp_sep"] (states if local trends method is applied to each
          grid point separately, bool)
        - [xx] (additional params depend on employed lv method)

    Notes
    -----
    - Assumptions:
        - all targets use same approach and same predictors
        - each predictor and each target has the same scenarios as keys
        - all available scenarios are used for training
        - identified parameters are valid for all training scenarios
        - if historical data is used for training, it has its own scenario
        - need to pass the params_lv dict if it already exists so that can continue to
          build on it
    - Disclaimer:
        - currently no method with preds implemented; but already have in there for
          consistency
    - TODO:
        - add ability to weight samples differently than equal weight for each scenario
          in AR process

    """

    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # because same approach for each targ
    pred_names = list(preds.keys())

    # specify necessary variables from config file
    wgt_scen_tr_eq = cfg.wgt_scen_tr_eq

    preds_lv = []
    # check if any preds from pr
    if len(params_lv) > 0:
        [preds_lv.append(pred) for pred in params_lv["preds"]]
    # for now only gv implemented, but could easily extend to rv (regional) lv (local)
    # if wanted such preds
    for pred in pred_names:
        if "gv" in pred:
            preds_lv.append(pred)
    # add new predictors to params_lv
    if len(params_lv) > 0:
        params_lv["preds"] = preds_lv

    method_lv = cfg.methods[targ_name]["lv"]

    scenarios_tr = list(targs[targ_name].keys())

    # prepare predictors and targets
    __, y = stack_predictors_and_targets(preds, targs)

    wgt_scen_eq = get_scenario_weights(targs[targ_name])
    if wgt_scen_tr_eq is False:
        wgt_scen_eq[:] = 1  # each sample same weight

    if len(params_lv) == 0:
        print("Initialize params_lv dictionary")
        params_lv = {}
        params_lv["targs"] = targ_names
        params_lv["esm"] = esm
        params_lv["method"] = method_lv
        params_lv["preds"] = preds_lv
        params_lv["scenarios"] = scenarios_tr
        params_lv["part_model_in_lt"] = False

    if "AR1_sci" in method_lv and wgt_scen_tr_eq:

        # assumption: target values I feed in here is already ready for AR1_sci method
        # if were to add any other method before (ie introduce Link et al method for
        # large-scale teleconnections), would have to execute it first & fit this one on
        # residuals

        params_lv = train_lv_AR1_sci(params_lv, targs, y, wgt_scen_eq, aux, cfg)
    else:
        raise ValueError(
            "The chosen method and / or weighting approach is not implemented."
        )

    # overwrites lv module if already exists, i.e., assumption: always lt before lv
    if save_params:
        dir_mesmer_params = cfg.dir_mesmer_params
        dir_mesmer_params_lv = dir_mesmer_params + "local/local_variability/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_lv):
            os.makedirs(dir_mesmer_params_lv)
            print("created dir:", dir_mesmer_params_lv)
        filename_parts = [
            "params_lv",
            method_lv,
            *preds_lv,
            *targ_names,
            esm,
            *scenarios_tr,
        ]
        filename_params_lv = dir_mesmer_params_lv + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_lv, filename_params_lv)

    return params_lv


def train_lv_AR1_sci(params_lv, targs, y, wgt_scen_eq, aux, cfg):
    """Derive parameters for AR(1) process with spatially-correlated innovations.

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
    targs : dict
        nested dictionary of targets with keys

        - [targ][scen] with 3d arrays (run, time, gp)
    y : np.ndarray
        3d array (sample, gp, targ) of targets
    wgt_scen_eq : np.ndarray
        1d array (sample) of sample weights
    aux : dict
        provides auxiliary variables needed for lv method at hand

        - ["phi_gc"] (Xd arrays of auxiliary variable)
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
        - do for each target variable independently
        - the variability is Gaussian
        - each scenario receives the same weight during training
    - Potential TODO:
        - add possibility to account for cross-correlation between different variables
          (i.e., joint instead of independent emulation)

    """

    print("Derive parameters for AR(1) processes with spatially correlated innovations")
    # AR(1)
    params_lv["AR1_int"] = {}
    params_lv["AR1_coef"] = {}
    params_lv["AR1_std_innovs"] = {}
    params_lv["L"] = {}  # localisation radius
    # empirical cov matrix of the local variability trained on here
    params_lv["ecov"] = {}
    params_lv["loc_ecov"] = {}  # localized empirical cov matrix
    # localized empirical cov matrix of the innovations of the AR(1) process
    params_lv["loc_ecov_AR1_innovs"] = {}

    # largely ignore prepared targets and use original ones instead because in original
    # easier to loop over individ runs / scenarios

    # fit parameters for each target individually
    for targ_name, targ in targs.items():

        params_scen = list()
        for scen, data in targ.items():

            nr_runs, nr_ts, nr_gps = data.shape

            # create temporary DataArray
            data = xr.DataArray(data, dims=("run", "time", "cell"))

            params = _fit_auto_regression_xr(data, dim="time", lags=1)
            params = params.mean("run")

            params_scen.append(params)

        params_scen = xr.concat(params_scen, dim="scen")
        params_scen = params_scen.mean("scen")

        params_lv["AR1_int"][targ_name] = params_scen.intercept.values
        params_lv["AR1_coef"][targ_name] = params_scen.coeffs.values.squeeze()
        params_lv["AR1_std_innovs"][targ_name] = params_scen.standard_deviation.values

        # determine localization radius, empirical cov matrix, and localized ecov matrix
        (
            params_lv["L"][targ_name],
            params_lv["ecov"][targ_name],
            params_lv["loc_ecov"][targ_name],
        ) = train_lv_find_localized_ecov(y[targ_name], wgt_scen_eq, aux, cfg)

        # compute localized cov matrix of the innovations of the AR(1) process
        loc_ecov_AR1_innovs = _adjust_ecov_ar1(
            params_lv["loc_ecov"][targ_name], params_lv["AR1_coef"][targ_name]
            )

        params_lv["loc_ecov_AR1_innovs"][targ_name] = loc_ecov_AR1_innovs

    return params_lv


def _adjust_ecov_ar1(ecov, ar_coefs):
    """
    adjust localized empirical covariance matrix for autoregressive process of order 1

    Parameters
    ----------
    ecov : 2D np.array
        Empirical covariance matrix.
    ar_coefs : 1D np.array
        The coefficients of the autoregressive process of order 1.

    Returns
    -------
    adjusted_ecov : np.array
        Adjusted empirical covariance matrix.

    Notes
    -----
    - Adjusts ``ecov`` for an AR(1) process according to [1]_, eq (8).

    - The formula is specific for an AR(1) process, see also https://github.com/MESMER-group/mesmer/pull/167#discussion_r912481495

    - According to [2]_ "The multiplication with the ``reduction_factor`` scales the
      empirical standard error under the assumption of an autoregressive process of 
      order 1 [3]_. This accounts for the fact that the variance of an autoregressive
      process is larger than that of the driving white noise process."

    - This formula is wrong in [1]_. However, it is correct in the code. See also [2]_
       and [3]_.

.. [1] Beusch, L., Gudmundsson, L., and Seneviratne, S. I.: Emulating Earth system model
   temperatures with MESMER: from global mean temperature trajectories to grid-point-
   level realizations on land, Earth Syst. Dynam., 11, 139–159, 
   https://doi.org/10.5194/esd-11-139-2020, 2020.

.. [2] Humphrey, V. and Gudmundsson, L.: GRACE-REC: a reconstruction of climate-driven
   water storage changes over the last century, Earth Syst. Sci. Data, 11, 1153–1170,
   https://doi.org/10.5194/essd-11-1153-2019, 2019. 

.. [3] Cressie, N. and Wikle, C. K.: Statistics for spatio-temporal data, John Wiley &
   Sons, Hoboken, New Jersey, USA, 2011.
    """

    reduction_factor = np.sqrt(1 - ar_coefs ** 2)
    reduction_factor = np.atleast_2d(reduction_factor)  # so it can be transposed

    # equivalent to ``diag(reduction_factor) @ ecov @ diag(reduction_factor)``
    return reduction_factor * reduction_factor.T * ecov


def train_lv_find_localized_ecov(y, wgt_scen_eq, aux, cfg):
    """
    Find suitable localization radius for empirical covariance matrix and derive
    localized empirical cov matrix.

    Parameters
    ----------
    y : np.ndarray
        2d array (sample, gp) of specific target
    wgt_scen_eq : np.ndarray
        1d array (sample) of sample weights
    aux : dict
        provides auxiliary variables needed for lv method at hand

        - ["phi_gc"] (dict with localisation radii as keys and each containing a 2d
          array (gp, gp) of of Gaspari-Cohn correlation matrix
    cfg : module
        config file containing metadata

    Returns
    -------
    L_sel : numpy.int64
        selected localization radius
    ecov : np.ndarray
        2d empirical covariance matrix array (gp, gp)
    loc_ecov : np.ndarray
        2d localized empirical covariance matrix array (gp, gp)

    Notes
    -----
    - Function could also handle determining ecov of several variables but would all
      have to be passed in same 2d y array (with corresponding wgt_scen_eq,
      aux["phi_gc"] shapes)

    """

    return _find_localized_empirical_covariance(
        y, wgt_scen_eq, aux["phi_gc"], cfg.max_iter_cv
    )


def _find_localized_empirical_covariance(y, wgt_scen_eq, phi_gc, max_iter_cv):

    # derive the indices for the cross validation
    nr_samples, nr_gridpoints = y.shape
    nr_it = np.min([nr_samples, max_iter_cv])
    idx_cv_out = np.zeros([nr_it, nr_samples], dtype=bool)
    for i in range(nr_it):
        idx_cv_out[i, i::max_iter_cv] = True

    # spatial cross-correlations with specified cross val folds
    L_set = sorted(phi_gc.keys())  # the localisation radii to loop through

    llh_max = float("-inf")
    llh_cv_sum = {}

    for L in L_set:
        llh_cv_sum[L] = 0

        for it in range(nr_it):
            # extract folds
            y_est = y[~idx_cv_out[it]]  # to estimate params
            y_cv = y[idx_cv_out[it]]  # to crossvalidate the estimate
            wgt_scen_eq_est = wgt_scen_eq[~idx_cv_out[it]]
            wgt_scen_eq_cv = wgt_scen_eq[idx_cv_out[it]]

            # compute ecov and likelihood of out fold to be drawn from it
            ecov = np.cov(y_est, rowvar=False, aweights=wgt_scen_eq_est)
            loc_ecov = phi_gc[L] * ecov

            # we want the mean of the normal distribution to be 0
            mean_0 = np.zeros(nr_gridpoints)

            # NOTE: 90 % of time is spent here - not much point optimizing the rest
            llh_cv_each_sample = multivariate_normal.logpdf(
                y_cv, mean=mean_0, cov=loc_ecov, allow_singular=True
            )
            # allow_singular = True because stms ran into singular matrices
            # ESMs eg affected: CanESM2, CanESM5, IPSL-CM5A-LR, MCM-UA-1-0
            # -> reassuring that saw that in these ESMs L values where matrix
            # is not singular yet can end up being selected

            # each cv sample gets its own likelihood -> can sum them up for overall
            # likelihood
            # sum over all samples = wgt average * nr_samples
            llh_cv_fold_sum = (
                np.average(llh_cv_each_sample, weights=wgt_scen_eq_cv)
                * wgt_scen_eq_cv.size
            )

            # add to full sum over all folds
            llh_cv_sum[L] += llh_cv_fold_sum

        # experience tells: once stop selecting larger localisation radii, will not
        # start again. Better to stop once max is reached (to limit computational effort
        # and amount of singular matrices).
        if llh_cv_sum[L] > llh_max:
            L_sel = L
            llh_max = llh_cv_sum[L]
            print("Newly selected L =", L_sel)
        else:
            print("Final selected L =", L_sel)
            break

    ecov = np.cov(y, rowvar=False, aweights=wgt_scen_eq)
    loc_ecov = phi_gc[L_sel] * ecov

    return L_sel, ecov, loc_ecov
