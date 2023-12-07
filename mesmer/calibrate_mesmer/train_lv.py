# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train local variability module of MESMER.
"""

import xarray as xr

from mesmer.io.save_mesmer_bundle import save_mesmer_data
from mesmer.stats import (
    _fit_auto_regression_scen_ens,
    adjust_covariance_ar1,
    find_localized_empirical_covariance,
)

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
        dictionary of local variability parameters

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
        raise ValueError("No such method and / or weighting approach.")

    # overwrites lv module if already exists, i.e., assumption: always lt before lv
    if save_params:
        save_mesmer_data(
            params_lv,
            cfg.dir_mesmer_params,
            "local",
            "local_variability",
            filename_parts=[
                "params_lv",
                method_lv,
                *preds_lv,
                *targ_names,
                esm,
                *scenarios_tr,
            ],
        )

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

        # create temporary DataArray
        dims = ("run", "time", "cell")
        data = [xr.DataArray(data, dims=dims) for data in targ.values()]

        params = _fit_auto_regression_scen_ens(*data, dim="time", ens_dim="run", lags=1)

        params_lv["AR1_int"][targ_name] = params.intercept.values
        params_lv["AR1_coef"][targ_name] = params.coeffs.values.squeeze()
        params_lv["AR1_std_innovs"][targ_name] = params.standard_deviation.values

        # determine localization radius, empirical cov matrix, and localized ecov matrix

        # y.dims = (sample, gridpoint)
        # wgt_scen_eq.dims = (sample,)
        # aux["phi_gc"].dims = (gridpoint, gripoint)
        # where sample = is a stacked "time, scenario, ensmember"

        res = train_lv_find_localized_ecov(y[targ_name], wgt_scen_eq, aux, cfg)
        params_lv["L"][targ_name] = res.localization_radius.values
        params_lv["ecov"][targ_name] = res.covariance.values
        params_lv["loc_ecov"][targ_name] = res.localized_covariance.values

        # adjust localized cov matrix with the coefficients of the AR(1) process
        loc_cov_ar1 = adjust_covariance_ar1(res.localized_covariance, params.coeffs)

        params_lv["loc_ecov_AR1_innovs"][targ_name] = loc_cov_ar1.values

    return params_lv


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
    localized_empirical_covariance : xr.Dataset
        Dataset containing three DataArrays:
    localization_radius : float
        Selected localization radius.
    covariance : xr.DataArray
        Empirical covariance matrix.
    localized_covariance : xr.DataArray
        Localized empirical covariance matrix.

    Notes
    -----
    - Function could also handle determining ecov of several variables but would all
      have to be passed in same 2d y array (with corresponding wgt_scen_eq,
      aux["phi_gc"] shapes)

    """

    data = xr.DataArray(y, dims=("sample", "cell"))
    weights = xr.DataArray(wgt_scen_eq, dims="sample")

    phi_gc = aux["phi_gc"]
    dims = ("cell_i", "cell_j")
    localizer = {k: xr.DataArray(v, dims=dims) for k, v in phi_gc.items()}

    dim = "sample"

    k_folds = cfg.max_iter_cv

    return find_localized_empirical_covariance(data, weights, localizer, dim, k_folds)
