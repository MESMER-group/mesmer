# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to create global variability emulations with MESMER.
"""

import numpy as np
import xarray as xr

from mesmer.io.save_mesmer_bundle import save_mesmer_data
from mesmer.stats import draw_auto_regression_uncorrelated


def create_emus_gv(params_gv, preds_gv, cfg, save_emus=True):
    """Create global variablity emulations for specified method.

    Parameters
    ----------
    params_gv : dict
        Parameters dictionary.

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - [xx] (additional keys depend on employed method and are listed in
          train_gv_T_method() function)

    preds_gv : dict
        nested dictionary of predictors for global variability with keys

        - [pred][scen]  (1d/2d arrays (time)/(run, time) of predictor for specific
          scenario)

    cfg : module
        config file containing metadata

    save_emus : bool, optional
        determines if emulation is saved or not, default = True

    Returns
    -------
    emus_gv : dict
        global variability emulations dictionary with keys

        - [scen] (2d array  (emus, time) of global trend emulation time series)

    Notes
    -----
    - Assumptions:
        - if no preds_gv needed, pass time as predictor instead such that can get info
          about how many scenarios / ts per scenario should be drawn for stochastic part

    """

    # specify necessary variables from config file
    nr_emus_v = cfg.nr_emus_v
    seed_all_scens = cfg.seed[params_gv["esm"]]

    pred_name = list(preds_gv.keys())[0]
    scens_out = list(preds_gv[pred_name].keys())

    if scens_out != list(seed_all_scens.keys()):
        raise ValueError(
            "The scenarios which should be emulated do not have a seed assigned in the"
            " config file. The emulations cannot be created."
        )

    # set up dict for emulations of global variability with emulated scenarios as keys
    emus_gv = {}

    for scen in scens_out:

        time_axis = 1 if preds_gv[pred_name][scen].ndim > 1 else 0

        nr_ts_emus_v = preds_gv[pred_name][scen].shape[time_axis]

        # apply the chosen method
        if params_gv["method"] == "AR":
            emus_gv[scen] = create_emus_gv_AR(
                params_gv, nr_emus_v, nr_ts_emus_v, seed_all_scens[scen]["gv"]
            )
        else:
            raise ValueError("The chosen method is currently not implemented.")
            # if the emus should depend on the scen, scen needs to be passed to the fct

    # save the global variability emus if requested
    if save_emus:
        save_mesmer_data(
            emus_gv,
            cfg.dir_mesmer_emus,
            "global",
            "global_variability",
            filename_parts=[
                "emus_gv",
                params_gv["method"],
                *params_gv["preds"],
                params_gv["targ"],
                params_gv["esm"],
                *scens_out,
            ],
        )

    return emus_gv


def create_emus_gv_AR(params_gv, nr_emus_v, nr_ts_emus_v, seed):
    """Draw global variablity emulations from an AR process.

    Parameters
    ----------
    params_gv : dict
        Parameters dictionary.

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - ["max_lag"] (maximum lag considered when finding suitable AR model, int)
        - ["sel_crit"] (selection criterion applied to find suitable AR model, str)
        - ["AR_int"] (intercept of the AR model, float)
        - ["AR_coefs"] (coefficients of the AR model for the lags which are contained in
          the selected AR model, list of floats)
        - ["AR_order_sel"] (selected AR order, int)
        - ["AR_std_innovs"] (standard deviation of the innovations of the selected AR
          model, float)

    nr_emus_v : int
        number of global variability emulations

    nr_ts_emus_v : int
        number of time steps in each global variability emulation

    seed : int
        esm and scenario specific seed for gv module to ensure reproducibility of
        results

    Returns
    -------
    emus_gv : dict
        global variability emulations dictionary with keys

        - [scen] (2d array  (emus, time) of global variability emulation time series)
    """

    # buffer so that initial start at 0 does not influence overall result
    # Should this buffer be based on the length of ar_lags instead of hard-coded?
    buffer = 50

    # re-name params for easier reading of code below
    ar_int = params_gv["AR_int"]
    ar_coefs = params_gv["AR_coefs"]
    AR_order_sel = params_gv["AR_order_sel"]
    AR_std_innovs = params_gv["AR_std_innovs"]

    # ensure ar_coefs are not a scalar
    ar_coefs = np.atleast_1d(ar_coefs)

    # if AR(0) process chosen, no AR_coefs are available -> to have code run
    # nevertheless ar_coefs and ar_lags are set to 0 (-> emus are created with
    # ar_int + innovs)
    if ar_coefs.size == 0:
        ar_coefs = [0]

    # only use the selected coeffs
    ar_coefs = ar_coefs[:AR_order_sel]

    # create intermediate ar_params Dataset
    # the variables are 1D (except coeffs)
    intercept = xr.DataArray(ar_int)
    coeffs = xr.DataArray(ar_coefs, dims="lags")
    variance = xr.DataArray(AR_std_innovs**2)

    ar_params = xr.Dataset(
        {"intercept": intercept, "coeffs": coeffs, "variance": variance}
    )

    emus_gv = draw_auto_regression_uncorrelated(
        ar_params,
        time=nr_ts_emus_v,
        realisation=nr_emus_v,
        seed=seed,
        buffer=buffer,
    )

    # get back the old order of the emus
    emus_gv = emus_gv.values.transpose()

    return emus_gv
