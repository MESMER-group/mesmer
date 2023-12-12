# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train global variability module of MESMER.
"""


import numpy as np
import xarray as xr

from mesmer.io.save_mesmer_bundle import save_mesmer_data
from mesmer.stats import _fit_auto_regression_scen_ens, _select_ar_order_scen_ens


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

    **kwargs : Any
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
        # specify parameters employed for AR process fitting

        kwargs["max_lag"] = kwargs.get("max_lag", 12)
        kwargs["sel_crit"] = kwargs.get("sel_crit", "bic")

        params_gv = train_gv_AR(params_gv, gv, kwargs["max_lag"], kwargs["sel_crit"])
    else:
        raise ValueError("No such method and/ or weighting approach.")

    # save the global variability parameters if requested
    if save_params:
        save_mesmer_data(
            params_gv,
            cfg.dir_mesmer_params,
            "global",
            "global_variability",
            filename_parts=[
                "params_gv",
                method_gv,
                *preds_gv,
                targ,
                esm,
                *scenarios_tr,
            ],
        )

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

    max_lag : int
        maximum number of lags considered during fitting

    sel_crit : str
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

    import statsmodels.api as sm

    params_gv["max_lag"] = max_lag
    params_gv["sel_crit"] = sel_crit

    # create temporary DataArray objects
    data = [xr.DataArray(data, dims=["run", "time"]) for data in gv.values()]

    AR_order = _select_ar_order_scen_ens(
        *data, dim="time", ens_dim="run", maxlag=max_lag, ic=sel_crit
    )
    params = _fit_auto_regression_scen_ens(
        *data, dim="time", ens_dim="run", lags=AR_order
    )

    # TODO: remove np.float64(...) (only here so the tests pass)
    params_gv["AR_order_sel"] = AR_order.item()
    params_gv["AR_int"] = np.float64(params.intercept.values)
    params_gv["AR_coefs"] = params.coeffs.values.squeeze()
    params_gv["AR_std_innovs"] = np.float64(params.standard_deviation.values)

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
