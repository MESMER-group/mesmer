# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to train global variability module of MESMER.
"""


import numpy as np
import statsmodels.api as sm
import xarray as xr
from packaging.version import Version

from mesmer.core.auto_regression import _fit_auto_regression_xr, _select_ar_order_xr
from mesmer.io.save_mesmer_bundle import save_mesmer_data


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
    **kwargs:
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
        # specifiy parameters employed for AR process fitting

        kwargs["max_lag"] = kwargs.get("max_lag", 12)
        kwargs["sel_crit"] = kwargs.get("sel_crit", "bic")

        params_gv = train_gv_AR(params_gv, gv, kwargs["max_lag"], kwargs["sel_crit"])
    else:
        msg = "The chosen method and / or weighting approach is currently not implemented."
        raise ValueError(msg)

    # save the global variability paramters if requested
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
    max_lag: int
        maximum number of lags considered during fitting
    sel_crit: str
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

    params_gv["max_lag"] = max_lag
    params_gv["sel_crit"] = sel_crit

    if Version(xr.__version__) >= Version("2022.03.0"):
        method = "method"
    else:
        method = "interpolation"

    # select the AR Order
    AR_order_scen = list()
    for scen in gv.keys():

        # create temporary DataArray
        data = xr.DataArray(gv[scen], dims=["run", "time"])

        AR_order = _select_ar_order_xr(data, dim="time", maxlag=max_lag, ic=sel_crit)

        # median over all ensemble members ("nearest" ensures an 'existing' lag is selected)
        AR_order = AR_order.quantile(q=0.5, **{method: "nearest"})
        AR_order_scen.append(AR_order)

    # median over all scenarios
    AR_order_scen = xr.concat(AR_order_scen, dim="scen")
    AR_order_sel = int(AR_order.quantile(q=0.5, **{method: "nearest"}).item())

    # determine the AR params for the selected AR order
    params_scen = list()
    for scen_idx, scen in enumerate(gv.keys()):
        data = gv[scen]

        # create temporary DataArray
        data = xr.DataArray(data, dims=("run", "time"))

        params = _fit_auto_regression_xr(data, dim="time", lags=AR_order_sel)
        params = params.mean("run")

        params_scen.append(params)

    params_scen = xr.concat(params_scen, dim="scen")
    params_scen = params_scen.mean("scen")

    # TODO: remove np.float64(...) (only here so the tests pass)
    params_gv["AR_order_sel"] = AR_order_sel
    params_gv["AR_int"] = np.float64(params_scen.intercept.values)
    params_gv["AR_coefs"] = params_scen.coeffs.values.squeeze()
    params_gv["AR_std_innovs"] = np.float64(params_scen.standard_deviation.values)

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
