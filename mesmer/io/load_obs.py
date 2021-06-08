# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in observations which are saved locally.
"""


import numpy as np
import pandas as pd
import xarray as xr


def load_obs(targ, prod, lon, lat, cfg, sel_ref="native", ignore_nans=True):
    """Load observations which you previously downloaded.

    Parameters
    ----------
    targ : str
        target variable (e.g., "tblend")
    prod : str
        product (e.g., "best" or "cw")
    cfg : module
        config file containing metadata
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)
    sel_ref : str, optional
        selected reference period, (e.g., "native" (original one) or "esm" (the one of
        the esm))
    ignore_nans : bool, optional
        if True global average = average across available gp, if False nan value if not
        all gps available

    Returns
    -------
    var : dict
        variable anomaly dictionary with keys

        - [obs] (4d array (run, time, lat, lon) of variable whereby run=1 because only a
          single realization of real world)
    GVAR : dict
        area-weighted global mean variable anomaly dictionary with keys

        - [obs] (2d array (run, time) of globally-averaged variable anomaly time series)
    time : np.ndarray
        1d array of years

    """

    # define the function mapping
    targ_func_mapping = {"tblend": load_obs_tblend}
    # once start working with other vars, extend dict eg {"pr": load_obs_pr}

    # select the function to apply
    load_targ = targ_func_mapping[targ]

    var, time = load_targ(prod, lon, lat, cfg, sel_ref)

    # compute global average
    lons, lats = np.meshgrid(lon["c"], lat["c"])
    wgt_2d = np.cos(np.deg2rad(lats))
    wgt_3d = np.tile(wgt_2d, [len(time), 1]).reshape(
        [len(time), wgt_2d.shape[0], wgt_2d.shape[1]]
    )
    if (
        ignore_nans
    ):  # grid points with nans are left aside (ie global mean is global mean of non nan gp)
        masked_var = np.ma.masked_array(var, np.isnan(var))
        GVAR = np.ma.average(masked_var, axis=(1, 2), weights=wgt_3d)
    else:  # each field with nans inside will result in nan as global mean
        GVAR = np.average(var, axis=(1, 2), weights=wgt_3d)

    # to adhere to same standards as the ESMs, add "scenario" key ("obs" for observational time period)
    var_dict = {}
    GVAR_dict = {}
    var_dict["obs"] = np.expand_dims(
        var, axis=0
    )  # same format as if were an ESM with a single run
    GVAR_dict["obs"] = np.expand_dims(GVAR, axis=0)

    return var_dict, GVAR_dict, time


def load_obs_tblend(prod, lon, lat, cfg, sel_ref):
    """Load spatially infilled tblend observations. Currently available: best and cw.

    Parameters
    ----------
    prod : str
        product (e.g., "best" or "cw")
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)
    cfg : module
        config file containing metadata
    sel_ref : str, optional
        selected reference period, (e.g., "native" (original one) or "esm" (the one of
        the esm))

    Returns
    -------
    var : np.ndarray
        3d array (time, lat, lon) of blended temperatures
    time np.ndarray
        1d array of years

    """

    if prod == "best":
        path = (
            cfg.dir_obs + "blended_temperatures/" + prod + "/best_ann_1850-2019_g025.nc"
        )
        tblend = xr.open_dataset(path).temperature
        tblend["time"] = np.arange(1850, 2020)
    elif prod == "cw":
        path = (
            cfg.dir_obs + "blended_temperatures/" + prod + "/cw_ann_1850-2018_g025.nc"
        )
        tblend = xr.open_dataset(path).temperature_anomaly
        tblend["time"] = np.arange(1850, 2019)

    # check if lon / lat of tblend matches lon / lat of cmip models
    if (tblend.lon != lon["c"]).any() or (tblend.lat != lat["c"]).any():
        raise ValueError(
            "The grids of the ESM output and the observations do not agree."
        )
        tblend = []

    # extract time
    time = tblend.time.values

    # rebaseline and extract array if requested, otherwise just extract array
    if sel_ref == "native":
        tblend = tblend.values
    elif sel_ref == "esm":
        ref = cfg.ref
        tblend = (
            tblend.values
            - tblend.sel(time=slice(ref["start"], ref["end"])).mean(dim="time").values
        )
        # .mean() ignores nan in the selected time slice. Only if all time steps are nans, the mean is a nan too.
    else:
        raise ValueError("No such re-baselining is currently implemented.")

    return tblend, time


def load_strat_aod(time, dir_obs):
    """Load observed global stratospheric aerosol optical depth time series.

    Parameters
    ----------
    time : np.ndarray
        1d array of years the AOD time series is required for
    dir_obs : str
        pathway to observations

    Returns
    -------
    aod_obs : np.ndarray
        1d array of observed global stratospheric AOD time series

    Notes
    -----
    - Assumption: time covers max full extend historical period (i.e., 1850 - 2014 for
      cimp6, 1850 - 2005 for cmip5)
    - potentially TODO: check if want to integrate it into load_obs() fct somehow, but
      likely not as it is quite different from other obs

    """

    path_file = dir_obs + "aerosols/isaod_gl.dat"
    ts = pd.read_csv(
        path_file, delim_whitespace=True, skiprows=11, names=("year", "month", "AOD")
    )
    date_range = pd.date_range(
        *(
            pd.to_datetime(
                [
                    str(ts["year"].iloc[0]) + "-" + str(ts["month"].iloc[0]),
                    str(ts["year"].iloc[-1]) + "-" + str(ts["month"].iloc[-1]),
                ]
            )
            + pd.offsets.MonthEnd()
        ),
        freq="m"
    )
    aod_obs = xr.DataArray(
        ts["AOD"].values, dims=("time",), coords=dict(time=("time", date_range))
    )

    aod_obs = aod_obs.groupby("time.year").mean("time")
    aod_obs = aod_obs.sel(year=slice(str(time[0]), str(time[-1]))).values

    return aod_obs
