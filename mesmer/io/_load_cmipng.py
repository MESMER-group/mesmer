# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in cmip5 and cmip6 data from the cmip-ng archive at ETHZ.
"""

import glob
import os
import warnings

import numpy as np
import xarray as xr

import mesmer

from ..io.load_constant_files import load_regs_ls_wgt_lon_lat
from ..utils import convert_dict_to_arr, extract_land


def load_cmip_data_all_esms(esms, scenarios, threshold_land, use_hfds, cfg):
    """Load tas and (potentially) hfds for several ESMs from cmip-ng archive at ETHZ

    Parameters
    ----------
    esms : list of str
        List of esms to load the data for
    scenarios : list of str
        List of scenarios to load the data for
    threshold_land : float
        Minimum land fraction so a grid point is considered.
    use_hfds : bool
        Whether to load hfds data.
    cfg : configuration
        Configuration values. Required are ``cfg.gen``, ``cfg.ref``, and
        ``cfg.dir_cmipng``.

    Returns
    -------
    time, lon, lat, ls, tas, gsat, ghfds

    Notes
    -----
    ghfds is None if ``use_hfds`` is set to False
    """

    # tas with global coverage
    tas_g = {}

    # global mean tas
    gsat = {}

    # global mean hfds (needed as predictor)
    ghfds = {} if use_hfds else None

    time = {}

    for esm in esms:
        print(f"- loading data for {esm}")

        time[esm] = {}

        # temporary dicts to gather data over scenarios
        tas_temp, gsat_temp, ghfds_temp = {}, {}, {}
        for scen in scenarios:
            out = load_cmipng("tas", esm, scen, cfg)

            if out[0] is None:
                warnings.warn(f"Scenario {scen} does not exist for tas for ESM {esm}")
                continue

            # unpack data
            tas_temp[scen], gsat_temp[scen], lon, lat, time[esm][scen] = out

            if use_hfds:
                _, ghfds_temp[scen], _, _, _ = load_cmipng("hfds", esm, scen, cfg)

        tas_g[esm] = convert_dict_to_arr(tas_temp)
        gsat[esm] = convert_dict_to_arr(gsat_temp)

        if use_hfds:
            ghfds[esm] = convert_dict_to_arr(ghfds_temp)

    # load in the constant files
    _, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(lon=lon, lat=lat)

    # extract land
    tas, _, ls = extract_land(tas_g, wgt=wgt_g, ls=ls, threshold_land=threshold_land)

    return time, lon, lat, ls, tas, gsat, ghfds


def extract_time_lon_lat_wgt3d(data):
    """
    Extract time, longitude, latitude, and 3d weights from file from ETHZ cmip-ng
    archive.

    Parameters
    ----------
    data : xr.core.dataset.Dataset
        dataset to extract time, lon, lat, and wgt3d from

    Returns
    -------
    time : np.ndarray
        1d array of years
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)
    wgt3d : np.ndarray
        3d array (time, lat, lon) of area weight of each grid point
    """

    # extract time
    time = data.time.dt.year.values

    # extract longitude and latitude at center of grid cell vectors
    lon = {}
    lat = {}
    lon["c"] = data.lon.values  # longitude at center (c) of grid cell
    lat["c"] = data.lat.values

    _, lats = np.meshgrid(lon["c"], lat["c"])
    wgt2d = np.cos(np.deg2rad(lats))
    wgt3d = np.tile(wgt2d, (time.size, 1, 1))

    return time, lon, lat, wgt3d


def _find_files_cmipng(gen, esm, var, scenario, dir_cmipng):
    """Find filename in ETHZ cmip-ng archive.

    Parameters
    ----------
    gen : int
        generation (5 or 6)
    esm : str
        Earth System Model (e.g., "CanESM2" or "CanESM5")
    var : str
        variable (e.g., "tas", "tos")
    scenario : str
        scenario (e.g., "rcp85" or "ssp585")
    dir_cmipng : str
        path to cmip-ng archive

    Returns
    -------
    path_runs_list : list
        list of paths to all the filenames found for the given input

    Notes
    -----
    - Not fool-proof enough yet for ESMs with different forcings. Could/ should be
      improved if there is time.
    - TODO:
        - improve and extend list of excluded runs/ ESMs for all cmip generations and
          variables
    """

    # for cmip5-ng
    if gen == 5:
        dir_name = os.path.join(dir_cmipng, var)

        if var == "tas":
            esms_excl = ["GISS-E2-H", "EC-EARTH"]
            # list of all ESMs which have excluded runs
            runs_excl = [
                "tas_ann_GISS-E2-H_rcp85_r2i1p1_g025.nc",
                "tas_ann_EC-EARTH_rcp45_r14i1p1_g025.nc",
            ]  # list of excluded runs

        else:
            esms_excl = []
            runs_excl = []
            print("TO DO: create list of excluded runs / ESMs for this variable")

        path = os.path.join(dir_name, f"{var}_ann_{esm}_{scenario}_r*i1p1_g025.nc")
        path_runs_list = sorted(glob.glob(path))

    # for cmip6-ng
    if gen == 6:
        dir_name = os.path.join(dir_cmipng, var, "ann", "g025")

        # TODO: remove hard-coding
        if var == "tas":
            esms_excl = [
                "ACCESS-ESM1-5",
                "IPSL-CM6A-LR",
                "UKESM1-0-LL",
            ]  # list of all ESMs which have excluded runs
            runs_excl = [
                "tas_ann_ACCESS-ESM1-5_ssp245_r21i1p1f1_g025.nc",
                "tas_ann_IPSL-CM6A-LR_ssp434_r1i1p1f2_g025.nc",
                "tas_ann_IPSL-CM6A-LR_ssp460_r1i1p1f2_g025.nc",
                "tas_ann_UKESM1-0-LL_ssp126_r5i1p1f2_g025.nc",
                "tas_ann_UKESM1-0-LL_ssp126_r6i1p1f2_g025.nc",
                "tas_ann_UKESM1-0-LL_ssp126_r7i1p1f2_g025.nc",
                "tas_ann_UKESM1-0-LL_ssp370_r5i1p1f2_g025.nc",
                "tas_ann_UKESM1-0-LL_ssp370_r6i1p1f2_g025.nc",
                "tas_ann_UKESM1-0-LL_ssp370_r7i1p1f2_g025.nc",
            ]  # list of excluded runs
            # TODO: extend list of excluded runs / ESMs
        elif var == "hfds":
            esms_excl = ["ACCESS-ESM1-5", "IPSL-CM6A-LR", "UKESM1-0-LL"]
            runs_excl = [
                "hfds_ann_ACCESS-ESM1-5_ssp126_r7i1p1f1_g025.nc",
                "hfds_ann_ACCESS-ESM1-5_ssp245_r7i1p1f1_g025.nc",
                "hfds_ann_ACCESS-ESM1-5_ssp370_r7i1p1f1_g025.nc",
                "hfds_ann_ACCESS-ESM1-5_ssp585_r7i1p1f1_g025.nc",
                "hfds_ann_IPSL-CM6A-LR_ssp434_r1i1p1f2_g025.nc",
                "hfds_ann_IPSL-CM6A-LR_ssp460_r1i1p1f2_g025.nc",
                "hfds_ann_UKESM1-0-LL_ssp126_r5i1p1f2_g025.nc",
                "hfds_ann_UKESM1-0-LL_ssp126_r6i1p1f2_g025.nc",
                "hfds_ann_UKESM1-0-LL_ssp126_r7i1p1f2_g025.nc",
                "hfds_ann_UKESM1-0-LL_ssp370_r5i1p1f2_g025.nc",
                "hfds_ann_UKESM1-0-LL_ssp370_r6i1p1f2_g025.nc",
                "hfds_ann_UKESM1-0-LL_ssp370_r7i1p1f2_g025.nc",
            ]
        else:
            esms_excl = []
            runs_excl = []
            print("TO DO: create list of excluded runs / ESMs for this variable")

        #  TODO: how to handle p2?
        # )
        path = os.path.join(dir_name, f"{var}_ann_{esm}_{scenario}_r*i1p1f*_g025.nc")
        path_runs_list_scen = sorted(glob.glob(path))

        path = os.path.join(dir_name, f"{var}_ann_{esm}_historical_r*i1p1f*_g025.nc")
        path_runs_list_hist = sorted(glob.glob(path))

        # check if both scenario and historical run are available for this realization,
        # if yes add to path_runs_list
        path_runs_list = []
        for run in path_runs_list_scen:
            run_hist = run.replace(scenario, "historical")
            if run_hist in path_runs_list_hist:
                path_runs_list.append(run)

        # TODO: redecide if I am fine with CanESM5 p2 but not all scenarios or if I
        # prefer the worse p1 which has all scenarios code below = old version when used
        #  p2 instead
        # if esm == "CanESM5":
        #     variant = "r*i1p2f*"
        # else:
        #     variant = "r*i1p1f*"

        # path = os.path.join(dir_name, f"{var}_ann_{esm}_{scenario}_{variant}_g025.nc")
        # path_runs_list = sorted(glob.glob(path))

    if len(path_runs_list) == 0:  # if no entries found, return the empty list
        return path_runs_list

    # ordering does not work for ESMs with > 9 runs -> find first run + put in front
    index_first = 0
    for i, s in enumerate(path_runs_list):
        if "r1i1" in s:
            index_first = i
            # if r1i1 run exists, move it to front; else leave first run at front

    # move first run to begin of list
    path_runs_list.insert(0, path_runs_list.pop(index_first))

    # exclude faulty runs in archive
    if esm in esms_excl:
        for run_excl in runs_excl:
            for run in path_runs_list:
                if run_excl in run:
                    path_runs_list.remove(run)

    return path_runs_list


def load_cmipng(targ, esm, scen, cfg):
    """Load ESM runs from cmip-ng archive at ETHZ.

    Parameters
    ----------
    targ : str
        target variable (e.g., "tas")
    esm : str
        Earth System Model (e.g., "CanESM2" or "CanESM5")
    scen : str
        future scenario (e.g., "rcp85" or "ssp585")
    cfg : module
        config file containing metadata

    Returns
    -------
    targ : dict
        target variable anomaly dictionary with keys

        - [run] (3d array (time, lat, lon) of variable)
    GTARG : dict
        area-weighted global mean target variable anomaly dictionary with keys

        - [run]  (1d array of globally-averaged variable anomaly time series)
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)
    time : np.ndarray
        1d array of years
    """

    # check if scenario describes
    if scen[:2] == "h-":
        scen_fut = scen[2:]
    else:
        raise ValueError(
            "No version without historical time period is currently implemented."
        )

    # once start working with other vars, extend dict eg {"pr": load_cmipng_pr,
    # "hfds": load_cmipng_hfds, "tas": load_cmipng_tas}
    targ_func_mapping = {"hfds": load_cmipng_hfds, "tas": load_cmipng_tas}

    load_targ = targ_func_mapping[targ]

    targ, GTARG, lon, lat, time = load_targ(esm, scen_fut, cfg)

    return targ, GTARG, lon, lat, time


def _load_cmipng_file(run_path, gen, scen):
    """Load file in ETHZ cmip-ng archive.

    Parameters
    ----------
    run_path : str
        path to file
    gen : int
        generation (5 or 6)
    scen : str
        future scenario (e.g., "rcp85" or "ssp585")

    Returns
    -------
    data : xr.core.dataset.Dataset
        loaded dataset
    run : int
        realization index
    """

    # account for difference in naming convention in cmipx-ng archives
    if gen == 5:

        # use_cftime because of employed calendar,
        data = xr.open_dataset(run_path, use_cftime=True)

        # rename to time for consistency with cmip6
        data = data.rename({"year": "time"})

        # extract ens member
        run = int(data.attrs["source_ensemble"].split("r")[1].split("i")[0])

    if gen == 6:

        run_path_hist = run_path.replace(scen, "historical")
        paths = [run_path_hist, run_path]
        preprocess = None

        if "ssp534-over" in run_path:
            run_path_ssp_585 = run_path.replace(scen, "ssp585")

            paths.append(run_path_ssp_585)
            preprocess = _preprocess_ssp534over

        data = xr.open_mfdataset(paths, combine="by_coords", preprocess=preprocess)

        run = data.attrs["realization_index"]

        # wrap data to [-180, 180)
        data = mesmer.grid.wrap_to_180(data)

    return data, run


def load_cmipng_hfds(esm, scen, cfg):
    """Load ESM hfds runs from cmip-ng archive at ETHZ.

    Parameters
    ----------
    esm : str
        Earth System Model (e.g., "CanESM2" or "CanESM5")
    scen : str
        future scenario (e.g., "rcp85" or "ssp585")
    cfg : module
        config file containnig metadata

    Returns
    -------
    hfds : dict
        Downward Heat Flux at Sea Water Surface (hfds) anomaly dictionary with keys

        - [run] (3d array (time, lat, lon) of variable)
    GHFDS : dict
        area-weighted global mean hfds anomaly dictionary with keys

        - [run]  (1d array of globally-averaged tas anomaly time series)
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)
    time : np.ndarray
        1d array of years

    """

    return _load_cmipng_var(esm, scen, cfg, "hfds")


def load_cmipng_tas(esm, scen, cfg):
    """Load ESM tas runs from cmip-ng archive at ETHZ.

    Parameters
    ----------
    esm : str
        Earth System Model (e.g., "CanESM2" or "CanESM5")
    scen : str
        future scenario (e.g., "rcp85" or "ssp585")
    cfg : module
        config file containing metadata

    Returns
    -------
    tas : dict
        2-m air temperature anomaly dictionary with keys

        - [run] (3d array (time, lat, lon) of variable)
    GSAT : dict
        area-weighted global mean 2-m air temperature anomaly dictionary with keys

        - [run]  (1d array of globally-averaged tas anomaly time series)
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)
    time : np.ndarray
        1d array of years

    """

    return _load_cmipng_var(esm, scen, cfg, "tas")


def _load_cmipng_var(esm, scen, cfg, varn):

    # specify necessary variables from cfg
    gen = cfg.gen
    ref = cfg.ref
    dir_cmipng = cfg.dir_cmipng

    if ref["type"] == "first":
        raise ValueError("reference type 'first' is no longer supported")

    # find the files which fulfill the specifications
    path_runs_list = _find_files_cmipng(gen, esm, varn, scen, dir_cmipng)

    # exit function in case path_runs_list is empty (i.e. no files found)
    if len(path_runs_list) == 0:
        dta = dta_globmean = lon = lat = time = None
        return dta, dta_globmean, lon, lat, time

    # load data on global grid and compute anomalies
    dta = {}
    dta_ref = {}

    reference_period = slice(ref["start"], ref["end"])

    # load data
    for run_path in path_runs_list:

        # account for difference in naming convention in cmipx-ng archives
        ds, run = _load_cmipng_file(run_path, gen, scen)

        dta[run] = ds[varn].values
        dta_ref[run] = ds[varn].sel(time=reference_period).mean(dim="time")

    # compute anomalies
    if ref["type"] == "all":

        # mean over all runs
        dta_ref = xr.concat(list(dta_ref.values()), dim="run").mean("run").values

        for run in dta:
            dta[run] -= dta_ref

    elif ref["type"] == "individ":

        for run in dta:
            dta[run] -= dta_ref[run].values

    # extract time, longitude, latitude, and 3d weights from data
    time, lon, lat, wgt3d = extract_time_lon_lat_wgt3d(ds)

    # compute area-weighted mean
    # ATTENTION: does not account for land fraction within grid cells. i.e., coastal
    # grid cells count as full ocean grid cells. Expected to have negligible impact on
    # global mean.

    dta_globmean = {}
    for run in dta:
        # account for missing values over land
        dta_masked = np.ma.masked_array(dta[run], np.isnan(dta[run]))
        # no need to keep mask since False everywhere
        dta_globmean[run] = np.average(dta_masked, axis=(1, 2), weights=wgt3d).data

    return dta, dta_globmean, lon, lat, time


def _preprocess_ssp534over(ds):
    """
    Preprocess datasets to manage to combine historical, ssp585, and ssp534-over into
    single time series.

    Parameters
    ----------
    ds : xr.Dataset
        dataset to be concatenated with other datasets

    Returns
    -------
    ds : xr.Dataset
        dataset cut after 2039 unless the start year is after 2030

    Notes
    -----
    - ssp534over starts in 2040, before it follows ssp585
    - This pre-processing allows to concatenate the individual files without running
      into overlapping time periods
    - Code received from Mathias Hauser on 20201117 (personal exchange via slack)

    """

    first_year = ds.time.dt.year[0]

    if first_year < 2030:
        ds = ds.sel(time=slice(None, "2039"))

    return ds
