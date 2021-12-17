# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in cmip5 and cmip6 data from the cmip-ng archive at ETHZ.
"""

import copy as copy
import glob as glob

import numpy as np
import xarray as xr


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

    lons, lats = np.meshgrid(lon["c"], lat["c"])
    wgt2d = np.cos(np.deg2rad(lats))
    wgt3d = np.tile(wgt2d, [len(time), 1]).reshape(
        [len(time), wgt2d.shape[0], wgt2d.shape[1]]
    )

    return time, lon, lat, wgt3d


def find_files_cmipng(gen, esm, var, scenario, dir_cmipng):
    """Find filname in ETHZ cmip-ng archive.

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
        dir_name = dir_cmipng + var + "/"

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

        path_runs_list = sorted(
            glob.glob(
                dir_name
                + var
                + "_ann_"
                + esm
                + "_"
                + scenario
                + "_"
                + "r*i1p1"
                + "_g025.nc"
            )
        )

    # for cmip6-ng
    if gen == 6:
        dir_name = dir_cmipng + var + "/ann/g025/"

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

        # print(
        #   "TO DO: rewrite selection of p2 in way that needs less space for CanESM5 (+ see if other ESMs need p2 too)"
        # )
        path_runs_list_scen = sorted(
            glob.glob(
                dir_name
                + var
                + "_ann_"
                + esm
                + "_"
                + scenario
                + "_"
                + "r*i1p1f*"
                + "_g025.nc"
            )
        )

        path_runs_list_hist = sorted(
            glob.glob(
                dir_name
                + var
                + "_ann_"
                + esm
                + "_historical_"
                + "r*i1p1f*"
                + "_g025.nc"
            )
        )

        # check if both scenario and historical run are available for this realization, if yes add to path_runs_list
        path_runs_list = []
        for run in path_runs_list_scen:
            run_hist = run.replace(scenario, "historical")
            if run_hist in path_runs_list_hist:
                path_runs_list.append(run)

        # TODO: redecide if I am fine with CanESM5 p2 but not all scenarios or if I prefer the worse p1 which has all scenarios
        # code below = old version when used p2 instead
        # if esm != "CanESM5":
        #    path_runs_list = sorted(
        #       glob.glob(
        #   dir_name
        #  + var
        # + "_ann_"
        # + esm
        # + "_"
        #      + scenario
        #     + "_"
        #    + "r*i1p1f*"
        #   + "_g025.nc"
        #   )
        # )

        # else:
        #   path_runs_list = sorted(
        #      glob.glob(
        #         dir_name
        #        + var
        #       + "_ann_"
        #      + esm
        #     + "_"
        #    + scenario
        #   + "_"
        #  + "r*i1p2f*"
        # + "_g025.nc"
        #    )
        # )
    if len(path_runs_list) == 0:  # if no entries found, return the empty list
        return path_runs_list

    # ordering does not work for ESMs with > 9 runs -> find first run + put in front
    index_first = 0
    for i, s in enumerate(path_runs_list):
        if "r1i1" in s:
            index_first = i
            # if r1i1 run exists, move it to front; else leave first run at front
    path_runs_list.insert(
        0, path_runs_list.pop(index_first)
    )  # move first run to begin of list

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

    targ_func_mapping = {"hfds": load_cmipng_hfds, "tas": load_cmipng_tas}
    # once start working with other vars, extend dict eg {"pr": load_cmipng_pr, "hfds": load_cmipng_hfds, "tas": load_cmipng_tas}

    load_targ = targ_func_mapping[targ]

    targ, GTARG, lon, lat, time = load_targ(esm, scen_fut, cfg)

    return targ, GTARG, lon, lat, time


def load_cmipng_file(run_path, gen, scen):
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
        data = (
            xr.open_dataset(run_path, use_cftime=True)
            .rename({"year": "time"})
            .roll(lon=72, roll_coords=True)
        )
        # use_cftime because of employed calendar,
        # rename to time for consistency with cmip6, roll so land in center
        data = data.assign_coords(
            lon=(((data.lon + 180) % 360) - 180)
        )  # assign_coords so that labels = reasonable
        run = int(
            data.attrs["source_ensemble"].split("r")[1].split("i")[0]
        )  # extract ens member

    if gen == 6:
        if "ssp534-over" in run_path:
            run_path_ssp_534over = run_path
            run_path_ssp_585 = run_path.replace(scen, "ssp585")
            run_path_hist = run_path.replace(scen, "historical")
            data = xr.open_mfdataset(
                [run_path_hist, run_path_ssp_585, run_path_ssp_534over],
                combine="by_coords",
                preprocess=preprocess_ssp534over,
            )
        else:  # for every other scenario
            run_path_ssp = run_path
            run_path_hist = run_path.replace(scen, "historical")
            data = xr.open_mfdataset([run_path_hist, run_path_ssp], combine="by_coords")

        data = data.roll(lon=72, roll_coords=True)  # roll so land in center
        data = data.assign_coords(
            lon=(((data.lon + 180) % 360) - 180)
        )  # assign_coords so that labels = reasonable
        data = data.sortby(["lat", "lon"])
        run = data.attrs["realization_index"]

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

    # find the files which fulfill the specifications
    path_runs_list = find_files_cmipng(gen, esm, varn, scen, dir_cmipng)

    # exit function in case path_runs_list is empty (i.e. no files found)
    if len(path_runs_list) == 0:
        dta = dta_global = lon = lat = time = None
        return dta, dta_global, lon, lat, time

    # load data on global grid and compute anomalies
    dta = {}
    run_nrs = {}

    for run_path in path_runs_list:
        # account for difference in naming convention in cmipx-ng archives
        data, run = load_cmipng_file(run_path, gen, scen)

        run_nrs[run_path] = run  # tmp saved this way for deriving anomalies later
        dta[run] = copy.deepcopy(data[varn].values)

        if ref["type"] == "all":
            if run_path == path_runs_list[0]:  # for the first run need to initialize
                dta_ref = (
                    data[varn]
                    .sel(time=slice(ref["start"], ref["end"]))
                    .mean(dim="time")
                    .values
                    * 1.0
                    / len(path_runs_list)
                )
            else:
                dta_ref += (
                    data[varn]
                    .sel(time=slice(ref["start"], ref["end"]))
                    .mean(dim="time")
                    .values
                    * 1.0
                    / len(path_runs_list)
                )  # sum up all ref climates

        if ref["type"] == "individ":
            dta_ref = (
                data[varn]
                .sel(time=slice(ref["start"], ref["end"]))
                .mean(dim="time")
                .values
            )
            dta[run] -= dta_ref  # compute anomalies

        # TODO: remove "first" (used for Leas first paper but does not really make sense)
        if ref["type"] == "first" and run == "1":
            # TO CHECK
            dta_ref = (
                data[varn]
                .sel(time=slice(ref["start"], ref["end"]))
                .mean(dim="time")
                .values
            )

    if ref["type"] == "all" or ref["type"] == "first":
        for run_path in path_runs_list:
            run = run_nrs[run_path]
            dta[run] -= dta_ref  # compute anomalies

    # extract time, longitude, latitude, and 3d weights from data
    time, lon, lat, wgt3d = extract_time_lon_lat_wgt3d(data)

    # compute area-weighted mean
    # ATTENTION: does not account for land fraction within grid cells. i.e., coastal
    # grid cells count as full ocean grid cells. Expected to have negligible impact on
    # global mean.

    dta_global = {}
    for run_path in path_runs_list:
        run = run_nrs[run_path]
        # account for missing values over land
        dta_masked = np.ma.masked_array(dta[run], np.isnan(dta[run]))
        # no need to keep mask since False everywhere
        dta_global[run] = np.average(dta_masked, axis=(1, 2), weights=wgt3d).data

    return dta, dta_global, lon, lat, time


def preprocess_ssp534over(ds):
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
