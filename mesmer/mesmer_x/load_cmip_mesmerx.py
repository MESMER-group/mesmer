# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in cmip5 and cmip6 data from the cmip-ng archive or ADDITIONAL DATA REPOSITORIES at ETHZ.
Because MESMER-X has to use variables that are not provided by CMIP6, it must support the use of other repositories.
WARNING: THIS CODE IS STILL ENTIRELY BASED ON THE ORIGINAL VERSION OF DATA STRCUTURES, WHICH ARE NESTED DICTIONARIES.
"""

import copy as copy
import glob as glob
import os

import cftime as cft
import numpy as np
import pandas as pd
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
    time = (
        data.time.values
    )  # years already extracted in 'update_time_axis', due to additional time formats introduced in other archives.

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


def find_files_cmip(esm, var, scenario, cfg, prescribed_members=None):
    """Find filename in ETHZ cmip-ng or any required archive.

    Parameters
    ----------
    esm : str
        Earth System Model (e.g., "CanESM2" or "CanESM5")
    var : str
        variable (e.g., "tas", "tos")
    scenario : str
        scenario (e.g., "rcp85" or "ssp585")
    cfg : module
        config file containing metadata

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

    # identifying correct arguments to look for file: directory name, exclusions, member_style, !!
    if var in ["tas", "hfds", "pr"]:
        if cfg.gen == 5:
            # directory name CMIP5-ng
            dir_name = cfg.dir_cmipng + var + "/"

            # preparing search of files
            prefix_path_file = dir_name + var + "_ann_" + esm + "_"
            memb_style = "r*i1p1"
            suffix_path_file = "_" + memb_style + "_g025.nc"

            # exclusions CMIP5-ng
            if var == "tas":
                esms_excl = ["GISS-E2-H"]  # list of all ESMs which have excluded runs
                runs_excl = [
                    "tas_ann_GISS-E2-H_rcp85_r2i1p1_g025"
                ]  # list of excluded runs
            else:
                esms_excl = []
                runs_excl = []
                print("TO DO: create list of excluded runs / ESMs for this variable")

        elif cfg.gen == 6:
            # directory name CMIP6-ng
            dir_name = cfg.dir_cmipng + var + "/ann/g025/"

            # preparing search of files
            prefix_path_file = dir_name + var + "_ann_" + esm + "_"
            memb_style = "r*i1p1f*"
            suffix_path_file = "_" + memb_style + "_g025.nc"

            # exclusions CMIP6-ng
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

    elif var in [
        "txx",
        "mrso",
        "mrsomean",
        "mrso_minmon",
        "fwixx",
        "fwixd",
        "fwisa",
        "fwils",
    ]:
        # styles for search
        memb_style = {5: "r*i1p1", 6: "r*i1p1f*"}[cfg.gen]
        resc_style = {5: "", 6: "_g*"}[cfg.gen]

        # preparing search of files
        prefix_path_file = os.path.join(cfg.dir_cmip_X, var + "*" + esm + "_")
        suffix_path_file = "_" + memb_style + resc_style + ".nc"

        # exclusions CMIP-X
        esms_excl = []
        runs_excl = []

    else:
        raise Exception("This variable has not been prepared yet.")

    # Looking for required files: scenario part
    path_runs_list_scen = sorted(
        glob.glob(prefix_path_file + scenario + suffix_path_file)
    )

    # Looking for required files: historical part
    path_runs_list_hist = sorted(
        glob.glob(prefix_path_file + "historical" + suffix_path_file)
    )

    # check if both scenario and historical run are available for this realization, if yes add to path_runs_list
    path_runs_list = []
    for run in path_runs_list_scen:
        run_hist = run.replace(scenario, "historical")
        if run_hist in path_runs_list_hist:
            path_runs_list.append(run)

    # selecting runs if prescribed selection
    if prescribed_members is not None:
        tmp = []
        for member in prescribed_members:
            check_runs = [member in run for run in path_runs_list]
            if np.any(check_runs):
                tmp.append( np.array(path_runs_list)[check_runs][0] )
        path_runs_list = tmp        

    if len(path_runs_list) == 0:  # if no entries found, return the empty list
        return path_runs_list

    # ordering all runs
    order = {}
    for s in path_runs_list:
        memb = str.split(s, "_")[-2][1:]
        order[int(memb[: memb.find("i")])] = s
    sorted_order = list(order.keys())
    sorted_order.sort(reverse=False)
    # putting at the end in the correct order, removing the first occurence
    for i_s in sorted_order:
        path_runs_list.append(order[i_s])
        path_runs_list.remove(order[i_s])

    # exclude faulty runs in archive
    if esm in esms_excl:
        for run_excl in runs_excl:
            for run in path_runs_list:
                if run_excl in run:
                    path_runs_list.remove(run)

    return path_runs_list


def load_cmip(targ, esm, scen, cfg, prescribed_members=None):
    """Load ESM VAR runs from cmip archives at ETHZ.

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

    # check if scenario is correctly defined as historical + scenario
    if scen[:2] == "h-":
        _scen = scen[2:]
    else:
        raise Exception(
            "No version without historical time period is currently implemented. Crashing this function."
        )

    # find the files which fulfill the specifications
    path_runs_list = find_files_cmip(esm, targ, _scen, cfg, prescribed_members=prescribed_members)

    # exit function in case path_runs_list is empty (ie no file matches the search criterion)
    if len(path_runs_list) == 0:
        var = GVAR = lon = lat = time = None
        return var, GVAR, lon, lat, time

    # load variable on global grid and compute anomalies thereof
    var, run_nrs, refs = {}, {}, {}
    if targ == "txx":
        var_in_file = "tasmax"
    else:
        var_in_file = targ
    for run_path in path_runs_list:
        data, run = load_cmip_file(run_path, _scen, cfg)

        # cutting file to the requested period
        data = data.sel(
            time=slice(
                max([data.time.values[0], float(cfg.time["start"])]),
                min([data.time.values[-1], float(cfg.time["end"])]),
            )
        )

        # tmp saved this way for deriving anomalies later
        run_nrs[run_path] = run
        var[run] = copy.deepcopy(data[var_in_file].values)
        # checking if need as well the standard deviation for normalization
        if targ not in [
            "mrso",
            "mrsomean",
            "mrso_minmon",
            "fwixx",
            "fwixd",
            "fwisa",
            "fwils",
        ]:
            refs[run] = [
                data[var_in_file]
                .sel(time=slice(cfg.ref["start"], cfg.ref["end"]))
                .mean(dim="time")
                .values
            ]
        else:
            refs[run] = [
                data[var_in_file]
                .sel(time=slice(cfg.ref["start"], cfg.ref["end"]))
                .mean(dim="time")
                .values,
                data[var_in_file]
                .sel(time=slice(cfg.ref["start"], cfg.ref["end"]))
                .std(dim="time")
                .values,
            ]

    # preparing references
    if cfg.ref["type"] == "all":  # averaging the reference over all
        var_ref = [
            sum([refs[run_path][i] for run_path in path_runs_list])
            / len(path_runs_list)
            for i in range(refs[run])
        ]
    elif cfg.ref["type"] == "first":  # choosing only the first one
        var_ref = refs["1"]
    elif cfg.ref["type"] in [
        "individ",
        "none",
    ]:  # no need to calculate a global reference
        pass

    # computing anomalies
    for run in var.keys():
        if targ in [
            "mrso",
            "mrsomean",
            "mrso_minmon",
            "fwixx",
            "fwixd",
            "fwisa",
            "fwils",
        ]:
            # Not removing anomaly or normalizing soil moisture.
            pass
            # Not removing anomaly or normalizing soil moisture. In SMA, normalizing
            # if cfg.ref["type"] in ["all","first"]:# removing precalculated reference
            #    var[run] -= var_ref[0]
            #    var[run] /= var_ref[1]
            # elif cfg.ref["type"] == 'individ':# individual references
            #    var[run] -= refs[run][0]
            #    var[run] /= refs[run][1]
            # elif cfg.ref["type"] == 'none':# not calculating anomalies
            #    pass
            # else:
            #    raise Exception('Unprepared reference type: '+cfg.ref["type"])

        else:
            if cfg.ref["type"] in ["all", "first"]:  # removing precalculated reference
                var[run] -= var_ref[0]
            elif cfg.ref["type"] == "individ":  # individual references
                var[run] -= refs[run][0]
            elif cfg.ref["type"] == "none":  # not calculating anomalies
                pass
            else:
                raise Exception("Unprepared reference type: " + cfg.ref["type"])

    # extract time, longitude, latitude, and 3d weights from data
    time, lon, lat, wgt3d = extract_time_lon_lat_wgt3d(data)

    # compute area-weighted GVAR from global var
    GVAR = {}
    for run_path in path_runs_list:
        run = run_nrs[run_path]
        if targ in ["hfds"]:  # specific case
            masked_var = np.ma.masked_array(
                var[run], np.isnan(var[run])
            )  # account for missing values over land
            GVAR[run] = np.average(
                masked_var, axis=(1, 2), weights=wgt3d
            )  # no need to keep mask since False everywhere
        else:
            GVAR[run] = np.average(var[run], axis=(1, 2), weights=wgt3d)

    return var, GVAR, lon, lat, time


def load_cmip_file(run_path, scen, cfg):
    """Load file in ETHZ cmip archives. This function integrates several corrections

    Parameters
    ----------
    run_path : str
        path to file
    scen : str
        future scenario (e.g., "rcp85" or "ssp585")
    cfg : module
        config file containing metadata

    Returns
    -------
    data : xr.core.dataset.Dataset
        loaded dataset
    run : int
        realization index
    """

    # account for difference in naming convention in cmipx-ng archives
    if cfg.gen == 5:
        data = (
            xr.open_dataset(run_path, use_cftime=True)
            .rename({"year": "time"})
            .roll(lon=72, roll_coords=True)
        )  # rename to time for consistency with cmip6, roll so land in center
        data = data.assign_coords(
            lon=(((data.lon + 180) % 360) - 180)
        )  # assign_coords so that labels = reasonable

        # update here the time axis of the files for more robust combine_first, due to different time formats.
        data.coords["time"] = update_time_axis(data.time.values, run_path)

        run = int(
            data.attrs["source_ensemble"].split("r")[1].split("i")[0]
        )  # extract ens member

    elif cfg.gen == 6:
        if "ssp534-over" in run_path:
            # Solving two issues here :
            # Issue of ssp534-over: it starts in 2040, before it follows ssp585: cut in 2039
            # Issue of the cmip_X repository: some historicals overlap over the scenario period. Choosing here that the scenario determines values over the overlap.
            ssp534over = xr.open_dataset(run_path, use_cftime=True)
            ssp585 = xr.open_dataset(run_path.replace(scen, "ssp585"), use_cftime=True)
            hh = xr.open_dataset(run_path.replace(scen, "historical"), use_cftime=True)

            # update here the time axis of the files for more robust combine_first, due to different time formats.
            ssp534over.coords["time"] = update_time_axis(
                ssp534over.time.values, run_path
            )
            ssp585.coords["time"] = update_time_axis(
                ssp585.time.values, run_path.replace(scen, "ssp585")
            )
            hh.coords["time"] = update_time_axis(
                hh.time.values, run_path.replace(scen, "historical")
            )

            # Order of scenarios: historical (1850-2014) - ssp585 (2015-2039) - ssp534-over (2040-...)
            data = ssp534over.combine_first(
                ssp585.sel(time=slice(None, "2039")).combine_first(hh)
            )

        else:  # for every other scenario
            # Issue of the cmip_X repository: some historicals overlap over the scenario period. Choosing here that the scenario determines values over the overlap.
            ssp = xr.open_dataset(run_path, use_cftime=True)
            hh = xr.open_dataset(run_path.replace(scen, "historical"), use_cftime=True)

            # update here the time axis of the files for more robust combine_first, due to different time formats.
            ssp.coords["time"] = update_time_axis(ssp.time.values, run_path)
            hh.coords["time"] = update_time_axis(
                hh.time.values, run_path.replace(scen, "historical")
            )

            # Choosing here that the scenario determines values over the overlap.
            data = ssp.combine_first(hh)

        data = data.roll(lon=72, roll_coords=True)  # roll so land in center
        data = data.assign_coords(
            lon=(((data.lon + 180) % 360) - 180)
        )  # assign_coords so that labels = reasonable
        run = data.attrs["variant_label"]  # data.attrs["realization_index"]
    if "time" in data.dims:
        data = data.transpose("time", "lat", "lon", ...)

    return data, run


def update_time_axis(time_axis, run_path):
    """This script adapts a timeaxis inputed to return a list of *YEARS*. It helps in solving several issues:
     1. The cmip-ng archive uses only the 'np.datetime64' format. All the others formats are encountered in the cmip_X repository.
     2. Depending on files, some may not have the same format for time over the full length of the scenario.
     3. Limitation of np.datetime64 to 1678-2262 limits its use for scenarios extended up to 2300.

    Args:
    time_axis : xarray.core.dataarray.DataArray
        time axis to change. It *MUST* be loaded from a dataset with 'use_cftime=True', because of issue 2. It can be a mix of Timestamp, DatetimeGregorian and datetime64.
    run_path : str
        used only to print an error message if the time format is unknown.

    Returns:
    time_new : np.ndarray
        1d array of years ONLY
    """

    tm = []
    for val in time_axis:
        if type(val) in [
            pd._libs.tslibs.timestamps.Timestamp,
            cft._cftime.DatetimeGregorian,
            cft._cftime.DatetimeNoLeap,
            cft.Datetime360Day,
            cft.DatetimeProlepticGregorian,
        ]:
            yr, mon, day = (
                val.year,
                val.month,
                val.day,
            )  # nothing to do, can keep "val" this way
        elif type(val) in [np.datetime64]:
            val = pd.DatetimeIndex([val])
            yr, mon, day = val[0].year, val[0].month, val[0].day
        else:
            raise Exception(
                "This time format ("
                + str(type(val))
                + ") is not prepared yet, but appears in: "
                + run_path
            )

        if np.all(
            np.isnan([yr, mon, day])
        ):  # some of cmip5_X files have np.datetime64 'NaT' instead of a valid format, but values are still provided by the ESM. To exclude?
            tm.append(
                [tm[-1][0] + 1, tm[-1][1], tm[-1][2]]
            )  # adding last year+1, same month and day.
        else:
            tm.append([yr, mon, day])
    if np.any(np.diff(np.array(tm)[:, 1]) != 0.0) or np.any(
        np.diff(np.array(tm)[:, 2]) != 0.0
    ):
        raise Exception("This file is not an annual file: " + run_path)
    else:
        time_new = np.array(tm)[:, 0]
    return time_new


def test_combination_vars(list_vars, esm, scen, cfg):
    """Check that files are available for all variables in list_vars for a specific under a specific scen

    Args:
    list_vars : list of str
        list of variables to check
    esm : str
        Earth System Model (e.g., 'CanESM2' or 'CanESM5')
    scen : str
        future scenario (e.g., 'rcp85' or 'ssp585'). Must contain the historical: 'h-rcp85' or 'h-ssp85'.
    cfg : module
        config file containnig metadata

    Returns:
    common_runs : list of str
        list of the runs that are available for all the required variables
    dico_runs : dict
        dictionary with the paths for all variables
    """
    # check if scenario is correctly defined as historical + scenario
    if scen[:2] == "h-":
        _scen = scen[2:]
    else:
        raise Exception(
            "No version without historical time period is currently implemented. Crashing this function."
        )

    # looping on all variables to get their respective list of paths
    dico_paths = {}
    for targ in list_vars:
        # find the files which fulfill the specifications
        dico_paths[targ] = find_files_cmip(esm, targ, _scen, cfg)

    # deducing available runs for each
    dico_runs = {
        targ: [str.split(run, "_")[-2] for run in dico_paths[targ]]
        for targ in list_vars
    }

    #
    # checking that at least one of the runs is available for all:
    common_runs = []
    for run in dico_runs[list_vars[0]]:
        if np.all([run in dico_runs[var] for var in list_vars]):
            common_runs.append(run)

    return common_runs, dico_paths
