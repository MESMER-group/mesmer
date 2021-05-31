# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to extract regions or time period of interest.
"""


import copy as copy

import numpy as np


def extract_land(var, reg_dict, wgt, ls, threshold_land=0.25):
    """
    Extract all land grid points and area weights in regions and in land-sea mask for
    given threshold.

    Parameters
    ----------
    var : dict
        nested variable dictionary with keys

        - [esm][scen] (4d array (run, time, lat, lon) of variable)
    reg_dict : dict
        region dictionary with keys

        - ["type"] (region type)
        - ["abbrevs"] (abbreviations for regions)
        - ["names"] (full names of regions)
        - ["grids"] (3d array (regions, lat, lon) of subsampled region fraction)
        - ["grid_b"] (2d array (lat, lon) of regions with each grid point being assigned
          to a single region ("binary" grid))
        - ["full"] (full Region object (for plotting region outlines))
    wgt : np.ndarray
        2d array (lat, lon) of weights to be used for area weighted means
    ls : dict
        land sea dictionary with keys

        - ["grid_raw"] (2d array (lat, lon) of subsampled land fraction)
        - ["grid_no_ANT"] (grid_raw with Antarctica removed)
    threshold_land : float, default=0.25
        threshold above which land fraction to consider a grid point as a land grid
        point

    Returns
    -------
    var_l : dict
        nested variable at land grid points dictionary with keys

        - [esm] (3d array (run, time, gp_l) of variable at land grid points)
    reg_dict : dict
        region dictionary with added keys

        - ["gps_l"] (2d array (region, gp_l) of region fraction at land grid points)
        - ["wgt_gps_l"] (2d array (region, gp_l) of area weights for each region on land)
        - ["gp_b_l"] (1d array of region index at land grid points with each grid point
          being assigned to a single region)
    ls : dict
        land sea dictionary with added keys

        - ["gp_l"] (1d array of fraction of land at land grid points)
        - ["grid_l"] (2d array (lat, lon) of fraction of land at land grid points)
        - ["idx_grid_l"] (2d boolean array (lat, lon) with land grid points = True for
          plotting on map)
        - ["grid_l_m"] (2d masked array (lat, lon) with ocean masked out for plotting on
          map)
        - ["wgt_gp_l"] (1d array of land area weights, i.e., area weight * land
          fraction)

    """

    # determine which grid points count as land grid points
    idx_l = ls["grid_no_ANT"] > threshold_land

    # extract land points + weights from land sea mask
    ls["gp_l"] = ls["grid_no_ANT"][idx_l]
    ls["grid_l"] = copy.deepcopy(ls["grid_no_ANT"])
    ls["grid_l"][~idx_l] = 0
    ls["idx_grid_l"] = (
        ls["grid_l"] > threshold_land
    )  # gives back a binary (boolean) mask to help with plotting
    ls["grid_l_m"] = np.ma.masked_array(
        ls["grid_l"], mask=np.logical_not(ls["idx_grid_l"])
    )  # masked array (ocean masked out)
    ls["wgt_gp_l"] = (
        wgt[idx_l] * ls["gp_l"]
    )  # weights for land points: multiply weight by land fraction

    # extract the land points + weights from the region grids
    reg_dict["gps_l"] = reg_dict["grids"][:, idx_l]  # country is the first axis
    reg_dict["wgt_gps_l"] = (
        wgt[idx_l] * reg_dict["gps_l"]
    )  # weights for regions (1st axis): region fraction * area weights
    if reg_dict["type"] == "srex" or reg_dict["type"] == "ar6.land":
        reg_dict["wgt_gps_l"] = (
            reg_dict["wgt_gps_l"] * ls["gp_l"]
        )  # * land fraction to account for coastal cells because SREX / ar6.land regions include ocean
    reg_dict["gp_b_l"] = reg_dict["grid_b"][
        idx_l
    ]  # not sure if needed; extracts land from the "binary" mask

    # extract the land points of the variable of interest
    var_l = {}
    for esm in var.keys():
        var_l[esm] = {}
        for scen in var[esm].keys():
            var_l[esm][scen] = var[esm][scen][
                :, :, idx_l
            ]  # run is the first axis, followed by time

    return var_l, reg_dict, ls


def extract_time_period(var, time, start, end):
    """Extract selected time period.

    Parameters
    ----------
    var : np.ndarray
        variable in 1-4d array

        - (time);
        - (run, time);
        - (run, time, gp_l);
        - (run, time, lat, lon)
    time : np.ndarray
        1d array of years
    start : str or int
        first year included in extracted time period
    end : str or int
        last year included in extracted time period

    Returns
    -------
    var_tp : np.ndarray
        variable 1-3d array

        - (time);
        - (time, gp_l);
        - (time, lat, lon);
    time_tp : np.ndarray
        1d array of years of extracted time period

    """

    # find index of start and end of time period
    idx_start = np.where(time == int(start))[0][0]
    idx_end = np.where(time == int(end))[0][0] + 1  # to include the end year

    # extract time period from variable dictionary
    if len(var.shape) > 1:
        var_tp = var[:, idx_start:idx_end]
    else:
        var_tp = var[idx_start:idx_end]

    # extract time period from time vector
    time_tp = time[idx_start:idx_end]

    return var_tp, time_tp
