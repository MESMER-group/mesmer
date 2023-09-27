# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to extract regions or time period of interest.
"""


import copy as copy
import warnings

import numpy as np


def extract_land(var, reg_dict=None, wgt=None, ls=None, threshold_land=0.25):
    """
    Extract all land grid points and area weights in regions and in land-sea mask for
    given threshold.

    Parameters
    ----------
    var : dict
        nested variable dictionary with keys

        - [esm][scen] (4d array (run, time, lat, lon) of variable)

    reg_dict : dict | None
        Deprecated. No longer has an effect.

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
        Deprecated (empty dict).
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

    if reg_dict is not None:
        warnings.warn("Passing `reg_dict` no longer has an effect.", FutureWarning)

    # determine which grid points count as land grid points
    idx_l = ls["grid_no_ANT"] > threshold_land

    # extract land points + weights from land sea mask
    ls["gp_l"] = ls["grid_no_ANT"][idx_l]
    ls["grid_l"] = copy.deepcopy(ls["grid_no_ANT"])
    ls["grid_l"][~idx_l] = 0
    # gives back a binary (boolean) mask to help with plotting
    ls["idx_grid_l"] = ls["grid_l"] > threshold_land
    # masked array (ocean masked out)
    ls["grid_l_m"] = np.ma.masked_array(
        ls["grid_l"], mask=np.logical_not(ls["idx_grid_l"])
    )
    # weights for land points: multiply weight by land fraction
    ls["wgt_gp_l"] = wgt[idx_l] * ls["gp_l"]

    # extract the land points of the variable of interest
    var_l = {}
    for esm in var.keys():
        var_l[esm] = {}
        for scen in var[esm].keys():
            # run is the first axis, followed by time
            var_l[esm][scen] = var[esm][scen][:, :, idx_l]

    return var_l, {}, ls


def extract_time_period(data, time, start, end):
    """Extract selected time period.

    Parameters
    ----------
    data : np.ndarray
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

    warnings.warn(
        "`extract_time_period` is deprecated in v0.9.0 and will be removed in a future "
        "version. Please raise an issue if you still use this function.",
        FutureWarning,
    )

    sel = (time >= start) & (time <= end)

    time = time[sel]
    data = data[:, sel, ...] if data.ndim > 1 else data[sel]

    return data, time
