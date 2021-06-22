# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in constant files such as region information, land-sea mask, area
weights, longitude and latitude information.
"""

import copy as copy
import os

import geopy.distance  # https://geopy.readthedocs.io/en/latest/ # give coord as (lat,lon)
import joblib
import numpy as np
import regionmask as regionmask

from ..utils.regionmaskcompat import mask_percentage
from ..utils.xrcompat import infer_interval_breaks


def gaspari_cohn(r):
    """
    Computes the smooth, exponentially decaying Gaspari-Cohn correlation function for a
    given r.

    Parameters
    ----------
    r : float
        d/L with d = geographical distance in km, L = localisation radius in km

    Returns
    -------
    y : np.ndarray
        Gaspari-Cohn correlation function value for given r

    Notes
    -----
    - Smooth exponentially decaying correlation function which mimics a Gaussian
      distribution but vanishes at r=2, i.e., 2x the localisation radius (L)
    - based on Gaspari-Cohn 1999, QJR (as taken from Carrassi et al 2018, Wiley
      Interdiscip. Rev. Clim. Change)

    """
    r = np.abs(r)

    if r >= 0 and r < 1:
        y = 1 - 5 / 3 * r ** 2 + 5 / 8 * r ** 3 + 1 / 2 * r ** 4 - 1 / 4 * r ** 5
    if r >= 1 and r < 2:
        y = (
            4
            - 5 * r
            + 5 / 3 * r ** 2
            + 5 / 8 * r ** 3
            - 1 / 2 * r ** 4
            + 1 / 12 * r ** 5
            - 2 / (3 * r)
        )
    if r >= 2:
        y = 0

    return y


def load_phi_gc(lon, lat, ls, cfg, L_start=1500, L_end=10000, L_interval=250):
    """
    Loads or creates (if not available yet) distance matrix and Gaspari-Cohn correlation
    matrix.

    Parameters
    ----------
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
        - ["e"] (1d array with longitudes at edges of grid cells)
        - ["grid"] (2d array (lat,lon) of longitudes)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)
        - ["e"] (1d array with latitudes at edges of grid cells)
        - ["grid"] (2d array (lat,lon) of latitudes)
    ls : dict
        land-sea dictionary with keys

        - ["grid_raw"] (2d array (lat,lon) of subsampled land fraction)
        - ["grid_no_ANT"] (grid_raw with Antarctica removed)
        - ["gp_l"] (1d array of fraction of land at land grid points)
        - ["grid_l"] (2d array (lat,lon) of fraction of land at land grid points)
        - ["idx_grid_l"] (2d boolean array (lat,lon) with land grid points = True for
          plotting on map)
        - ["grid_l_m"] (2d masked array (lat,lon) with ocean masked out for plotting on
          map)
        - ["wgt_gp_l"] (1d array of land area weights, i.e.,
          area weight * land fraction)
    cfg : module
        config file containing metadata
    L_start : int, optional
        smallest localisation radius which is tested
    L_end : int, optional
        largest localisation radius which is tested
    L_interval : int, optional
        spacing interval between tested localisation radii

    Returns
    -------
    phi_gc : np.ndarray
        2d array (gp, gp) of Gaspari-Cohn correlation matrix for grid points used for
        covariance localisation

    Notes
    -----
    - If no complete number of L_intervals fits between L_start and L_end, L_intervals
      are repeated until the closest possible L value below L_end is reached.
    - L_end should not exceed 10000 by much because eventually ValueError: the input matrix must be positive semidefinite in train_lv())
    """

    dir_aux = cfg.dir_aux
    threshold_land = cfg.threshold_land

    L_set = np.arange(L_start, L_end + 1, L_interval)

    # geodistance for all gps for certain threshold
    geodist_name = "geodist_landthres_{tl:1.2f}.pkl".format(tl=threshold_land)
    if not os.path.exists(dir_aux + geodist_name):
        # create geodist matrix + save it
        print("compute geographical distance between all land points")
        nr_gp_l = ls["idx_grid_l"].sum()
        lon_l_vec = lon["grid"][ls["idx_grid_l"]]
        lat_l_vec = lat["grid"][ls["idx_grid_l"]]
        geodist = np.zeros([nr_gp_l, nr_gp_l])

        # could be sped up by only computing upper or lower half of matrix
        # since only needs to be done 1x for every land threshold, not implemented
        for i in np.arange(nr_gp_l):
            for j in np.arange(nr_gp_l):
                geodist[i, j] = geopy.distance.distance(
                    (lat_l_vec[i], lon_l_vec[i]), (lat_l_vec[j], lon_l_vec[j])
                ).km

            if i % 200 == 0:
                print("done with gp", i)

        # create auxiliary directory if does not exist already (else leave directory unaltered)
        os.makedirs(dir_aux, exist_ok=True)

        # save the geodist file
        joblib.dump(geodist, dir_aux + geodist_name)

    else:
        # load geodist matrix
        geodist = joblib.load(dir_aux + geodist_name)

    # gaspari-cohn correlation function phi
    phi_gc_name = "phi_gaspari-cohn_landthres_{tl:1.2f}_Lset_{L_start}-{L_interval}-{L_end}.pkl".format(
        tl=threshold_land, L_start=L_start, L_interval=L_interval, L_end=L_end
    )

    if not os.path.exists(dir_aux + phi_gc_name):
        print(
            "compute Gaspari-Cohn correlation function phi for a number of localization radii"
        )

        phi_gc = {}
        for L in L_set:
            phi_gc[L] = np.zeros(geodist.shape)
            for i in np.arange(geodist.shape[0]):
                for j in np.arange(geodist.shape[1]):
                    phi_gc[L][i, j] = gaspari_cohn(geodist[i, j] / L)
            print("done with L:", L)

        joblib.dump(phi_gc, dir_aux + phi_gc_name)

    else:
        phi_gc = joblib.load(dir_aux + phi_gc_name)

    return phi_gc


def load_regs_ls_wgt_lon_lat(reg_type, lon, lat):
    """Load constant files.

    Parameters
    ----------
    reg_type : str
        region type ("ar6.land", "countries", "srex")
    lon : dict
        longitude dictionary with key

        - ["c"] (1d array with longitudes at center of grid cell)
    lat : dict
        latitude dictionary with key

        - ["c"] (1d array with latitudes at center of grid cell)

    Returns
    -------
    reg_dict : dict
        region dictionary with keys

        - ["type"] (region type)
        - ["abbrevs"] (abbreviations for regions)
        - ["names"] (full names of regions)
        - ["grids"] (3d array (region, lat, lon) of subsampled region fraction)
        - ["grid_b"] (2d array (lat, lon) of regions with each grid point being assigned
          to a single region ("binary" grid))
        - ["full"] (full Region object (for plotting region outlines))
    ls : dict
        land-sea dictionary with keys

        - ["grid_raw"] (2d array (lat, lon) of subsampled land fraction)
        - ["grid_no_ANT"] (grid_raw with Antarctica removed)
    wgt : np.ndarray
        2d array (lat,lon) of weights to be used for area weighted means
    lon : dict
        longitude dictionary with added keys

        - ["e"] (1d array with longitudes at edges of grid cells)
        - ["grid"] (2d array (lat,lon) of longitudes)
    lat : dict
        latitude dictionary with added keys

        - ["e"] (1d array with latitudes at edges of grid cells)
        - ["grid"] (2d array (lat,lon) of latitudes)

    Notes
    -----
    - If additional region types are added in this function,
      mesmer.utils.select.extract_land() needs to be adapted too

    """

    # choose the Regions object depending on the region type
    if reg_type == "countries":
        reg = regionmask.defined_regions.natural_earth.countries_110
    elif reg_type == "srex":
        reg = regionmask.defined_regions.srex
    elif reg_type == "ar6.land":
        reg = regionmask.defined_regions.ar6.land

    # extract all the desired information from the Regions object
    reg_dict = {}
    reg_dict["type"] = reg_type
    reg_dict["abbrevs"] = reg.abbrevs
    reg_dict["names"] = reg.names
    reg_dict["grids"] = mask_percentage(
        reg, lon["c"], lat["c"]
    ).values  # have fraction of grid cells
    reg_dict["grid_b"] = reg.mask(
        lon["c"], lat["c"]
    ).values  # not sure yet if needed: "binary" grid with each grid point assigned to single country
    reg_dict[
        "full"
    ] = reg  # to be used for plotting outlines (mainly useful for srex regs)

    # obtain a (subsampled) land-sea mask
    ls = {}
    ls["grid_raw"] = np.squeeze(
        mask_percentage(
            regionmask.defined_regions.natural_earth.land_110, lon["c"], lat["c"]
        ).values
    )
    # gives fraction of land -> in extract_land() script decide above which land fraction threshold to consider a grid point as a land grid point

    # remove Antarctica
    idx_ANT = np.where(lat["c"] < -60)[0]
    ls["grid_no_ANT"] = copy.deepcopy(ls["grid_raw"])
    ls["grid_no_ANT"][idx_ANT] = 0  #

    # derive the weights
    lon["grid"], lat["grid"] = np.meshgrid(lon["c"], lat["c"])
    wgt = np.cos(np.deg2rad(lat["grid"]))

    # derive longitude / latitude of edges of grid cells for plotting with pcolormesh
    lon["e"], lat["e"] = infer_interval_breaks(lon["c"], lat["c"])

    return reg_dict, ls, wgt, lon, lat
