# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in constant files such as region information, land-sea mask, area
weights, longitude and latitude information.
"""

import copy
import os

import joblib
import numpy as np
import regionmask
from packaging.version import Version

from ..utils.regionmaskcompat import mask_3D_frac_approx
from ..utils.xrcompat import infer_interval_breaks


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
    - L_end should not exceed 10'000 by much because eventually ValueError: the input
      matrix must be positive semidefinite in train_lv())
    """

    from mesmer.core.computation import calc_geodist_exact, gaspari_cohn

    dir_aux = cfg.dir_aux
    threshold_land = cfg.threshold_land

    # geodistance for all gps for certain threshold
    geodist_name = f"geodist_landthres_{threshold_land:1.2f}.pkl"

    # gaspari-cohn correlation function phi
    phi_gc_name = "phi_gaspari-cohn_landthres_{tl:1.2f}_Lset_{L_start}-{L_interval}-{L_end}.pkl".format(
        tl=threshold_land, L_start=L_start, L_interval=L_interval, L_end=L_end
    )

    fullname_geodist = os.path.join(dir_aux, geodist_name)
    fullname_phi_gc = os.path.join(dir_aux, phi_gc_name)

    L_set = np.arange(L_start, L_end + 1, L_interval)

    if not os.path.exists(fullname_geodist):
        # create geodist matrix + save it
        print("compute geographical distance between all land points")

        # extract land gridpoints
        lon_l_vec = lon["grid"][ls["idx_grid_l"]]
        lat_l_vec = lat["grid"][ls["idx_grid_l"]]

        geodist = calc_geodist_exact(lon_l_vec, lat_l_vec)

        # create auxiliary directory if does not exist already
        os.makedirs(dir_aux, exist_ok=True)

        # save the geodist file
        joblib.dump(geodist, fullname_geodist)

    else:
        # load geodist matrix
        geodist = joblib.load(fullname_geodist)

    if not os.path.exists(fullname_phi_gc):
        print("compute Gaspari-Cohn correlation function phi")

        phi_gc = {}
        for L in L_set:
            phi_gc[L] = gaspari_cohn(geodist / L)
            print("done with L:", L)

        joblib.dump(phi_gc, fullname_phi_gc)

    else:
        phi_gc = joblib.load(fullname_phi_gc)

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
        if Version(regionmask.__version__) >= Version("0.9.0"):
            reg = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
        else:
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
    # have fraction of grid cells
    reg_dict["grids"] = mask_3D_frac_approx(reg, lon["c"], lat["c"]).values
    # not sure if needed: "binary" grid with each grid point assigned to single country
    reg_dict["grid_b"] = reg.mask(lon["c"], lat["c"]).values
    # to be used for plotting outlines (mainly useful for srex regs)
    reg_dict["full"] = reg

    # obtain a (subsampled) land-sea mask
    ls = {}
    if Version(regionmask.__version__) >= Version("0.9.0"):
        land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    else:
        land_110 = regionmask.defined_regions.natural_earth.land_110

    # gives fraction of land -> in extract_land() script decide above which land
    # fraction threshold to consider a grid point as a land grid point
    ls["grid_raw"] = np.squeeze(
        mask_3D_frac_approx(land_110, lon["c"], lat["c"]).values
    )

    # remove Antarctica
    idx_ANT = np.where(lat["c"] < -60)[0]
    ls["grid_no_ANT"] = copy.deepcopy(ls["grid_raw"])
    ls["grid_no_ANT"][idx_ANT] = 0

    # derive the weights
    lon["grid"], lat["grid"] = np.meshgrid(lon["c"], lat["c"])
    wgt = np.cos(np.deg2rad(lat["grid"]))

    # derive longitude / latitude of edges of grid cells for plotting with pcolormesh
    lon["e"], lat["e"] = infer_interval_breaks(lon["c"], lat["c"])

    return reg_dict, ls, wgt, lon, lat
