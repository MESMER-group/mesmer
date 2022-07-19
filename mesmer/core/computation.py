# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import pyproj


def gaspari_cohn(r):
    """
    smooth, exponentially decaying Gaspari-Cohn correlation function

    Parameters
    ----------
    r : np.ndarray
        d/L with d = geographical distance in km, L = localisation radius in km

    Returns
    -------
    y : np.ndarray
        Gaspari-Cohn correlation function value for given r

    Notes
    -----
    - Smooth exponentially decaying correlation function which mimics a Gaussian
      distribution but vanishes at r=2, i.e., 2 x the localisation radius (L)
    - based on Gaspari-Cohn 1999, QJR (as taken from Carrassi et al 2018, Wiley
      Interdiscip. Rev. Clim. Change)

    """
    r = np.abs(r)
    shape = r.shape
    # flatten the array
    r = r.ravel()

    y = np.zeros(r.shape)

    # subset the data
    sel = (r >= 0) & (r < 1)
    r_s = r[sel]
    y[sel] = (
        1 - 5 / 3 * r_s**2 + 5 / 8 * r_s**3 + 1 / 2 * r_s**4 - 1 / 4 * r_s**5
    )

    sel = (r >= 1) & (r < 2)
    r_s = r[sel]

    y[sel] = (
        4
        - 5 * r_s
        + 5 / 3 * r_s**2
        + 5 / 8 * r_s**3
        - 1 / 2 * r_s**4
        + 1 / 12 * r_s**5
        - 2 / (3 * r_s)
    )

    return y.reshape(shape)


def calc_geodist_exact(lon, lat):
    """exact great circle distance based on WSG 84

    Parameters
    ----------
    lon : array-like
        1D array of longitudes
    lat : array-like
        1D array of latitudes

    Returns
    -------
    geodist : np.array
        2D array of great circle distances.
    """

    # ensure correct shape
    lon, lat = np.asarray(lon), np.asarray(lat)
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError("lon and lat need to be 1D arrays of the same shape")

    geod = pyproj.Geod(ellps="WGS84")

    n_points = len(lon)

    geodist = np.zeros([n_points, n_points])

    # calculate only the upper right half of the triangle
    for i in range(n_points):

        # need to duplicate gridpoint (required by geod.inv)
        lt = np.tile(lat[i], n_points - (i + 1))
        ln = np.tile(lon[i], n_points - (i + 1))

        geodist[i, i + 1 :] = geod.inv(ln, lt, lon[i + 1 :], lat[i + 1 :])[2]

    # convert m to km
    geodist /= 1000
    # fill the lower left half of the triangle (in-place)
    geodist += np.transpose(geodist)

    return geodist
