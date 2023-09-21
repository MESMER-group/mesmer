# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import pyproj
import xarray as xr


def gaspari_cohn(r):
    """smooth, exponentially decaying Gaspari-Cohn correlation function

    Parameters
    ----------
    r : xr.DataArray, np.array
        Values for which to calculate the value of the Gaspari-Cohn correlation function
        (e.g. normalised geographical distances)

    Returns
    -------
    out : xr.DataArray
        Gaspari-Cohn correlation function

    Notes
    -----
    - Smooth exponentially decaying correlation function which mimics a Gaussian
      distribution but vanishes at r = 2, i.e., 2 x the localisation radius (L)

    - based on Gaspari-Cohn 1999 [1]_ (as taken from Carrassi et al., 2018 [2]_)

    - r = d / L, with d = geographical distance in km, L = localisation radius in km

    .. [1] Gaspari, G. and Cohn, S.E. (1999), Construction of correlation functions in
       two and three dimensions. Q.J.R. Meteorol. Soc., 125: 723-757.
       https://doi.org/10.1002/qj.49712555417

    .. [2] Carrassi, A, Bocquet, M, Bertino, L, Evensen, G. Data assimilation in the
       geosciences: An overview of methods, issues, and perspectives. WIREs Clim Change.
       2018; 9:e535. https://doi.org/10.1002/wcc.535

    """

    if isinstance(r, xr.Dataset):
        raise TypeError("Dataset is not supported, please pass a DataArray")

    # make it work for numpy arrays
    if not isinstance(r, xr.DataArray):
        return _gaspari_cohn_np(r)

    out = _gaspari_cohn_np(r.values)

    out = xr.DataArray(out, dims=r.dims, coords=r.coords, attrs=r.attrs)

    return out


def _gaspari_cohn_np(r):

    r = np.abs(r)

    out = np.zeros(r.shape)

    # compute for 0 <= r < 1
    sel = (r >= 0) & (r < 1)
    r_sel = r[sel]

    out[sel] = (
        1
        - 5 / 3 * r_sel**2
        + 5 / 8 * r_sel**3
        + 1 / 2 * r_sel**4
        - 1 / 4 * r_sel**5
    )

    # compute for 1 <= r < 2
    sel = (r >= 1) & (r < 2)
    r_sel = r[sel]

    out[sel] = (
        4
        - 5 * r_sel
        + 5 / 3 * r_sel**2
        + 5 / 8 * r_sel**3
        - 1 / 2 * r_sel**4
        + 1 / 12 * r_sel**5
        - 2 / (3 * r_sel)
    )

    return out


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
