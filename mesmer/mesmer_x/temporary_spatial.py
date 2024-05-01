# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import pyproj
import xarray as xr

# from .utils import create_equal_dim_names


# TAKEN FROM CURRENT CODE OF MESMER, TO REMOVE WITH REBASE


def geodist_exact(lon, lat, equal_dim_suffixes=("_i", "_j")):
    """exact great circle distance based on WSG 84

    Parameters
    ----------
    lon : xr.DataArray, np.ndarray
        1D array of longitudes
    lat : xr.DataArray, np.ndarray
        1D array of latitudes
    equal_dim_suffixes : tuple of str, default: ("_i", "_j")
        Suffixes to add to the the name of ``dim`` for the geodist array (xr.DataArray
        cannot have two dimensions with the same name).

    Returns
    -------
    geodist : xr.DataArray, np.ndarray
        2D array of great circle distances.
    """

    # TODO: allow Dataset (e.g. using cf_xarray)
    if isinstance(lon, xr.Dataset) or isinstance(lat, xr.Dataset):
        raise TypeError("Dataset is not supported, please pass a DataArray")

    # handle numpy arrays
    if not isinstance(lon, xr.DataArray) or not isinstance(lat, xr.DataArray):
        return _geodist_exact(np.asarray(lon), np.asarray(lat))

    # TODO: allow differently named lon and lat dims?
    if lon.dims != lat.dims:
        raise AssertionError(
            f"lon and lat have different dims: {lon.dims} vs. {lat.dims}. Expected "
            "equally named dimensions from a stacked array"
        )

    geodist = _geodist_exact(lon.values, lat.values)

    (dim,) = lon.dims
    dims = create_equal_dim_names(dim, equal_dim_suffixes)

    # TODO: assign coords?
    geodist = xr.DataArray(geodist, dims=dims)

    return geodist


def _geodist_exact(lon, lat):

    # ensure correct shape
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError("lon and lat must be 1D arrays of the same shape")

    geod = pyproj.Geod(ellps="WGS84")

    n_points = lon.size

    geodist = np.zeros([n_points, n_points])

    # calculate only the upper right half of the triangle
    for i in range(n_points - 1):

        # need to duplicate gridpoint (required by geod.inv)
        lt = np.repeat(lat[i : i + 1], n_points - (i + 1))
        ln = np.repeat(lon[i : i + 1], n_points - (i + 1))

        geodist[i, i + 1 :] = geod.inv(ln, lt, lon[i + 1 :], lat[i + 1 :])[2]

    # convert m to km
    geodist /= 1000
    # fill the lower left half of the triangle (in-place)
    geodist += np.transpose(geodist)

    return geodist


def gaspari_cohn_correlation_matrices(geodist, localisation_radii):
    """Gaspari-Cohn correlation matrices for a range of localisation radii

    Parameters
    ----------
    geodist : xr.DataArray, np.ndarray
        2D array of great circle distances. Calculated from e.g. ``geodist_exact``.
    localisation_radii : iterable of float
        Localisation radii to test (in km)

    Returns
    -------
    gaspari_cohn_correlation_matrices: dict[float : :obj:`xr.DataArray`, :obj:`np.ndarray`]
        Gaspari-Cohn correlation matrix (values) for each localisation radius (keys)

    Notes
    -----
    Values in ``localisation_radii`` should not exceed 10'000 km by much because
    it can lead to correlation matrices which are not positive semidefinite.

    See Also
    --------
    gaspari_cohn, geodist_exact

    """

    out = {lr: gaspari_cohn(geodist / lr) for lr in localisation_radii}

    return out


def gaspari_cohn(r):
    """smooth, exponentially decaying Gaspari-Cohn correlation function

    Parameters
    ----------
    r : xr.DataArray, np.ndarray
        Values for which to calculate the value of the Gaspari-Cohn correlation function
        (e.g. normalised geographical distances)

    Returns
    -------
    out : xr.DataArray, , np.ndarray
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

    # NOTE: see https://github.com/MESMER-group/mesmer/issues/157

    out[sel] = (
        # fmt: off
        1 - r_sel**2 * (5 / 3 - r_sel * (5 / 8 + r_sel * (1 / 2 - 1 / 4 * r_sel)))
        # fmt: on
    )

    # compute for 1 <= r < 2
    sel = (r >= 1) & (r < 2)
    r_sel = r[sel]

    out[sel] = (
        # fmt: off
        4 - r_sel * (5 - r_sel * (5 / 3 + r_sel * (5 / 8 - r_sel * (1 / 2 - 1 / 12 * r_sel)))) - 2 / (3 * r_sel)
        # fmt: on
    )

    return out
