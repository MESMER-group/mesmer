# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import pyproj
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances

from mesmer._core.utils import _create_equal_dim_names


def geodist_exact(
    lon: xr.DataArray | np.ndarray,
    lat: xr.DataArray | np.ndarray,
    *,
    equal_dim_suffixes: tuple[str, str] = ("_i", "_j"),
):
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
    dims = _create_equal_dim_names(dim, equal_dim_suffixes)

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

        geodist[i, i + 1 :] = geod.inv(
            ln.squeeze(),
            lt.squeeze(),
            lon[i + 1 :].squeeze(),
            lat[i + 1 :].squeeze(),
        )[2]

    # convert m to km
    geodist /= 1000
    # fill the lower left half of the triangle (in-place)
    geodist += np.transpose(geodist)

    return geodist


def closest_neighbors(lon: xr.DataArray, lat: xr.DataArray, n_closest: int):
    """n closest neighbors based on spherical distance

    Given an array of (lat, lon) coordinates, this function computes the pairwise
    distance between all coordinate locations in the array and then returns an array
    of indices that gives the indices of the n_closest locations

    Parameters
    ----------
    coordinate_path: str or Path
        String or pahtlib.Path object pointing to an .npy file that contains an array of shape (n_locations, 2)
        tghat contains (lat, lon) coordinates for all n_locations
    n_closest: int
        number of closest locations to compute

    Returns
    -------
    closest_neighbors : xr.DataArray
        Contains the indices of the n_closest coordinates for each location. That is,
        for the location with index i, i.e. coords[i], the coordinates of the n_closest
        locations are given by coords[selected_loc_[i]]

    """

    if lon.dims != lat.dims:
        raise ValueError(
            f"lon and lat have different dims: {lon.dims} vs. {lat.dims}. Expected "
            "equally named dimensions from a stacked array"
        )

    if lon.ndim != 1:
        raise ValueError("Expected 1D data - i.e. stacked longitude and latitude")

    coords = np.column_stack([lat.values, lon.values])
    coords = np.deg2rad(coords)

    # multiply by 6371000 / 1000 to convert to km
    dist = haversine_distances(coords, coords)
    selected_loc = np.argsort(dist, axis=1)[:, :n_closest]

    (dim,) = lon.dims

    closest_neighbors = xr.DataArray(
        selected_loc,
        dims=(dim, "closest_gridcells"),
        coords={
            lon.name: lon,
            lat.name: lat,
            "closest_gridcells": np.arange(n_closest),
        },
    )

    return closest_neighbors
