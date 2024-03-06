import warnings

import numpy as np
import xarray as xr


def _weighted_if_dim(obj, weights, dims):

    # xarray applies weighted to all data_vars - even if they do not have the
    # corresponding dimensions - we don't want that
    # https://github.com/pydata/xarray/issues/7027

    def _weighted_mean(da):
        if dims is None or all(dim in da.dims for dim in dims):
            return da.weighted(weights).mean(dims, keep_attrs=True)
        return da

    if isinstance(obj, xr.Dataset):
        return obj.map(_weighted_mean, keep_attrs=True)

    return obj.weighted(weights).mean(dims, keep_attrs=True)


def lat_weights(lat_coords):
    """area weights based on the cosine of the latitude

    Parameters
    ----------
    lat_coords : xr.DataArray
        Latitude coordinates.

    Returns
    -------
    weights : xr.DataArray
        Cosine weights of ``lat_coords``.

    """

    if np.ndim(lat_coords) > 1:
        warnings.warn("cos(lat) is not a good approximation for non-regular grids")

    if np.max(np.abs(lat_coords)) > 90:
        raise ValueError("`lat_coords` must be between -90 and 90 (inclusive)")

    weights = np.cos(np.deg2rad(lat_coords))

    return weights


def weighted_mean(data, weights, dims=None):
    """weighted mean - convenience function which ignores data_vars missing dims

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Array reduce to the global mean.
    weights : xr.DataArray
        DataArray containing the area of each grid cell (or a measure proportional to
        the grid cell area).
    dims : Hashable or Iterable of Hashable, optional
        Dimension(s) over which to apply the weighted ``mean``.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array converted to an unstructured grid.

    """

    if isinstance(dims, str):
        dims = [dims]

    # ensure grids are equal
    try:
        xr.align(data, weights, join="exact")
    except ValueError:
        raise ValueError("`data` and `weights` don't exactly align.")

    return _weighted_if_dim(data, weights, dims)


def global_mean(data, weights=None, x_dim="lon", y_dim="lat"):
    """calculate global weighted mean

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Array reduce to the global mean.
    weights : xr.DataArray, optional
        DataArray containing the area of each grid cell (or a measure proportional to
        the grid cell area). If not given will compute it from the cosine of the
        latitudes.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array converted to an unstructured grid.

    """

    if weights is None:
        weights = lat_weights(data[y_dim])

    return weighted_mean(data, weights, [x_dim, y_dim])
