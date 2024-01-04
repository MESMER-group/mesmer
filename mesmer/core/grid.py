import pandas as pd
import xarray as xr
from packaging.version import Version


def _lon_to_180(lon):

    with xr.set_options(keep_attrs=True):
        lon = ((lon + 180) % 360) - 180

    if isinstance(lon, xr.DataArray):
        lon = lon.assign_coords({lon.name: lon})

    return lon


def _lon_to_360(lon):

    with xr.set_options(keep_attrs=True):
        lon = lon % 360

    if isinstance(lon, xr.DataArray):
        lon = lon.assign_coords({lon.name: lon})

    return lon


def wrap_to_180(obj, lon_name="lon"):
    """
    wrap array with longitude to [-180..180)

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        object with longitude coordinates
    lon_name : str, default: "lon"
        name of the longitude ('lon', 'longitude', ...)

    Returns
    -------
    wrapped : Dataset
        Another dataset array wrapped around.
    """

    new_lon = _lon_to_180(obj[lon_name])

    obj = obj.assign_coords(**{lon_name: new_lon})
    obj = obj.sortby(lon_name)

    return obj


def wrap_to_360(obj, lon_name="lon"):
    """
    wrap array with longitude to [0..360)

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        object with longitude coordinates
    lon_name : str, default: "lon"
        name of the longitude ('lon', 'longitude', ...)

    Returns
    -------
    wrapped : Dataset
        Another dataset array wrapped around.
    """

    new_lon = _lon_to_360(obj[lon_name])

    obj = obj.assign_coords(**{lon_name: new_lon})
    obj = obj.sortby(lon_name)

    return obj


def stack_lat_lon(
    data,
    *,
    x_dim="lon",
    y_dim="lat",
    stack_dim="gridcell",
    multiindex=False,
    dropna=True,
):
    """Stack a regular lat-lon grid to a 1D (unstructured) grid

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Array to convert to an 1D grid.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.
    stack_dim : str, default: "gridcell"
        Name of the new dimension.
    multiindex : bool, default: False
        If the new `stack_dim` should be returned as a MultiIndex.
    dropna : bool, default: True
        Drops each 'gridcell' if any NA values are present at any point in the timeseries.

    Returns
    -------
    data : xr.Dataset | xr.DataArray
        Array converted to an 1D grid.
    """

    dims = {stack_dim: (y_dim, x_dim)}

    data = data.stack(dims)

    if not multiindex:
        # there is a bug in xarray v2022.06 (Index refactor)
        if Version(xr.__version__) == Version("2022.6"):
            raise TypeError("There is a bug in xarray v2022.06. Please update xarray.")

        data = data.reset_index(stack_dim)

    if dropna:
        data = data.dropna(stack_dim)

    return data


def unstack_lat_lon_and_align(
    data, coords_orig, *, x_dim="lon", y_dim="lat", stack_dim="gridcell"
):
    """unstack an 1D grid to a regular lat-lon grid and align with original coords

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Array with 1D grid to unstack and align.
    coords_orig : xr.Dataset | xr.DataArray
        xarray object containing the original coordinates before it was converted to the
        1D grid.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.
    stack_dim : str, default: "gridcell"
        Name of the new dimension.

    Returns
    -------
    data : xr.Dataset | xr.DataArray
        Array converted to a regular lat-lon grid.
    """

    data = unstack_lat_lon(data, x_dim=x_dim, y_dim=y_dim, stack_dim=stack_dim)

    data = align_to_coords(data, coords_orig)

    return data


def unstack_lat_lon(data, *, x_dim="lon", y_dim="lat", stack_dim="gridcell"):
    """unstack an 1D grid to a regular lat-lon grid but do not align

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Array with 1D grid to unstack and align.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.
    stack_dim : str, default: "gridcell"
        Name of the new dimension.

    Returns
    -------
    data : xr.Dataset | xr.DataArray
        Array converted to a regular lat-lon grid (unaligned).
    """

    # a MultiIndex is needed to unstack
    if not isinstance(data.indexes.get(stack_dim), pd.MultiIndex):
        data = data.set_index({stack_dim: (y_dim, x_dim)})

    return data.unstack(stack_dim)


def align_to_coords(data, coords_orig):
    """align an unstacked lat-lon grid with its original coords

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Unstacked array with lat-lon to align.
    coords_orig : xr.Dataset | xr.DataArray
        xarray object containing the original coordinates before it was converted to the
        1D grid.

    Returns
    -------
    data : xr.Dataset | xr.DataArray
        Array aligned with original grid.
    """

    # ensure we don't loose entire rows/ columns
    data = xr.align(data, coords_orig, join="right")[0]

    # make sure non-dimension coords are correct
    return data.assign_coords(coords_orig.coords)
