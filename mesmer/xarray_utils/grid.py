import pandas as pd
import xarray as xr
from packaging.version import Version


def stack_lat_lon(
    obj,
    *,
    x_dim="lon",
    y_dim="lat",
    stack_dim="gridcell",
    multiindex=False,
    dropna=True
):
    """Stack a regular lat-lon grid to a 1D (unstructured) grid

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
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
    obj : xr.Dataset | xr.DataArray
        Array converted to an 1D grid.
    """

    dims = {stack_dim: (y_dim, x_dim)}

    obj = obj.stack(dims)

    if not multiindex:
        # there is a bug in xarray v2022.06 (Index refactor)
        if Version(xr.__version__) == Version("2022.6"):
            raise TypeError("There is a bug in xarray v2022.06. Please update xarray.")

        obj = obj.reset_index(stack_dim)

    if dropna:
        obj = obj.dropna(stack_dim)

    return obj


def unstack_lat_lon_and_align(
    obj, coords_orig, *, x_dim="lon", y_dim="lat", stack_dim="gridcell"
):
    """unstack an 1D grid to a regular lat-lon grid and align with orignal coords

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
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
    obj : xr.Dataset | xr.DataArray
        Array converted to a regular lat-lon grid.
    """

    obj = unstack_lat_lon(obj, x_dim=x_dim, y_dim=y_dim, stack_dim=stack_dim)

    obj = align_to_coords(obj, coords_orig)

    return obj


def unstack_lat_lon(obj, *, x_dim="lon", y_dim="lat", stack_dim="gridcell"):
    """unstack an 1D grid to a regular lat-lon grid but do not align

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Array with 1D grid to unstack and align.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.
    stack_dim : str, default: "gridcell"
        Name of the new dimension.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array converted to a regular lat-lon grid (unaligned).
    """

    # a MultiIndex is needed to unstack
    if not isinstance(obj.indexes.get(stack_dim), pd.MultiIndex):
        obj = obj.set_index({stack_dim: (y_dim, x_dim)})

    return obj.unstack(stack_dim)


def align_to_coords(obj, coords_orig):
    """align an unstacked lat-lon grid with it's orignal coords

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Unstacked array with lat-lon to align.
    coords_orig : xr.Dataset | xr.DataArray
        xarray object containing the original coordinates before it was converted to the
        1D grid.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array aligned with original grid.
    """

    # ensure we don't loose entire rows/ columns
    obj = xr.align(obj, coords_orig, join="right")[0]

    # make sure non-dimension coords are correct
    return obj.assign_coords(coords_orig.coords)
