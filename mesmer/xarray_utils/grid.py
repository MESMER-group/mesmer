import pandas as pd
import xarray as xr
from packaging.version import Version


def to_unstructured(
    obj, x_dim="lon", y_dim="lat", cell_dim="cell", multiindex=False, dropna=True
):
    """stack a regular grid to an unstructured grid

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Array to convert to an unstructured grid.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.
    cell_dim : str, default: "cell"
        Name of the new dimension.
    multiindex : bool, default: False
        If the new `cell_dim` should be returned as a MultiIndex.
    dropna : bool, default: True
        Drops each 'cell' if any NA values are present.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array converted to an unstructured grid.
    """

    dims = {cell_dim: (y_dim, x_dim)}
    # dims = {cell_dim: (x_dim, y_dim)}

    obj = obj.stack(dims)

    if not multiindex:
        # there is a bug in xarray v2022.06 (Index refactor)
        if Version(xr.__version__) == Version("2022.3"):
            raise TypeError("There is a bug in xarray v2022.03. Please update xarray.")

        obj = obj.reset_index(cell_dim)

    if dropna:
        obj = obj.dropna(cell_dim)

    return obj


def from_unstructured(obj, coords_orig, x_dim="lon", y_dim="lat", cell_dim="cell"):
    """unstack an unstructured grid to a regular grid

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Array to convert to an unstructured grid.
    coords_orig : xr.Dataset | xr.DataArray
        xarray object containing the original coordinates before it was converted to the
        unstructured grid.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.
    cell_dim : str, default: "cell"
        Name of the new dimension.
    multiindex : bool, default: False
        If the new `cell_dim` should be returned as a MultiIndex.
    dropna : bool, default: True
        Drops each 'cell' if any NA values are present.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array converted to an unstructured grid.
    """

    # a MultiIndex is needed to unstack
    if not isinstance(obj.indexes.get(cell_dim), pd.MultiIndex):
        obj = obj.set_index({cell_dim: (y_dim, x_dim)})

    obj = obj.unstack(cell_dim)

    # ensure we don't loose entire rows/ columns
    obj = xr.align(obj, coords_orig, join="right")[0]

    # make sure non-dimension coords are correct
    obj = obj.assign_coords(coords_orig)

    return obj
