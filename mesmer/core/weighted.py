import warnings

import numpy as np
import xarray as xr

from mesmer.core._datatreecompat import map_over_datasets
from mesmer.core.datatree import _datatree_wrapper


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


@_datatree_wrapper
def lat_weights(data, y_dim="lat"):
    """area weights based on the cosine of the latitude

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset | xr.DataTree | array_like
        Latitude coordinates.
    y_dim : str, default: "lat"
        Name of the y dimension to retrieve the coordinates from.

    Returns
    -------
    weights : xr.DataArray | xr.Dataset | xr.DataTree | np.ndarray
        Cosine weights of ``lat_coords``. If a Dataset or DataTree is passed the result
        is stored in a DataArray named ``weights``.
    """

    is_dataset = isinstance(data, xr.Dataset)
    is_dataarray_or_set = is_dataset or isinstance(data, xr.DataArray)

    if is_dataarray_or_set:
        data = data[y_dim]

    if np.ndim(data) > 1:
        warnings.warn("cos(lat) is not a good approximation for non-regular grids")

    if np.max(np.abs(data)) > 90:
        raise ValueError("`lat_coords` must be between -90 and 90 (inclusive)")

    weights = np.cos(np.deg2rad(data))

    if not is_dataarray_or_set:
        return weights

    weights.name = "weights"

    # map_over_datasets requires a Dataset
    if is_dataset:
        weights = weights.to_dataset()

    return weights


def weighted_mean(data, weights, *, dims=None):
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

    # map_over_datasets does not work with kwargs, so need separate funct to be able to pass
    # weights as kw
    return _weighted_mean(data, weights, dims)


@_datatree_wrapper
def _weighted_mean(data, weights, /, dims=None):
    if isinstance(dims, str):
        dims = [dims]

    if isinstance(weights, xr.Dataset):
        if "weights" not in weights:
            raise ValueError("weights does not contain a variable named weights")
        weights = weights.weights

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
    data : xr.DataArray | xr.Dataset | xr.DataTree
        Array reduce to the global mean.
    weights : xr.DataArray | xr.Dataset | xr.DataTree | array_like, optional
        Array containing the area of each grid cell (or a measure proportional to
        the grid cell area). If not given will compute it from the cosine of the
        latitudes using ``lat_weights``.
    x_dim : str, default: "lon"
        Name of the x-dimension.
    y_dim : str, default: "lat"
        Name of the y-dimension.

    Returns
    -------
    obj : xr.DataTree |  xr.Dataset | xr.DataArray
        Array converted to an unstructured grid.


    See also
    --------
    lat_weights
    """

    return _global_mean(data, weights, x_dim=x_dim, y_dim=y_dim)


@_datatree_wrapper
def _global_mean(data, weights, /, *, x_dim, y_dim):

    if weights is None:
        weights = lat_weights(data, y_dim)

    return weighted_mean(data, weights, dims=(x_dim, y_dim))


def equal_scenario_weights_from_datatree(
    dt: xr.DataTree, ens_dim: str = "member", time_dim: str = "time"
) -> xr.DataTree:
    """
    Create a DataTree isomorphic to ``dt``, holding the weights for each scenario to weight the ensemble members of each
    scenario such that each scenario contributes equally to some fitting procedure.
    The weight of each member = 1 / number of members in the scenario, so weights = 1 / ds[ens_dim].size.

    Thus, if all scenarios have the same number of members, all weights will be equal.
    If one scenario has more members than the others, its weights will be smaller.
    Weights are always along the time and ens dim, if there are more dimensions in a dataset, they will be dropped.

    Parameters:
    -----------
    dt : DataTree
        DataTree holding the ``xr.Datasets`` for which the weights should be created. Each dataset must have at least
        ens_dim and time_dim as dimensions, but can have more dimensions.
    ens_dim : str
        Name of the dimension along which the weights should be created. Default is "member".
    time_dim : str
        Name of the time dimension, will be filled with equal values for each ensemble member. Default is "time".

    Returns:
    --------
    DataTree
        DataTree holding the weights for each scenario isomorphic to dt, where each dataset has dimensions (time_dim, ens_dim).

    Example:
    --------
    >>> dt = DataTree()
    >>> dt["ssp119"] = DataTree(xr.Dataset({"tas": xr.DataArray(np.ones((20, 3)), dims=("time", "member"))}))
    >>> dt["ssp585"] = DataTree(xr.Dataset({"tas": xr.DataArray(np.ones((20, 2)), dims=("time", "member"))}))
    >>> weights = equal_scenario_weights_from_datatree(dt)
    >>> weights
    DataTree('None', parent=None)
       ├── DataTree('ssp119')
       │       Dimensions:  (time: 20, member: 3)
       │       Coordinates:
       │         * time     (time) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
       │         * member   (member) int64 0 1 2
       │       Data variables:
       │           weights  (time, member) float64 0.3333 0.3333 0.3333 ... 0.3333 0.3333
       └── DataTree('ssp585')
               Dimensions:  (time: 20, member: 2)
               Coordinates:
               * time     (time) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
               * member   (member) int64 0 1
               Data variables:
                   weights  (time, member) float64 0.5 0.5 0.5 0.5 0.5 ... 0.5 0.5 0.5 0.5 0.5

    """
    if dt.depth != 1:
        raise ValueError(f"DataTree must have a depth of 1, not {dt.depth}.")

    def _create_weights(ds: xr.Dataset) -> xr.Dataset:
        dims = set(ds.dims)
        if ens_dim not in dims:
            raise ValueError(f"Member dimension '{ens_dim}' not found in dataset.")
        if time_dim not in dims:
            raise ValueError(f"Time dimension '{time_dim}' not found in dataset.")

        name, *others = ds.data_vars
        if others:
            raise ValueError("Dataset must only contain one data variable.")

        # create weights
        dims = [time_dim, ens_dim]
        shape = [ds[time_dim].size, ds[ens_dim].size]

        data = np.full(shape, fill_value=1 / ds[ens_dim].size)

        weights = xr.DataArray(data, dims=dims)

        # add back coords if they were there on ds
        if ds[time_dim].coords:
            weights = weights.assign_coords(ds[time_dim].coords)
        if ds[ens_dim].coords:
            weights = weights.assign_coords(ds[ens_dim].coords)

        return xr.Dataset({"weights": weights})

    weights = map_over_datasets(_create_weights, dt)

    return weights
