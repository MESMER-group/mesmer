import warnings

import numpy as np
import xarray as xr

from mesmer.datatree import (
    _datatree_wrapper,
    _unpool_scen_ens,
    map_over_datasets,
    pool_scen_ens,
)


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


@_datatree_wrapper
def weighted_mean(data, weights, /, *, dims=None):
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
        ds_dims = set(ds.dims)
        if ens_dim not in ds_dims:
            raise ValueError(f"Member dimension '{ens_dim}' not found in dataset.")
        if time_dim not in ds_dims:
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


def get_weights_density(pred_data):
    """generate inverse data-density weights

    Generate weights for the each sample, based on the inverse of the density of the
    predictors. More precisely, the density of the predictors is represented by a
    multidimensional kernel density estimate using gaussian kernels where each
    dimension is one of the predictors. Subsequently, the weights are the inverse
    of this density of the predictors. Consequently, samples in regions of this
    space with low density will have higher weights, this is, "unusual" samples
    will have more weight.

    Parameters
    ----------
    pred_data : xr.DataTree | xr.Dataset | np.array
        Predictors for the training sample. Each node must be a scenario,
        with a xarray dataset (time, member). Each predictor is a variable.

    Returns
    -------
    weights : DataTree
        Weights for the sample, based on the inverse of the density of the
        predictors, summing to 1.

    """

    def _weights(data):
        from scipy.stats import gaussian_kde

        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd which has too many assumptions
        kde_histogram = gaussian_kde(data)

        # calculating density of points over the sample
        density = kde_histogram.pdf(x=data)

        # preparing the output
        return (1 / density) / np.sum(1 / density)

    def _weights_ds(ds):

        if len(ds.dims) > 1:
            msg = f"Can only handle 1D predictors, but pred_data has {len(ds.dims)}D"
            raise ValueError(msg)

        array_pred = ds.to_array("predictor")

        (non_predictor_dim,) = set(array_pred.dims) - {"predictor"}

        weights = xr.apply_ufunc(
            _weights,
            array_pred,
            input_core_dims=[["predictor", non_predictor_dim]],
            output_core_dims=[[non_predictor_dim]],
        )

        return weights

    if isinstance(pred_data, xr.DataTree):

        # reshaping data into array
        # need an array where each predictor is a column in a np.array
        # and all samples of that predictor is in one line
        pred_stacked = pool_scen_ens(pred_data)

        weights_stacked = _weights_ds(pred_stacked)
        weights_stacked.name = "weights"

        return _unpool_scen_ens(weights_stacked)

    elif isinstance(pred_data, xr.Dataset):

        return _weights_ds(pred_data).to_dataset(name="weights")

    elif isinstance(pred_data, np.ndarray):
        array_pred = pred_data
        return _weights(array_pred)


def _weighted_median(data, weights):
    """
    Parameters
    ----------
      data : numpy.array
        Data to calculate the median from.
    weights:  numpy.array
        Weights to apply

    Returns
    -------
    weighted_median

    References
    ----------

    Adapted form

    https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
    @author Jack Peterson (jack@tinybike.net)
    """

    weights = weights[~np.isnan(data)]
    data = data[~np.isnan(data)]

    indexer_array = np.argsort(data)
    s_data = data[indexer_array]
    s_weights = weights[indexer_array]

    midpoint = 0.5 * np.sum(s_weights)

    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if any(cs_weights == midpoint):
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median
