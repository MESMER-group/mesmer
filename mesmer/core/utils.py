import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from datatree import DataTree
from packaging.version import Version


class OptimizeWarning(UserWarning):
    pass


class LinAlgWarning(UserWarning):
    pass


def create_equal_dim_names(dim, suffixes):

    if not len(suffixes) == 2:
        raise ValueError("must provide exactly two suffixes")

    return tuple(f"{dim}{suffix}" for suffix in suffixes)


def _minimize_local_discrete(func, sequence: Iterable, **kwargs):
    """find the local minimum for a function that consumes discrete input

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Should take the elements of ``sequence``
        as input and return a float that is to be minimized.
    sequence : iterable
        An iterable with discrete values to evaluate func for.
    **kwargs : Mapping
        Keyword arguments passed to `func`.

    Returns
    -------
    element
        The element from sequence which corresponds to the local minimum.

    Raises
    ------
    ValueError : if `func` returns negative infinity for any input.

    Notes
    -----
    - The function determines the local minimum, i.e., the loop is aborted if
      `func(sequence[i-1]) >= func(sequence[i])`.
    """

    current_min = float("inf")
    # ensure it's a list because we cannot get an item from an iterable
    sequence = list(sequence)

    for i, element in enumerate(sequence):

        res = func(element, **kwargs)

        if np.isneginf(res):
            raise ValueError("`fun` returned `-inf`")
        # skip element if inf is returned - not sure about this?
        elif np.isinf(res):
            warnings.warn("`fun` returned `inf`", OptimizeWarning)

        if res < current_min:
            current_min = res
        else:
            # need to return element from the previous iteration
            sel = i - 1
            if sel == 0:
                warnings.warn("First element is local minimum.", OptimizeWarning)
            return sequence[sel]

    warnings.warn("No local minimum found, returning the last element", OptimizeWarning)

    return element


def _to_set(arg):

    if arg is None:
        arg = set()

    if isinstance(arg, str):
        arg = {arg}

    arg = set(arg)

    return arg


def _assert_annual_data(time):
    """assert time coords has annual frequency"""

    freq = xr.infer_freq(time)

    if freq is None:
        raise ValueError(
            "Annual data is required but data of unknown frequency was passed"
        )
    # pandas v2.2 and xarray v2023.11.0 changed the time freq string for year
    if not (freq.startswith("A") or freq.startswith("Y")):
        raise ValueError(
            f"Annual data is required but data with frequency {freq} was passed"
        )


def upsample_yearly_data(yearly_data, monthly_time, time_dim="time"):
    """Upsample yearly data to monthly resolution by repeating yearly values.

    Parameters
    ----------
    yearly_data : xarray.DataArray
        Yearly values to upsample.

    monthly_time: xarray.DataArray
        Monthly time used to define the time coordinates of the upsampled data.

    Returns
    -------
    upsampled_yearly_data: xarray.DataArray
        Upsampled yearly temperature values containing the yearly values for every month of the corresponding year.
    """
    _check_dataarray_form(yearly_data, "yearly_data", required_dims=time_dim)
    _check_dataarray_form(monthly_time, "monthly_time", ndim=1, required_dims=time_dim)

    if yearly_data[time_dim].size * 12 != monthly_time.size:
        raise ValueError(
            "Length of monthly time not equal to 12 times the length of yearly data."
        )

    # make sure monthly and yearly data both start at the beginning of the period
    # pandas v2.2 changed the time freq string for year
    freq = "AS" if Version(pd.__version__) < Version("2.2") else "YS"
    year = yearly_data.resample({time_dim: freq}).bfill()
    month = monthly_time.resample({time_dim: "MS"}).bfill()

    # forward fill yearly values to monthly resolution
    upsampled_yearly_data = year.reindex_like(month, method="ffill")

    # make sure the time dimension of the upsampled data is the same as the original monthly time
    upsampled_yearly_data = year.reindex_like(monthly_time, method="ffill")

    return upsampled_yearly_data


def _check_dataset_form(
    obj,
    name: str = "obj",
    *,
    required_vars: str | set[str] | None = set(),
    optional_vars: str | set[str] = set(),
    requires_other_vars: bool = False,
):
    """check if a dataset conforms to some conditions

    obj: Any
        object to check.
    name : str, default: 'obj'
        Name to use in error messages.
    required_vars, str, set of str, optional
        Variables that obj is required to contain.
    optional_vars: str, set of str, optional
        Variables that the obj may contain, only
        relevant if `requires_other_vars` is True
    requires_other_vars: bool, default: False
        obj is required to contain other variables than
        required_vars or optional_vars

    Raises
    ------
    TypeError: if obj is not a xr.Dataset
    ValueError: if any of the conditions is violated

    """

    required_vars = _to_set(required_vars)
    optional_vars = _to_set(optional_vars)

    if not isinstance(obj, xr.Dataset):
        raise TypeError(f"Expected {name} to be an xr.Dataset, got {type(obj)}")

    data_vars = set(obj.data_vars)

    missing_vars = required_vars - data_vars
    if missing_vars:
        missing_vars = ",".join(missing_vars)
        raise ValueError(f"{name} is missing the required data_vars: {missing_vars}")

    n_vars_except = len(data_vars - (required_vars | optional_vars))
    if requires_other_vars and n_vars_except == 0:

        raise ValueError(f"Expected additional variables on {name}")


def _check_dataarray_form(
    obj,
    name: str = "obj",
    *,
    ndim: int | tuple[int, ...] | None = None,
    required_dims: str | set[str] = set(),
    shape: tuple[int] | None = None,
):
    """check if a dataset conforms to some conditions

    obj: Any
        object to check.
    name : str, default: 'obj'
        Name to use in error messages.
    ndim : int | tuple of int, optional
        Number of required dimensions, can be a tuple of int if several are possible.
    required_dims: str, set of str, optional
        Names of dims that are required for obj
    shape : tuple of ints, default: None
        Required shape. Ignored if None.

    Raises
    ------
    TypeError: if obj is not a xr.DataArray
    ValueError: if any of the conditions is violated

    """

    required_dims = _to_set(required_dims)

    if not isinstance(obj, xr.DataArray):
        raise TypeError(f"Expected {name} to be an xr.DataArray, got {type(obj)}")

    ndim = (ndim,) if np.isscalar(ndim) else ndim
    if ndim is not None and obj.ndim not in ndim:
        *a, b = map(lambda x: f"{x}D", ndim)
        ndim = (a and ", ".join(a) + " or " or "") + b
        raise ValueError(f"{name} should be {ndim}, but is {obj.ndim}D")

    if required_dims - set(obj.dims):
        missing_dims = " ,".join(required_dims - set(obj.dims))
        raise ValueError(f"{name} is missing the required dims: {missing_dims}")

    if shape is not None and obj.shape != shape:
        raise ValueError(f"{name} has wrong shape - expected {shape}, got {obj.shape}")


def collapse_datatree_into_dataset(dt: DataTree, dim: str) -> xr.Dataset:
    """
    Take a ``DataTree`` and collapse it into a single ``xr.Dataset`` along dim.
    All datasets in the ``DataTree`` must have the same dimensions and each dimension must have a coordinate.

    Parameters
    ----------
    dt : DataTree
        The DataTree to collapse.
    dim : str
        The dimension to concatenate the datasets along.

    Returns
    -------
    xr.Dataset
        The collapsed dataset.

    Raises
    ------
    ValueError
        If all datasets do not have the same dimensions.
        If any dimension does not have a coordinate.
    """
    # TODO: could potentially be replaced by DataTree.merge_child_nodes in the future?
    datasets = [subtree.to_dataset() for subtree in dt.subtree if not subtree.is_empty]

    # Check if all datasets have the same dimensions
    first_dims = set(datasets[0].dims)
    if not all(set(ds.dims) == first_dims for ds in datasets):
        raise ValueError("All datasets must have the same dimensions")

    # Check that all dimensions have coordinates
    for ds in datasets:
        for ds_dim in ds.dims:
            if ds[ds_dim].coords == {}:
                raise ValueError(
                    f"Dimension '{ds_dim}' must have a coordinate/coordinates."
                )

    # Concatenate datasets along the specified dimension
    ds = xr.concat(datasets, dim=dim)
    ds = ds.assign_coords(
        {dim: [subtree.name for subtree in dt.subtree if not subtree.is_empty]}
    )

    return ds


def _datatree_to_arraydict(dt: DataTree) -> dict[str, xr.DataArray]:
    """
    Convert a DataTree to a dict of xr.DataArrays

    Parameters
    ----------
    dt : DataTree
        DataTree to convert. Note that each subtree should have a dataset with only one data variable.
        And every node needs to have a name

    Returns
    -------
    dict of xr.DataArray
        Dictionary of xr.DataArrays

    Raises
    ------
    ValueError
        If the dataset in a subtree has more than one data variable
    """
    # TODO: temporary, should not be necessary once DataTree can hold DataArrays
    predictors_dict = {}
    for subtree in dt.subtree:
        if not subtree.is_empty:
            ds = subtree.to_dataset()
            data_vars = list(ds.keys())

            if len(data_vars) > 1:
                raise ValueError(
                    f"Dataset in node '{subtree.name}' must have only one data variable."
                )

            predictors_dict[subtree.name] = ds[data_vars[0]]

    return predictors_dict


def stack_linear_regression_data(
    predictors: DataTree,
    target: DataTree,
    weights: DataTree | None,
    *,
    stacking_dims: list[str],
    collapse_dim: str = "scenario",
    stacked_dim: str = "sample",
) -> tuple[DataTree, xr.Dataset, xr.Dataset | None]:
    """
    prepares data for Linear Regression:
    1. Broadcasts predictors to target
    2. Collapses DataTrees into DataSets
    3. Stacks the DataSets along the stacking dimension(s)

    Parameters
    ----------
    predictors : DataTree
        A ``DataTree`` of ``xr.Dataset`` objects used as predictors. The ``DataTree``
        must have subtrees for each predictor each of which has to have at least one
        leaf, holding a ``xr.Dataset`` representing a scenario. The subtrees of
        different predictors must be isomorphic (i.e. have the save scenarios). The ``xr.Dataset``
        must at least contain `dim` and each ``xr.Dataset`` must only hold one data variable.
    target : DataTree
        A ``DataTree``holding the targets. Must be isomorphic to the predictor subtrees, i.e.
        have the same scenarios. Each leaf must hold a ``xr.Dataset`` which must be at least 2D
        and contain `dim`, but may also contain a dimension for ensemble members.
    weights : DataTree, default: None.
        Individual weights for each sample, must be isomorphic to target. Must at least contain
        `dim`, and must have the ensemble member dimesnion if target has it.
    stacking_dims : list[str]
        Dimension(s) to stack.
    collapse_dim : str, default: "scenario"
        Dimension along which to collapse the DataTrees, will automatically be added to the
        stacking dims.
    stacked_dim : str, default: "sample"
        Name of the stacked dimension.

    Returns
    -------
    tuple
        Tuple of the prepared predictors, target and weights, where the predictors and target are
        stacked along the stacking dimensions and the weights are stacked along the stacking dimensions
        and the ensemble member dimension.

    Notes
    -----
    Dimensions which exist along the target but are not in the stacking_dims will be excluded from the
    broadcasting of the predictors.
    """

    stacking_dims_all = stacking_dims + [collapse_dim]

    # exclude target dimensions from broadcasting which are not in the stacking_dims
    exclude_dim = set(target.leaves[0].ds.dims) - set(stacking_dims)

    # predictors need to be
    predictors_stacked = DataTree()
    for key, subtree in predictors.items():
        # 1) broadcast to target
        pred_broadcasted = subtree.broadcast_like(target, exclude=exclude_dim)
        # 2) collapsed into DataSets
        predictor_ds = collapse_datatree_into_dataset(
            pred_broadcasted, dim=collapse_dim
        )
        # 3) stacked
        predictors_stacked[key] = DataTree(
            predictor_ds.stack(
                {stacked_dim: stacking_dims_all}, create_index=False
            ).dropna(dim=stacked_dim)
        )

    # target needs to be
    # 1) collapsed into DataSet
    target_ds = collapse_datatree_into_dataset(target, dim=collapse_dim)
    # 2) stacked
    target_stacked = target_ds.stack(
        {stacked_dim: stacking_dims_all}, create_index=False
    ).dropna(dim=stacked_dim)

    # weights need to be
    if weights is not None:
        # 1) collapsed into DataSet
        weights_ds = collapse_datatree_into_dataset(weights, dim=collapse_dim)
        # 2) stacked
        weights_stacked = weights_ds.stack(
            {stacked_dim: stacking_dims_all}, create_index=False
        ).dropna(dim=stacked_dim)
    else:
        weights_stacked = None

    return predictors_stacked, target_stacked, weights_stacked
