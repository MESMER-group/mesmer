import functools
from collections.abc import Callable, Iterable
from typing import ParamSpec, TypeVar, overload

import xarray as xr
from packaging.version import Version
from xarray.core import dtypes
from xarray.core.types import CombineAttrsOptions, CompatOptions, JoinOptions

P = ParamSpec("P")
T = TypeVar("T")


if Version(xr.__version__) < Version("2025.03"):
    raise ImportError(
        f"xarray version {xr.__version__} not supported - please upgrade to v2025.03 ("
        "or later)"
    )


def _skip_empty_nodes(func):

    @functools.wraps(func)
    def _func(*args, **kwargs):
        # extract only the dataset args from args
        ds_args = [arg for arg in args if isinstance(arg, xr.Dataset)]

        # if any datasets are empty, return empty
        if any((not ds.coords and not ds.data_vars) for ds in ds_args):
            return xr.Dataset()

        # return func with right order of args
        return func(*args, **kwargs)

    return _func


def map_over_datasets(func, *args, kwargs=None):
    """
    Applies a function to every dataset in one or more DataTree objects with
    the same structure (ie.., that are isomorphic), returning new trees which
    store the results.

    adapted version of xr.map_over_datasets which skips empty nodes

    Parameters
    ----------
    func : callable
        Function to apply to datasets with signature:

        `func(*args: Dataset, **kwargs) -> Union[Dataset, tuple[Dataset, ...]]`.

        (i.e. func must accept at least one Dataset and return at least one Dataset.)
    *args : tuple, optional
        Positional arguments passed on to `func`. Any DataTree arguments will be
        converted to Dataset objects via `.dataset`.
    kwargs : dict, optional
        Optional keyword arguments passed directly to ``func``.


    See Also
    --------
    xr.map_over_datasets

    Notes
    -----
    For the discussion in xarray see https://github.com/pydata/xarray/issues/9693

    """

    return xr.map_over_datasets(_skip_empty_nodes(func), *args, kwargs=kwargs)


def _extract_single_dataarray_from_dt(
    dt: xr.DataTree, name: str = "node"
) -> xr.DataArray:
    """
    Extract a single DataArray from a DataTree node, holding a ``Dataset`` with one ``DataArray``.
    """

    if not dt.has_data:
        raise ValueError(f"{name} has no data.")

    ds = dt.to_dataset()
    var_name, *others = ds.keys()
    if others:
        o = ", ".join(map(str, others))
        raise ValueError(
            f"Node must only contain one data variable, {name} has {o} and {var_name}."
        )

    da = ds[var_name]
    return da


def collapse_datatree_into_dataset(
    dt: xr.DataTree, dim: str, **concat_kwargs
) -> xr.Dataset:
    """
    Take a ``DataTree`` and collapse **all its subtrees** into a single ``xr.Dataset`` along dim.
    Shallow wrapper around ``xr.concat``.

    All subtrees are converted to ``xr.Dataset`` objects and concatenated along the
    specified dimension. The dimension along which the datasets are concatenated will be added as a
    coordinate to the resulting dataset and the name of each subtree will be used as the coordinate
    value for this new dimension. Internally, xr.concat is used to concatenate the datasets, so
    all keyword arguments that can be passed to xr.concat can be passed to this function as well.

    Parameters
    ----------
    dt : DataTree
        The DataTree to collapse.
    dim : str
        The dimension to concatenate the datasets along.
    **concat_kwargs : dict
        Additional keyword arguments to pass to ``xr.concat``.

    Returns
    -------
    xr.Dataset
        The collapsed dataset.
    """
    # TODO: could potentially be replaced by DataTree.merge_child_nodes in the future?
    datasets = [subtree.to_dataset() for subtree in dt.subtree if not subtree.is_empty]

    # Concatenate datasets along the specified dimension
    ds = xr.concat(datasets, dim=dim, **concat_kwargs)
    dims = [subtree.name for subtree in dt.subtree if not subtree.is_empty]
    ds = ds.assign_coords({dim: dims})

    return ds


def _stack_datatree(
    dt: xr.DataTree,
    *,
    member_dim: str | None = "member",
    time_dim: str = "time",
    scenario_dim: str = "scenario",
    sample_dim: str = "sample",
) -> xr.Dataset:
    """
    stack datatree along a new dimension, named sample_dim. Each node needs a time- and
    member-dimension. The scenario dimension will be filled with the names of each node.

    Parameters
    ----------
    dt : xr.DataTree
        DataTree to stack
    member_dim : str | None, default: "member"
        Name of the member dimension already present on each dataset.
    time_dim : str, default: "time"
        Name of the time dimension already present in each dataset.
    scenario_dim : str, default: "scenario"
        Name of the scenario dimension that will used in the stacked dataset.
    sample_dim : str, default: "sample"
        Name of the stacked dimension.

    Returns
    -------
    stacked : xr.Dataset
        Dataset stacked along the sample dimension
    """

    # NOTE: we want time to be the fastest changing variable (i.e. we want to stack one
    # member after the other) for this "member" _must_ be _before_ "time"

    # otherwise the order does not matter - probably because scenario is only
    # a single entry and we concat after stacking

    if member_dim is None:
        dims = (scenario_dim, time_dim)
    else:
        dims = (scenario_dim, member_dim, time_dim)

    out = list()
    for path, (node,) in xr.group_subtrees(dt):

        if node.has_data:
            ds = node.to_dataset()

            if member_dim is not None and member_dim not in ds.dims:
                available_dims = "', '".join(ds.dims)
                msg = (
                    f"`member_dim` ('{member_dim}') not available in node '{path}' "
                    f"with available dims: '{available_dims}'. If no `member_dim` is "
                    "available, set it to `None`."
                )

                raise ValueError(msg)

            ds = ds.expand_dims({scenario_dim: [path]})
            ds = ds.stack({sample_dim: dims}, create_index=False)

            out.append(ds)

    out = xr.concat(out, dim=sample_dim)

    return out.transpose(sample_dim, ...)


@overload
def broadcast_and_stack_scenarios(
    predictors: xr.DataTree,
    target: xr.DataTree,
    weights: None = None,
    *,
    time_dim: str = "time",
    member_dim: str = "member",
    scenario_dim: str = "scenario",
    sample_dim: str = "sample",
) -> tuple[xr.Dataset, xr.Dataset, None]: ...
@overload
def broadcast_and_stack_scenarios(
    predictors: xr.DataTree,
    target: xr.DataTree,
    weights: xr.DataTree,
    *,
    time_dim: str = "time",
    member_dim: str = "member",
    scenario_dim: str = "scenario",
    sample_dim: str = "sample",
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]: ...


def broadcast_and_stack_scenarios(
    predictors: xr.DataTree,
    target: xr.DataTree,
    weights: xr.DataTree | None = None,
    *,
    time_dim: str = "time",
    member_dim: str | None = "member",
    scenario_dim: str = "scenario",
    sample_dim: str = "sample",
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
    """
    prepare predictors, target, and weights for statistical functions, i.e. converts
    several nD DataTree nodes into a single 2D Dataset with sample and target dimensions,
    where sample consists of the time, member, and scenario dimensions, and the target
    dimension is

    1. Broadcasts predictors to target
    2. Stacks the DataTrees along the sample dimension

    Parameters
    ----------
    predictors : DataTree
        A ``DataTree`` of ``xr.Dataset`` objects used as predictors. The ``DataTree``
        must have nodes for each scenario, each of which holds a Dataset where the
        predictor(s) are contained as data variables. The ``xr.Dataset`` must contain
        ``time_dim`` and at least one data variable.
    target : DataTree
        A ``DataTree`` holding the targets. Must be isomorphic to the predictor tree, i.e.
        have the same scenarios. Each leaf must hold a ``xr.Dataset`` which must contain
       ``time_dim``.
    weights : DataTree or None, default: None
        Individual weights for each sample, must be isomorphic to target.
    time_dim : str, default: "time"
        Name of the time dimension.
    member_dim : str, default: "member"
        Name of the member dimension.
    scenario_dim : str, default: "scenario"
        Name of the scenario dimension.
    sample_dim : str, default: "sample"
        Name of the sample dimension.

    Returns
    -------
    tuple of stacked predictors, target and weights
        Tuple of the prepared predictors, target and weights, where the predictors and target are
        stacked along the stacking dimensions and the weights are stacked along the stacking dimensions
        and the ensemble member dimension.

    Notes
    -----
    Dimensions which exist along the target but are not in the stacking_dims will be excluded from the
    broadcasting of the predictors.

    Example for how the predictor ``DataTree`` should look like:
    ├─ hist
    |        datavars: tas, hfds, ...
    ├─ scen1
    |        datavars: tas, hfds...
    └─ ...
    with 'hist' and 'scen1' being the scenarios, holding each a dataset with the same dimensions.
    """

    # exclude target dimensions from broadcasting which are not in the stacking_dims
    # i.e. avoid broadcasting lat/ lon
    exclude_dim = set(target.leaves[0].ds.dims) - {time_dim, member_dim}

    dims = {
        "time_dim": time_dim,
        "member_dim": member_dim,
        "scenario_dim": scenario_dim,
        "sample_dim": sample_dim,
    }

    # prepare predictors
    # 1) broadcast to target (because pred is averaged over member)

    # TODO: use DataTree method again, once available
    # pred_broadcast = pred.broadcast_like(target, exclude=exclude_dim)
    pred_broadcast = map_over_datasets(
        xr.Dataset.broadcast_like, predictors, target, kwargs={"exclude": exclude_dim}
    )

    # 2) stack
    predictors_stacked = _stack_datatree(pred_broadcast, **dims)

    # prepare target
    target_stacked = _stack_datatree(target, **dims)

    # prepare weights
    if weights is not None:
        weights_stacked = _stack_datatree(weights, **dims)
    else:
        weights_stacked = None

    return predictors_stacked, target_stacked, weights_stacked


def _datatree_wrapper(func: Callable[P, T]) -> Callable[P, T]:
    """wrapper to extend functions so DataTree can be passed

    NOTE: DataTree arguments __must__ be passed as args (positional) and not as
    kwargs

    see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
    for the typing
    """

    @functools.wraps(func)
    def _inner(*args: P.args, **kwargs: P.kwargs) -> T:

        # check to ensure there are no DataTree in kwargs. Altough this is not very
        # efficient, it has bitten me before.
        dt_kwargs = [key for key, val in kwargs.items() if isinstance(val, xr.DataTree)]
        if dt_kwargs:
            dt_kwargs_names = "', '".join(dt_kwargs)
            msg = (
                "Passed a `DataTree` as keyword argument which is not allowed."
                f" Passed `DataTree` kwargs: '{dt_kwargs_names}'"
            )
            raise TypeError(msg)

        if any(isinstance(arg, xr.DataTree) for arg in args):
            return map_over_datasets(func, *args, kwargs=kwargs)

        return func(*args, **kwargs)

    return _inner


def merge(
    objects: Iterable[xr.DataTree],
    compat: CompatOptions = "no_conflicts",
    join: JoinOptions = "outer",
    fill_value: object = dtypes.NA,
    combine_attrs: CombineAttrsOptions = "override",
):
    """
    Merge the datasets of each node of isomorphic DataTree objects together.
    Wraps `xarray.merge <https://docs.xarray.dev/en/stable/generated/xarray.merge.html>`_.

    Parameters
    ----------
    objects : Iterable of DataTree
        The DataTree objects to merge. All DataTree objects must have the same structure, i.e. be isomorphic.
    compat : {'no_conflicts', 'identical', 'equals', 'override', 'broadcast_equals'}, default: 'no_conflicts'
        String indicating how to compare variables of the same name for potential conflicts,
        for details see `xarray.merge`.
    join : {'outer', 'inner', 'left', 'right'}, default: 'outer'
        String indicating how to join the datasets of the DataTree objects, for details see
        `xarray.merge`.
    fill_value : object, default: dtypes.NA
        Value to use for missing data, for details see `xarray.merge`.
    combine_attrs : {'no_conflicts', 'identical', 'equals', 'override', 'drop'}, default: 'override'
        String indicating how to combine attributes of the datasets, for details see
        `xarray.merge`.

    Returns
    -------
    xr.DataTree
        A new DataTree object containing the merged datasets from each node of the input DataTree objects.
    """
    kwargs = {
        "compat": compat,
        "join": join,
        "fill_value": fill_value,
        "combine_attrs": combine_attrs,
    }
    return map_over_datasets(
        lambda *objs, **kwargs: xr.merge(objs, **kwargs), *objects, kwargs=kwargs
    )
