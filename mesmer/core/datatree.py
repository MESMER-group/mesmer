from typing import overload

import xarray as xr

from mesmer.core._datatreecompat import map_over_datasets


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
        others = ", ".join(map(str, others))
        raise ValueError(
            f"Node must only contain one data variable, {name} has {others} and {var_name}."
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


def stack_datatree(dt: xr.DataTree) -> xr.Dataset:

    # TODO: allow passing indexes (but see below)

    out = list()
    for path, (node,) in xr.group_subtrees(dt):

        if node.has_data:
            ds = node.to_dataset()
            ds = ds.expand_dims(scenario=[path])

            # NOTE: we want time to be the fastest changing variable (i.e. we
            # want to stack one member after the other) for this
            #  "member" must be _before_ "time"

            # the first three are equivalent - most likely because scenario is only
            # a single entry and we concat after stacking (?)
            ds = ds.stack(sample=("scenario", "member", "time"), create_index=False)
            # ds = ds.stack(sample=("member", "time", "scenario"), create_index=False)
            # ds = ds.stack(sample=("member", "scenario", "time"), create_index=False)

            # these 3 do not work and lead to several
            # ds = ds.stack(sample=("time", "member", "scenario"), create_index=False)
            # ds = ds.stack(sample=("time", "scenario", "member"), create_index=False)
            # ds = ds.stack(sample=("scenario", "time", "member"), create_index=False)

            out.append(ds)

    out = xr.concat(out, dim="sample")

    return out


@overload
def stack_datatrees_for_linear_regression(
    predictors: xr.DataTree,
    target: xr.DataTree,
    weights: None = None,
    *,
    stacking_dims: list[str],
    collapse_dim: str = "scenario",
    stacked_dim: str = "sample",
) -> tuple[xr.DataTree, xr.Dataset, None]: ...
@overload
def stack_datatrees_for_linear_regression(
    predictors: xr.DataTree,
    target: xr.DataTree,
    weights: xr.DataTree,
    *,
    stacking_dims: list[str],
    collapse_dim: str = "scenario",
    stacked_dim: str = "sample",
) -> tuple[xr.DataTree, xr.Dataset, xr.Dataset]: ...


def stack_datatrees_for_linear_regression(
    predictors: xr.DataTree,
    target: xr.DataTree,
    weights: xr.DataTree | None = None,
    *,
    stacking_dims: list[str],
    collapse_dim: str = "scenario",
    stacked_dim: str = "sample",
) -> tuple[xr.DataTree, xr.Dataset, xr.Dataset | None]:
    """
    prepares data for Linear Regression:
    1. Broadcasts predictors to target
    2. Collapses DataTrees into DataSets
    3. Stacks the Datasets along the stacking dimension(s)

    Parameters
    ----------
    predictors : DataTree
        A ``DataTree`` of ``xr.Dataset`` objects used as predictors. The ``DataTree``
        must have subtrees for each predictor, each of which has to have at least one
        non-empty leaf, holding a ``xr.Dataset`` representing a scenario. The subtrees of
        different predictors must be isomorphic (i.e. have the save scenarios). The ``xr.Dataset``
        must at least contain `dim` and each ``xr.Dataset`` must only hold one data variable.
    target : DataTree
        A ``DataTree``holding the targets. Must be isomorphic to the predictor subtrees, i.e.
        have the same scenarios. Each leaf must hold a ``xr.Dataset`` which must be at least 2D
        and contain `dim`, but may also contain a dimension for ensemble members.
    weights : DataTree or None, default: None
        Individual weights for each sample, must be isomorphic to target. Must at least contain
        `dim`, and must have the ensemble member dimension if target has it.
    stacking_dims : list[str]
        Dimension(s) to stack.
    collapse_dim : str, default: "scenario"
        Dimension along which to collapse the DataTrees, will automatically be added to the
        stacking dims.
    stacked_dim : str, default: "sample"
        Name of the stacked dimension.

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
    ├─ tas
    │  ├─ hist
    │  ├─ scen1
    │  └─ ...
    ├─ hfds
    │  ├─ hist
    │  ├─ scen1
    │  └─ ...
    └─ ...
    with 'hist' and 'scen1' being the scenarios, holding each a dataset with the same dimensions.
    """

    stacking_dims_all = stacking_dims + [collapse_dim]
    stack_dim = {stacked_dim: stacking_dims_all}

    # exclude target dimensions from broadcasting which are not in the stacking_dims
    exclude_dim = set(target.leaves[0].ds.dims) - set(stacking_dims)

    # prepare predictors
    predictors_stacked = xr.DataTree()
    for key, pred in predictors.items():
        # 1) broadcast to target (because pred is averaged over member)
        pred_broadcast = map_over_datasets(
            xr.Dataset.broadcast_like, pred, target, kwargs={"exclude": exclude_dim}
        )
        # TODO: use DataTree method again, once available
        # pred_broadcast = pred.broadcast_like(target, exclude=exclude_dim)

        predictors_stacked[key] = xr.DataTree(stack_datatree(pred_broadcast))

    # prepare target
    target_stacked = stack_datatree(target)

    # prepare weights
    if weights is not None:
        weights_stacked = stack_datatree(weights)
    else:
        weights_stacked = None

    return predictors_stacked, target_stacked, weights_stacked
