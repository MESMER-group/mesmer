import xarray as xr
from datatree import DataTree


def _extract_single_dataarray_from_dt(dt: DataTree) -> xr.DataArray:
    """
    Extract a single DataArray from a DataTree node, holding a ``Dataset`` with one ``DataArray``.
    """
    if not dt.has_data:
        raise ValueError("DataTree must contain data.")

    ds = dt.to_dataset()
    name, *others = ds.keys()
    if others:
        raise ValueError("DataTree must only contain one data variable.")

    da = ds.to_array().isel(variable=0).drop_vars("variable")
    return da.rename(name)


def collapse_datatree_into_dataset(dt: DataTree, dim: str) -> xr.Dataset:
    """
    Take a ``DataTree`` and collapse **all subtrees** in it into a single ``xr.Dataset`` along dim.
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
    # TODO: implement and test options for join and coords
    # TODO: could potentially be replaced by DataTree.merge_child_nodes in the future?
    datasets = [subtree.to_dataset() for subtree in dt.subtree if not subtree.is_empty]

    # Check if all datasets have the same dimensions
    first_dims = set(datasets[0].dims)
    if not all(set(ds.dims) == first_dims for ds in datasets):
        raise ValueError("All datasets must have the same dimensions")

    # Check that all dimensions have coordinates
    for ds in datasets:
        missing_coords = set(ds.dims) - set(ds.coords)
        if missing_coords:
            missing_coords = "', '".join(sorted(missing_coords))
            raise ValueError(
                f"Dimension(s) '{missing_coords}' must have a coordinate/coordinates."
            )

    # Concatenate datasets along the specified dimension
    ds = xr.concat(datasets, dim=dim)
    ds = ds.assign_coords(
        {dim: [subtree.name for subtree in dt.subtree if not subtree.is_empty]}
    )

    return ds


def stack_linear_regression_datatrees(
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
    stack_dim = {stacked_dim: stacking_dims_all}

    # exclude target dimensions from broadcasting which are not in the stacking_dims
    exclude_dim = set(target.leaves[0].ds.dims) - set(stacking_dims)

    # predictors need to be
    predictors_stacked = DataTree()
    for key, subtree in predictors.items():
        # 1) broadcast to target
        pred_broadcast = subtree.broadcast_like(target, exclude=exclude_dim)
        # 2) collapsed into DataSets
        predictor_ds = collapse_datatree_into_dataset(pred_broadcast, dim=collapse_dim)
        # 3) stacked
        predictors_stacked[key] = DataTree(
            predictor_ds.stack(stack_dim, create_index=False)
        )
    predictors_stacked = predictors_stacked.dropna(dim=stacked_dim)

    # target needs to be
    # 1) collapsed into DataSet
    target_ds = collapse_datatree_into_dataset(target, dim=collapse_dim)
    # 2) stacked
    target_stacked = target_ds.stack(stack_dim, create_index=False)
    target_stacked = target_stacked.dropna(dim=stacked_dim)

    # weights need to be
    if weights is not None:
        # 1) collapsed into DataSet
        weights_ds = collapse_datatree_into_dataset(weights, dim=collapse_dim)
        # 2) stacked
        weights_stacked = weights_ds.stack(stack_dim, create_index=False)
        weights_stacked = weights_stacked.dropna(dim=stacked_dim)
    else:
        weights_stacked = None

    return predictors_stacked, target_stacked, weights_stacked
