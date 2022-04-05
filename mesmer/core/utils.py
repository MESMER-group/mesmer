from typing import Set, Union

import xarray as xr


def _to_set(arg):

    if arg is None:
        arg = {}

    if isinstance(arg, str):
        arg = {arg}

    arg = set(arg)

    return arg


def _check_dataset_form(
    obj,
    name: str = "obj",
    *,
    required_vars: Union[str, Set[str]] = set(),
    optional_vars: Union[str, Set[str]] = set(),
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
    ndim: int = None,
    required_dims: Union[str, Set[str]] = set(),
):
    """check if a dataset conforms to some conditions

    obj: Any
        object to check.
    name : str, default: 'obj'
        Name to use in error messages.
    ndim, int, optional
        Number of required dimensions
    required_dims: str, set of str, optional
        Names of dims that are required for obj

    Raises
    ------
    TypeError: if obj is not a xr.DataArray
    ValueError: if any of the conditions is violated

    """

    required_dims = _to_set(required_dims)

    if not isinstance(obj, xr.DataArray):
        raise TypeError(f"Expected {name} to be an xr.DataArray, got {type(obj)}")

    if ndim is not None and ndim != obj.ndim:
        raise ValueError(f"{name} should be {ndim}-dimensional, but is {obj.ndim}D")

    if required_dims - set(obj.dims):
        missing_dims = " ,".join(required_dims - set(obj.dims))
        raise ValueError(f"{name} is missing the required dims: {missing_dims}")
