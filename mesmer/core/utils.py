from typing import Set, Union

import xarray as xr


def _check_dataarray_form(
    da: xr.DataArray,
    name: str = None,
    ndim: int = None,
    required_dims: Union[str, Set[str]] = {},
):

    if name is None:
        name = "da"

    if isinstance(required_dims, str):
        required_dims = {required_dims}

    required_dims = set(required_dims)

    if required_dims is None:
        required_dims = {}

    if not isinstance(da, xr.DataArray):
        raise TypeError(f"Expected {name} to be an xr.DataArray, got {type(da)}")

    if ndim is not None and ndim != da.ndim:
        raise ValueError(f"{name} should be {ndim}-dimensional, but is {da.ndim}D")

    if required_dims - set(da.dims):
        missing_dims = " ,".join(required_dims - set(da.dims))
        raise ValueError(f"{name} is missing the required dims: {missing_dims}")
