import os
import warnings
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import cast

import numpy as np
import pandas as pd
import threadpoolctl
import xarray as xr

from mesmer.datatree import _datatree_wrapper


class OptimizeWarning(UserWarning):
    pass


class LinAlgWarning(UserWarning):
    pass


def _create_equal_dim_names(dim: str, suffixes: tuple[str, str]) -> tuple[str, str]:
    """appends suffixes to a dimension name

    required as two axes cannot share the same dimension name in xarray

    Parameters
    ----------
    dim : str
        Dimension name.
    suffixes : tuple[str, str]
        The suffixes to add to the dim name.

    Returns
    -------
    suffixed_dims : tuple[str, str]
        Dimension name with suffixes added
    """

    if not len(suffixes) == 2:
        raise ValueError("must provide exactly two suffixes")

    return tuple(f"{dim}{suffix}" for suffix in suffixes)


def _minimize_local_discrete(func: Callable, sequence: Iterable, **kwargs):
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
    ValueError : if `func` returns negative infinity for any input or all elements are `inf`.

    Notes
    -----
    - The function determines the local minimum, i.e., the loop is aborted if
      `func(sequence[i-1]) >= func(sequence[i])`.
    """

    current_min = float("inf")
    # ensure it's a list because we cannot get an item from an iterable
    sequence = list(sequence)
    element = None
    posinf_positions = list()

    for i, element in enumerate(sequence):

        res = func(element, **kwargs)

        if np.isneginf(res):
            raise ValueError(f"`fun` returned `-inf` at position '{i}'")
        # skip element if inf is returned - not sure about this?
        elif np.isinf(res):
            posinf_positions.append(str(i))

        if res <= current_min:
            current_min = res
        else:
            # need to return element from the previous iteration
            sel = i - 1
            if sel == 0:
                warnings.warn("First element is local minimum.", OptimizeWarning)

            if posinf_positions:
                positions = "', '".join(posinf_positions)
                msg = f"`fun` returned `inf` at position(s) '{positions}'"
                warnings.warn(msg, OptimizeWarning)

            return sequence[sel]

    if np.isinf(res):
        raise ValueError("`fun` returned `inf` for all positions")

    warnings.warn("No local minimum found, returning the last element", OptimizeWarning)

    return element


def _to_set(arg) -> set:

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


@_datatree_wrapper
def upsample_yearly_data(
    yearly_data: xr.DataArray | xr.Dataset | xr.DataTree,
    monthly_time: xr.DataArray | xr.Dataset | xr.DataTree,
    time_dim: str = "time",
):
    """Upsample yearly data to monthly resolution by repeating yearly values.

    Parameters
    ----------
    yearly_data : xarray.DataArray | xr.Dataset | xr.DataTree
        Yearly values to upsample.

    monthly_time : xarray.DataArray | xr.Dataset | xr.DataTree
        Monthly time used to define the time coordinates of the upsampled data.

    time_dim : str, default: 'time'
        Name of the time dimension.

    Returns
    -------
    upsampled_yearly_data: xarray.DataArray
        Upsampled yearly temperature values containing the yearly values for every month
        of the corresponding year.
    """

    _assert_required_coords(yearly_data, "yearly_data", required_coords=time_dim)
    _assert_required_coords(monthly_time, "monthly_time", required_coords=time_dim)

    # read out time coords - this also works if it's already time coords
    monthly_time = monthly_time[time_dim]
    _check_dataarray_form(monthly_time, "monthly_time", ndim=1)

    if yearly_data[time_dim].size * 12 != monthly_time.size:
        raise ValueError(
            "Length of monthly time not equal to 12 times the length of yearly data."
        )

    # we need to pass the dim (`time_dim` may be a no-dim-coordinate)
    # i.e., time_dim and sample_dim may or may not be the same
    (sample_dim,) = monthly_time.dims

    if isinstance(yearly_data.indexes.get(sample_dim), pd.MultiIndex):
        raise ValueError(
            f"The dimension of the time coords ({sample_dim}) is a pandas.MultiIndex,"
            " which is currently not supported. Potentially call"
            f" `yearly_data.reset_index('{sample_dim}')` first."
        )

    upsampled_yearly_data = (
        # repeats the data along new dimension
        yearly_data.expand_dims({"__new__": 12})
        # stack to remove new dim; target dim must have new name
        .stack(__sample__=(sample_dim, "__new__"), create_index=False)
        # so we need to rename it back
        .swap_dims(__sample__=sample_dim)
        # and ensure the time coords the ones from the monthly data
        .assign_coords({time_dim: (sample_dim, monthly_time.values)})
    )

    return upsampled_yearly_data


def _check_dataset_form(
    obj,
    name: str = "obj",
    *,
    required_vars: str | Iterable[str] | None = None,
    optional_vars: str | Iterable[str] | None = None,
    requires_other_vars: bool = False,
):
    """check if a dataset conforms to some conditions

    obj: Any
        object to check.
    name : str, default: 'obj'
        Name to use in error messages.
    required_vars, str, iterable of str, optional
        Variables that obj is required to contain.
    optional_vars: str, iterable of str, optional
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

    __tracebackhide__ = True

    required_vars = _to_set(required_vars)
    optional_vars = _to_set(optional_vars)

    if not isinstance(obj, xr.Dataset):
        raise TypeError(f"Expected {name} to be an xr.Dataset, got {type(obj)}")

    data_vars = set(obj.data_vars)

    missing_vars = required_vars - data_vars
    if missing_vars:
        missing = ",".join(missing_vars)
        raise ValueError(f"{name} is missing the required data_vars: {missing}")

    n_vars_except = len(data_vars - (required_vars | optional_vars))
    if requires_other_vars and n_vars_except == 0:

        raise ValueError(f"Expected additional variables on {name}")


def _check_dataarray_form(
    obj,
    name: str = "obj",
    *,
    ndim: tuple[int, ...] | int | None = None,
    required_dims: str | Iterable[str] | None = None,
    required_coords: str | Iterable[str] | None = None,
    shape: tuple[int, ...] | None = None,
):
    """check if a dataset conforms to some conditions

    obj: Any
        object to check.
    name : str, default: 'obj'
        Name to use in error messages.
    ndim, int or tuple of int, optional
        Number of required dimensions, can be a tuple of int if several are possible.
    required_dims: str, iterable of str, optional
        Names of dims that are required for obj
    shape : tuple of ints, default: None
        Required shape. Ignored if None.

    Raises
    ------
    TypeError: if obj is not a xr.DataArray
    ValueError: if any of the conditions is violated

    """

    __tracebackhide__ = True

    if not isinstance(obj, xr.DataArray):
        raise TypeError(f"Expected {name} to be an xr.DataArray, got {type(obj)}")

    ndim = cast(tuple[int], (ndim,) if np.isscalar(ndim) else ndim)
    if ndim is not None and obj.ndim not in ndim:
        *a, b = map(lambda x: f"{x}D", ndim)
        ndim_options = (a and ", ".join(a) + " or " or "") + b
        raise ValueError(f"{name} should be {ndim_options}, but is {obj.ndim}D")

    _assert_required_dims(obj, name=name, required_dims=required_dims)

    _assert_required_coords(obj, name=name, required_coords=required_coords)

    if shape is not None and obj.shape != shape:
        raise ValueError(f"{name} has wrong shape - expected {shape}, got {obj.shape}")


def _assert_required_dims(
    obj, name: str = "obj", required_dims: str | Iterable[str] | None = None
):

    __tracebackhide__ = True

    required_dims = _to_set(required_dims)

    if required_dims - set(obj.dims):
        missing_dims = " ,".join(required_dims - set(obj.dims))
        raise ValueError(f"{name} is missing the required dims: {missing_dims}")


def _assert_required_coords(
    obj, name: str = "obj", required_coords: str | Iterable[str] | None = None
):

    __tracebackhide__ = True

    required_coords = _to_set(required_coords)

    if required_coords - set(obj.coords):
        missing_coords = " ,".join(required_coords - set(obj.coords))
        raise ValueError(f"{name} is missing the required coords: {missing_coords}")


@contextmanager
def _set_threads_from_options():

    from mesmer.core.options import OPTIONS

    threads_option = OPTIONS["threads"]

    if threads_option == "default":
        threads = min(os.cpu_count() // 2, 16)
    else:
        threads = threads_option

    with threadpoolctl.threadpool_limits(limits=threads):
        yield
