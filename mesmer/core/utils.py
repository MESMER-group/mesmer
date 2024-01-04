import warnings
from typing import Union

import numpy as np
import xarray as xr


class OptimizeWarning(UserWarning):
    pass


class LinAlgWarning(UserWarning):
    pass


def create_equal_dim_names(dim, suffixes):

    if not len(suffixes) == 2:
        raise ValueError("must provide exactly two suffixes")

    return tuple(f"{dim}{suffix}" for suffix in suffixes)


def _minimize_local_discrete(func, sequence, **kwargs):
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
            return sequence[i - 1]

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


def _check_dataset_form(
    obj,
    name: str = "obj",
    *,
    required_vars: Union[str, set[str]] = set(),
    optional_vars: Union[str, set[str]] = set(),
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
    required_dims: Union[str, set[str]] = set(),
    shape=None,
):
    """check if a dataset conforms to some conditions

    obj: Any
        object to check.
    name : str, default: 'obj'
        Name to use in error messages.
    ndim, int, optional
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
