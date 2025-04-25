import functools

import xarray as xr
from packaging.version import Version

if Version(xr.__version__) < Version("2025.03"):
    raise ImportError(
        f"xarray version {xr.__version__} not supported - please upgrade to v2025.03 ("
        "or later)"
    )


def _skip_empty_nodes(func):
    @functools.wraps(func)
    def _func(ds, *args, **kwargs):
        if not ds:
            return ds
        return func(ds, *args, **kwargs)

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


__all__ = [
    "map_over_datasets",
]
