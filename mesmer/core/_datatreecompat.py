import functools

import xarray as xr
from packaging.version import Version

if Version(xr.__version__) < Version("2025.03"):
    raise ImportError(
        f"xarray version {xr.__version__} not supported - please upgrade to v2025.03 ("
        "or later)"
    )


def skip_empty_nodes(func):
    @functools.wraps(func)
    def _func(ds, *args, **kwargs):
        if not ds:
            return ds
        return func(ds, *args, **kwargs)

    return _func


def map_over_datasets(func, *args, kwargs=None):

    return xr.map_over_datasets(skip_empty_nodes(func), *args, kwargs=kwargs)


__all__ = [
    "map_over_datasets",
]
