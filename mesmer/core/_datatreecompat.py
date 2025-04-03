import functools

import xarray as xr
from packaging.version import Version

if Version(xr.__version__) > Version("2025.01"):

    def skip_empty_nodes(func):
        @functools.wraps(func)
        def _func(ds, *args, **kwargs):
            if not ds:
                return ds
            return func(ds, *args, **kwargs)

        return _func

    from xarray import map_over_datasets as _map_over_datasets

    def map_over_datasets(func, *args, kwargs=None):

        return _map_over_datasets(skip_empty_nodes(func), *args, kwargs=kwargs)

else:
    raise ImportError(
        f"xarray version {xr.__version__} not supported - please upgrade to v2025.02 ("
        "or later)"
    )

__all__ = [
    "map_over_datasets",
]
