import functools

import xarray as xr
from packaging.version import Version

if Version(xr.__version__) < Version("2024.10"):

    from datatree import DataTree, map_over_subtree, open_datatree

    def map_over_datasets(func, *args, **kwargs):
        "compatibility layer for older xarray versions"

        return map_over_subtree(func)(*args, **kwargs)

else:

    def skip_empty_nodes(func):
        @functools.wraps(func)
        def _func(ds, *args, **kwargs):
            # print(ds)
            if not ds:
                return ds
            return func(ds, *args, **kwargs)

        return _func

    from xarray import DataTree, open_datatree
    from xarray import map_over_datasets as _map_over_datasets

    def map_over_datasets(func, *args, **kwargs):

        # https://github.com/pydata/xarray/issues/10009
        func = functools.partial(func, **kwargs)

        return _map_over_datasets(skip_empty_nodes(func), *args)

    # raise ValueError("Currently not supported")

__all__ = [
    "DataTree",
    "map_over_datasets",
    "open_datatree",
]
