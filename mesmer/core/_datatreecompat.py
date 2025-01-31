from functools import partial

import xarray as xr
from packaging.version import Version

if Version(xr.__version__) < Version("2024.10"):

    from datatree import map_over_subtree

    def map_over_datasets(func, *args, **kwargs):
        "compatibility layer for older xarray versions"

        return map_over_subtree(func)(*args, **kwargs)

else:

    import functools

    def skip_empty_nodes(func):
        @functools.wraps(func)
        def _func(ds, *args, **kwargs):
            # print(ds)
            if not ds or not ds.data_vars:
                return ds
            return func(ds, *args, **kwargs)

        return _func

    from xarray import map_over_datasets as _map_over_datasets

    def map_over_datasets(func, *args, **kwargs):

        # https://github.com/pydata/xarray/issues/10009
        func = partial(func, **kwargs)

        return _map_over_datasets(skip_empty_nodes(func), *args)

    # raise ValueError("Currently not supported")
