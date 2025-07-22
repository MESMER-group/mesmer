from typing import TypeVar

import xarray as xr

T_DataArraySetTree = TypeVar(
    "T_DataArraySetTree", xr.DataArray, xr.Dataset, xr.DataTree
)
