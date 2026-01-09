# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from sklearn.base import TransformerMixin


class SklearnXarrayTransformer:
    def __init__(
        self,
        transformer: TransformerMixin,
        sample_dim: str,
        feature_dim: str,
        group_dims: Sequence[str] | None = None,
    ):
        self.transformer = transformer
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.group_dims = group_dims
        self.feature_coords = None

    def _stack_samples(self, da: xr.DataArray) -> xr.DataArray:
        stack_dims = (
            self.group_dims + [self.feature_dim]
            if self.group_dims
            else [self.feature_dim]
        )
        stacked = da.stack(_feature=stack_dims)
        stacked = stacked.transpose(self.sample_dim, "_feature")
        self.feature_coords = stacked.coords["_feature"]
        return stacked

    def _unstack_samples(
        self, arr: np.ndarray, da_template: xr.DataArray
    ) -> xr.DataArray:
        stacked = self._stack_samples(da_template)
        out = xr.DataArray(
            arr,
            dims=stacked.dims,
            coords=stacked.coords,
        )
        # Properly unstack to recover original shape
        return out.unstack("_feature").transpose(*da_template.dims)

    def fit(self, da: xr.DataArray):
        stacked = self._stack_samples(da)
        self.transformer.fit(stacked.values)
        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
        stacked = self._stack_samples(da)
        arr = self.transformer.transform(stacked.values)
        return self._unstack_samples(arr, da)

    def inverse_transform(self, da: xr.DataArray) -> xr.DataArray:
        stacked = self._stack_samples(da)
        arr = self.transformer.inverse_transform(stacked.values)
        return self._unstack_samples(arr, da)

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        return self.fit(da).transform(da)

    def get_params_as_xarray(self, param_name: str) -> xr.DataArray:
        param_values = getattr(self.transformer, param_name)
        return xr.DataArray(
            data=param_values,
            dims=["_feature"],
            coords={"_feature": self.feature_coords},
        ).unstack("_feature")
