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
from sklearn.pipeline import Pipeline


class XarrayPipeline:
    def __init__(self, steps: Sequence, sample_dim: str, feature_dim: str):
        """
        steps: list of tuples (name, sklearn_transformer)
        sample_dim: the dimension representing samples (rows for sklearn)
        feature_dim: the dimension representing features (columns for sklearn)
        """
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.pipeline = Pipeline(steps)
        self.feature_coords = None
        self._training_feature_coords = None  # <<< initialize here

    def _to_numpy(
        self, da: xr.DataArray, feature_override: str | None = None
    ) -> np.ndarray:
        """Convert DataArray to (samples, features) for sklearn."""
        if feature_override is not None:
            feature_dim = feature_override
            # DON'T overwrite self.feature_coords here, it's for training features
        else:
            if self.feature_dim in da.dims:
                feature_dim = self.feature_dim
                self.feature_coords = da.coords[
                    self.feature_dim
                ]  # only during normal transform
            elif "component" in da.dims:
                feature_dim = "component"
                # do NOT overwrite self.feature_coords
            else:
                raise KeyError(
                    f"Expected a feature dimension '{self.feature_dim}' or 'component', got {da.dims}"
                )

        sample_dim = self.sample_dim
        if sample_dim not in da.dims:
            raise KeyError(
                f"Expected sample dimension '{sample_dim}' in DataArray dims, got {da.dims}"
            )

        da_transposed = da.transpose(sample_dim, feature_dim)
        return da_transposed.values

    def _to_xarray(
        self,
        arr: np.ndarray,
        da_template: xr.DataArray,
        feature_override: str | None = None,
    ) -> xr.DataArray:
        """Reconstruct DataArray from numpy array with proper coordinates."""
        feature_dim = feature_override or self.feature_dim

        if (
            feature_dim in da_template.dims
            and arr.shape[1] == da_template.sizes[feature_dim]
        ):
            coords = {
                self.sample_dim: da_template.coords[self.sample_dim],
                feature_dim: da_template.coords[feature_dim],
            }
            dims = (self.sample_dim, feature_dim)
        else:
            coords = {
                self.sample_dim: da_template.coords[self.sample_dim],
                "component": np.arange(arr.shape[1]),
            }
            dims = (self.sample_dim, "component")

        return xr.DataArray(arr, dims=dims, coords=coords)

    def fit(self, da: xr.DataArray, y=None, **fit_params):
        arr = self._to_numpy(da)
        self.pipeline.fit(arr, y=y, **fit_params)
        # store training feature coordinates (original features, e.g., 'gridcell')
        self._training_feature_coords = da.coords[self.feature_dim]
        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
        arr = self._to_numpy(da)
        out = self.pipeline.transform(arr)
        return self._to_xarray(out, da)

    def fit_transform(self, da: xr.DataArray, y=None, **fit_params) -> xr.DataArray:
        arr = self._to_numpy(da)
        out = self.pipeline.fit_transform(arr, y=y, **fit_params)
        return self._to_xarray(out, da)

    def inverse_transform(self, da: xr.DataArray) -> xr.DataArray:
        # Input features = 'component'
        arr = self._to_numpy(da, feature_override="component")
        out = self.pipeline.inverse_transform(arr)
        # Use training features (gridcell) coords
        return xr.DataArray(
            data=out,
            dims=(self.sample_dim, self.feature_dim),
            coords={
                self.sample_dim: da.coords[self.sample_dim],
                self.feature_dim: self._training_feature_coords,
            },
        )

    def get_params_as_xarray(self, step_name: str, param_name: str) -> xr.DataArray:
        """Return a parameter from a step as an xarray.DataArray with feature coords."""
        step = dict(self.pipeline.named_steps)[step_name]
        param = getattr(step, param_name)
        # If parameter is 2D, assume (components, features)
        if isinstance(param, np.ndarray):
            if param.ndim == 1:
                dims = [self.feature_dim]
                coords = {self.feature_dim: self.feature_coords}
            else:
                # 2D case: assume components x features
                dims = ["component", self.feature_dim]
                coords = {
                    self.feature_dim: self.feature_coords,
                    "component": np.arange(param.shape[0]),
                }
            return xr.DataArray(param, dims=dims, coords=coords)
        else:
            return param


# class XarrayPipeline:
#     def __init__(
#         self,
#         steps: Sequence,
#         sample_dim: str,
#         feature_dim: str,
#         group_dim: Optional[str] = None,
#     ):
#         self.sample_dim = sample_dim
#         self.feature_dim = feature_dim
#         self.group_dim = group_dim

#         self.steps = steps
#         self.pipeline = Pipeline(steps) if group_dim is None else None
#         self.pipelines: Dict = {}

#         self.feature_coords = None

#     # ---------------------
#     # helpers
#     # ---------------------

#     def _to_numpy(self, da: xr.DataArray) -> np.ndarray:
#         da = da.transpose(self.sample_dim, self.feature_dim)
#         self.feature_coords = da.coords[self.feature_dim]
#         return da.values

#     def _to_xarray(self, arr: np.ndarray, da_template: xr.DataArray) -> xr.DataArray:
#         if arr.shape[1] == da_template.sizes[self.feature_dim]:
#             dims = (self.sample_dim, self.feature_dim)
#             coords = {
#                 self.sample_dim: da_template[self.sample_dim],
#                 self.feature_dim: da_template[self.feature_dim],
#             }
#         else:
#             dims = (self.sample_dim, "component")
#             coords = {
#                 self.sample_dim: da_template[self.sample_dim],
#                 "component": np.arange(arr.shape[1]),
#             }

#         return xr.DataArray(arr, dims=dims, coords=coords)

#     # ---------------------
#     # core sklearn methods
#     # ---------------------

#     def fit(self, da: xr.DataArray, y=None, **fit_params):
#         if self.group_dim is None:
#             arr = self._to_numpy(da)
#             self.pipeline.fit(arr, y=y, **fit_params)
#             return self

#         self.pipelines = {}
#         for g, da_g in da.groupby(self.group_dim):
#             pipe = Pipeline(self.steps)
#             arr = self._to_numpy(da_g)
#             pipe.fit(arr, y=y, **fit_params)
#             self.pipelines[g] = pipe

#         return self

#     def transform(self, da: xr.DataArray) -> xr.DataArray:
#         if self.group_dim is None:
#             arr = self._to_numpy(da)
#             out = self.pipeline.transform(arr)
#             return self._to_xarray(out, da)

#         out = []
#         for g, da_g in da.groupby(self.group_dim):
#             pipe = self.pipelines[g]
#             arr = self._to_numpy(da_g)
#             transformed = pipe.transform(arr)
#             out.append(self._to_xarray(transformed, da_g))

#         return xr.concat(out, dim=self.group_dim)

#     def fit_transform(self, da: xr.DataArray, y=None, **fit_params) -> xr.DataArray:
#         self.fit(da, y=y, **fit_params)
#         return self.transform(da)

#     def inverse_transform(self, da: xr.DataArray) -> xr.DataArray:
#         if self.group_dim is None:
#             arr = self._to_numpy(da)
#             out = self.pipeline.inverse_transform(arr)
#             return self._to_xarray(out, da)

#         out = []
#         for g, da_g in da.groupby(self.group_dim):
#             pipe = self.pipelines[g]
#             arr = self._to_numpy(da_g)
#             inv = pipe.inverse_transform(arr)
#             out.append(self._to_xarray(inv, da_g))

#         return xr.concat(out, dim=self.group_dim)

#     # ---------------------
#     # parameter access
#     # ---------------------

#     def get_params(self, deep=True):
#         if self.group_dim is None:
#             return self.pipeline.get_params(deep=deep)
#         return {g: p.get_params(deep=deep) for g, p in self.pipelines.items()}

#     def set_params(self, **params):
#         if self.group_dim is None:
#             self.pipeline.set_params(**params)
#         else:
#             for p in self.pipelines.values():
#                 p.set_params(**params)
#         return self

#     def get_params_as_xarray(self, step_name: str, param_name: str) -> xr.DataArray:
#         if self.group_dim is None:
#             step = self.pipeline.named_steps[step_name]
#             param = getattr(step, param_name)
#             return self._param_to_xarray(param)

#         arrays = []
#         for g, pipe in self.pipelines.items():
#             step = pipe.named_steps[step_name]
#             param = getattr(step, param_name)
#             da = self._param_to_xarray(param)
#             da = da.expand_dims({self.group_dim: [g]})
#             arrays.append(da)

#         return xr.concat(arrays, dim=self.group_dim)

#     def _param_to_xarray(self, param):
#         if param.ndim == 1:
#             return xr.DataArray(
#                 param,
#                 dims=[self.feature_dim],
#                 coords={self.feature_dim: self.feature_coords},
#             )

#         return xr.DataArray(
#             param,
#             dims=["component", self.feature_dim],
#             coords={
#                 "component": np.arange(param.shape[0]),
#                 self.feature_dim: self.feature_coords,
#             },
#         )
