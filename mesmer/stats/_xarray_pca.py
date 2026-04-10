from collections.abc import Sequence
import itertools
import importlib

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA


class SklearnXarrayPCA(BaseEstimator, TransformerMixin):
    """
    Apply PCA to an xarray.DataArray with optional grouping.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        PCA instance (can use n_components as int or float).
    sample_dim : str
        Sample dimension.
    feature_dim : str
        Feature dimension (input space).
    component_dim : str
        Output PCA dimension.
    group_dims : Sequence[str] | None
        Apply independent PCA per group.
    """

    def __init__(
        self,
        pca: PCA,
        sample_dim: str,
        feature_dim: str,
        component_dim: str = "component",
        group_dims: Sequence[str] | None = None,
    ):
        self.pca = pca
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.component_dim = component_dim
        self.group_dims = tuple(group_dims) if group_dims is not None else None

        if self.group_dims:
            if self.sample_dim in self.group_dims or self.feature_dim in self.group_dims:
                raise ValueError("sample_dim and feature_dim cannot be in group_dims")

        self._pcas = None
        self._feature_coords = None

    # -------------------------
    # utilities
    # -------------------------
    def _check_fitted(self):
        if self._pcas is None:
            raise ValueError("Not fitted yet.")

    def _clone_pca(self):
        return clone(self.pca)

    def _get_group_keys(self, da):
        if self.group_dims is None:
            return [None]
        coords = [da[d].values for d in self.group_dims]
        return list(itertools.product(*coords))

    def _slice_group(self, da, key):
        if key is None:
            return da
        out = da
        for dim, val in zip(self.group_dims, key):
            out = out.sel({dim: val})
        return out

    # -------------------------
    # core logic
    # -------------------------
    def fit(self, da: xr.DataArray):
        self._pcas = {}

        if self._feature_coords is None:
            self._feature_coords = da[self.feature_dim]

        for key in self._get_group_keys(da):
            da_slice = self._slice_group(da, key)

            X = da_slice.transpose(self.sample_dim, self.feature_dim).values

            pca = self._clone_pca()
            pca.fit(X)

            self._pcas[key] = pca

        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
        self._check_fitted()

        out_list = []

        for key in self._get_group_keys(da):
            da_slice = self._slice_group(da, key)
            X = da_slice.transpose(self.sample_dim, self.feature_dim).values

            pca = self._pcas[key]
            X_t = pca.transform(X)

            da_out = xr.DataArray(
                X_t,
                dims=(self.sample_dim, self.component_dim),
                coords={
                    self.sample_dim: da_slice[self.sample_dim],
                    self.component_dim: np.arange(X_t.shape[1]),
                },
            )

            if key is not None:
                for dim, val in zip(self.group_dims, key):
                    da_out = da_out.assign_coords({dim: val}).expand_dims(dim)

            out_list.append(da_out)

        return self._combine_groups(out_list)

    def inverse_transform(self, da: xr.DataArray) -> xr.DataArray:
        self._check_fitted()

        out_list = []

        for key in self._get_group_keys(da):
            da_slice = self._slice_group(da, key)
            X = da_slice.transpose(self.sample_dim, self.component_dim).values

            pca = self._pcas[key]

            # only use valid components (remove padding)
            n_comp = pca.n_components_
            X = X[:, :n_comp]

            X_inv = pca.inverse_transform(X)

            da_out = xr.DataArray(
                X_inv,
                dims=(self.sample_dim, self.feature_dim),
                coords={
                    self.sample_dim: da_slice[self.sample_dim],
                    self.feature_dim: self._feature_coords,
                },
            )

            if key is not None:
                for dim, val in zip(self.group_dims, key):
                    da_out = da_out.assign_coords({dim: val}).expand_dims(dim)

            out_list.append(da_out)

        out = xr.combine_by_coords(out_list)
        return out.transpose(*out.dims)

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        self.fit(da)
        return self.transform(da)

    # -------------------------
    # combine helper (handles variable n_components)
    # -------------------------
    def _combine_groups(self, out_list):
        # if component_dim is not present, just combine directly
        if self.component_dim not in out_list[0].dims:
            out = xr.combine_by_coords(out_list)
            return out.transpose(*out.dims)

        # otherwise: existing logic
        max_comp = max(arr.sizes[self.component_dim] for arr in out_list)

        aligned = []
        for arr in out_list:
            pad = max_comp - arr.sizes[self.component_dim]
            if pad > 0:
                arr = arr.pad(
                    {self.component_dim: (0, pad)},
                    constant_values=np.nan,
                )
            aligned.append(arr)

        out = xr.combine_by_coords(aligned)
        return out.transpose(*out.dims)

    # -------------------------
    # parameter export
    # -------------------------
    def get_params_as_xarray(self, param_name: str) -> xr.DataArray:
        self._check_fitted()
        out_list = []

        for key, pca in self._pcas.items():
            if not hasattr(pca, param_name):
                raise AttributeError(param_name)

            values = getattr(pca, param_name)

            # --- explicit handling of PCA attributes ---
            if param_name == "components_":
                dims = (self.component_dim, self.feature_dim)
                coords = {
                    self.component_dim: np.arange(values.shape[0]),
                    self.feature_dim: self._feature_coords,
                }

            elif param_name == "mean_":
                dims = (self.feature_dim,)
                coords = {
                    self.feature_dim: self._feature_coords,
                }

            elif param_name in (
                "explained_variance_",
                "explained_variance_ratio_",
                "singular_values_",
            ):
                dims = (self.component_dim,)
                coords = {
                    self.component_dim: np.arange(values.shape[0]),
                }

            else:
                # skip unsupported attributes explicitly
                continue

            da_out = xr.DataArray(values, dims=dims, coords=coords)

            # add group coordinates back
            if key is not None:
                for dim, val in zip(self.group_dims, key):
                    da_out = da_out.assign_coords({dim: val}).expand_dims(dim)

            out_list.append(da_out)

        if not out_list:
            raise ValueError(f"No valid data found for parameter '{param_name}'")

        return self._combine_groups(out_list) if len(out_list) > 1 else out_list[0]

    # -------------------------
    # serialization
    # -------------------------
    def to_dataset(self) -> xr.Dataset:
        self._check_fitted()
        example = next(iter(self._pcas.values()))

        required_attrs = [
            "components_",
            "mean_",
        ]

        optional_attrs = [
            "explained_variance_",
            "explained_variance_ratio_",
            "singular_values_",
        ]

        data_vars = {}

        # required → MUST succeed
        for attr in required_attrs:
            if not hasattr(example, attr):
                raise RuntimeError(f"PCA is missing required attribute '{attr}'")

            data_vars[attr] = self.get_params_as_xarray(attr)

        # optional → best effort
        for attr in optional_attrs:
            if hasattr(example, attr):
                try:
                    data_vars[attr] = self.get_params_as_xarray(attr)
                except Exception:
                    pass

        ds = xr.Dataset(data_vars)

        ds.attrs.update({
            "pca_class": type(self.pca).__name__,
            "pca_module": type(self.pca).__module__,
            "sample_dim": self.sample_dim,
            "feature_dim": self.feature_dim,
            "component_dim": self.component_dim,
            "group_dims": list(self.group_dims) if self.group_dims else None,
        })

        return ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        module = importlib.import_module(ds.attrs["pca_module"])
        pca_cls = getattr(module, ds.attrs["pca_class"])

        group_dims = ds.attrs.get("group_dims")

        if group_dims is None:
            group_dims = None
        elif isinstance(group_dims, str):
            group_dims = (group_dims,)   # critical fix
        else:
            group_dims = tuple(group_dims)

        obj = cls(
            pca=pca_cls(),
            sample_dim=ds.attrs["sample_dim"],
            feature_dim=ds.attrs["feature_dim"],
            component_dim=ds.attrs["component_dim"],
            group_dims=group_dims,
        )

        obj._pcas = {}

        if obj.group_dims is None:
            keys = [None]
        else:
            coords = [ds.coords[d].values for d in obj.group_dims]
            keys = list(itertools.product(*coords))

        for key in keys:
            pca = pca_cls()

            for var in ds.data_vars:
                values = ds[var]

                if key is not None:
                    sel = dict(zip(obj.group_dims, key))
                    sel = {k: v for k, v in sel.items() if k in values.dims}

                    if sel:
                        values = values.sel(sel)

                # --- enforce correct dimension order ---
                if var == "components_":
                    values = values.transpose(obj.component_dim, obj.feature_dim)

                elif var == "mean_":
                    values = values.transpose(obj.feature_dim)

                elif var in ("explained_variance_", "explained_variance_ratio_", "singular_values_"):
                    values = values.transpose(obj.component_dim)
                # now safe
                setattr(pca, var, values.values)

            # critical reconstruction
            if hasattr(pca, "components_"):
                pca.n_components_ = pca.components_.shape[0]
                pca.n_features_in_ = pca.components_.shape[1]
                if pca.components_.shape[0] > pca.components_.shape[1]:
                    raise ValueError(
                        "components_ likely transposed: expected (n_components, n_features)"
                    )

            obj._pcas[key] = pca

        # FIX HERE
        obj._feature_coords = ds.coords[obj.feature_dim]

        return obj