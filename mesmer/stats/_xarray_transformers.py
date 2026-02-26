# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M-TP
"""

import importlib
import itertools
from collections.abc import Sequence

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin, clone


class SklearnXarrayTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a sklearn transformer to an xarray.DataArray independently
    along feature dimension,and optionally independently along multiple
    group dimensions.

    Parameters
    ----------
    transformer : sklearn.base.TransformerMixin
        The sklearn transformer to apply.
        Only works for 1D feature-aligned fitted attributes
        (e.g., StandardScaler, MinMaxScaler, ...).
    sample_dim : str
        Dimension along which samples are represented.
    feature_dim : str
        Dimension along which features are represented.
    group_dims : Sequence[str] | None
        Optional dimension(s) along which independent transformations are applied.
        If None, the DataArray should be 2-D and the transformer is applied to the entire
        DataArray.
    """

    def __init__(
        self,
        transformer,
        sample_dim: str,
        feature_dim: str,
        group_dims: Sequence[str] | None = None,
    ):
        self.transformer = transformer
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.group_dims = tuple(group_dims) if group_dims is not None else None

        if not hasattr(transformer, "fit"):
            raise TypeError("not implemented for the selected transformer")

        if self.group_dims:
            if (
                self.sample_dim in self.group_dims
                or self.feature_dim in self.group_dims
            ):
                raise ValueError(
                    "sample_dim and feature_dim cannot be included in group_dims"
                )

        # set during fit
        self._transformers = None
        self._feature_coords = None

    def _check_fitted(self):
        if self._transformers is None:
            raise ValueError("This transformer is not fitted yet. Call 'fit' first.")

    def _check_feature_coords(self, da: xr.DataArray):
        # short explanation: I orginially checked
        # self._feature_coords is not None and not da[self.feature_dim].identical(self._feature_coords);
        # however, this fails also if dtypes or encoding changes,
        # so I implemented a slightly more forgiving version.
        if self._feature_coords is None:
            return

        current = da[self.feature_dim].values
        stored = self._feature_coords.values

        if current.shape != stored.shape or not np.array_equal(current, stored):
            raise ValueError(
                f"Feature coordinates differ from those used during fit "
                f"along dimension '{self.feature_dim}'."
            )

    def _clone_transformer(self):
        return clone(self.transformer)

    def _apply_per_group(self, da: xr.DataArray, method: str) -> xr.DataArray:
        """
        Apply a transformer method ('fit', 'transform', 'inverse_transform') per group.
        """
        self._check_feature_coords(da)

        out_list = []

        # If no group dims, create a single dummy key
        if self.group_dims is None:
            group_keys = [None]
        else:
            # Cartesian product of group coordinates
            group_coords = [da[dim].values for dim in self.group_dims]
            group_keys = list(itertools.product(*group_coords))

        for key in group_keys:
            if key is None:
                da_slice = da
            else:
                # slice per group
                da_slice = da
                for dim, val in zip(self.group_dims, key):
                    da_slice = da_slice.sel({dim: val})

            # ensure 2D array: (samples, features)
            X = da_slice.transpose(self.sample_dim, self.feature_dim).values

            if method == "fit":
                transformer = self._clone_transformer()
                transformer.fit(X)
                self._transformers[key] = transformer
                out_X = X
            else:
                transformer = self._transformers[key]
                out_X = getattr(transformer, method)(X)

            da_out = xr.DataArray(
                out_X,
                dims=(self.sample_dim, self.feature_dim),
                coords={
                    self.sample_dim: da_slice[self.sample_dim],
                    self.feature_dim: da_slice[self.feature_dim],
                },
            )

            # reattach group coordinates if present
            if key is not None:
                for dim, val in zip(self.group_dims, key):
                    da_out = da_out.assign_coords({dim: val}).expand_dims(dim)

            out_list.append(da_out)

            # combine slices
        out = xr.combine_by_coords(out_list)

        # ensure dimensions have the same order as original
        out = out.transpose(*da.dims)

        # overwrite coordinates with original coordinates
        # this somehow doesnt work as expected...
        for dim in da.dims:
            out = out.assign_coords({dim: da[dim]})

        # out = out.assign_coords({self.feature_dim: da[self.feature_dim]})

        return out

    def fit(self, da: xr.DataArray):
        """
        Fit transformer(s) independently for each group.
        """
        self._transformers = {}
        # store feature coordinates from first slice
        if self._feature_coords is None:
            self._feature_coords = da[self.feature_dim]

        self._apply_per_group(da, method="fit")
        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
        """
        Apply the fitted transformer(s) to each group.
        """
        self._check_fitted()
        return self._apply_per_group(da, method="transform")

    def inverse_transform(self, da: xr.DataArray) -> xr.DataArray:
        """
        Apply inverse_transform per group.
        """
        self._check_fitted()
        return self._apply_per_group(da, method="inverse_transform")

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        self.fit(da)
        return self.transform(da)

    def get_params_as_xarray(self, param_name: str) -> xr.DataArray:
        """
        Return a fitted transformer parameter as an xarray.DataArray.
        """
        self._check_fitted()
        out_list = []

        for key, transformer in self._transformers.items():
            if not hasattr(transformer, param_name):
                raise AttributeError(f"Transformer has no attribute '{param_name}'")
            values = getattr(transformer, param_name)
            if values.ndim != 1:
                raise ValueError(
                    f"Parameter '{param_name}' is not 1D and cannot be mapped to feature_dim."
                )

            da_out = xr.DataArray(
                values,
                dims=(self.feature_dim,),
                coords={self.feature_dim: self._feature_coords},
            )

            if key is not None:
                for dim, val in zip(self.group_dims, key):
                    da_out = da_out.assign_coords({dim: val}).expand_dims(dim)

            out_list.append(da_out)

        return xr.combine_by_coords(out_list) if len(out_list) > 1 else out_list[0]

    def to_dataset(self) -> xr.Dataset:
        "Storing xarray transformer as a dataset"
        self._check_fitted()

        data_vars = {}

        for param in dir(next(iter(self._transformers.values()))):
            if param.endswith("_"):
                try:
                    da = self.get_params_as_xarray(param)
                    data_vars[param] = da
                except Exception:
                    pass  # skip non-1D params

        ds = xr.Dataset(data_vars)

        ds.attrs.update(
            {
                "transformer_class": type(self.transformer).__name__,
                "transformer_module": type(self.transformer).__module__,
                "sample_dim": self.sample_dim,
                "feature_dim": self.feature_dim,
                "group_dims": self.group_dims,
            }
        )

        # in case there is only one group dim,
        # it needs to be in the right format to ensure
        # iteration works
        ds.attrs["group_dims"] = (
            list(self.group_dims) if self.group_dims is not None else None
        )

        return ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        "Initalizing xarray transformer from a stored dataset"
        module = importlib.import_module(ds.attrs["transformer_module"])
        transformer_cls = getattr(module, ds.attrs["transformer_class"])

        # this is needed to format group_dims correctly
        # ensuring there can be none, one or multiple
        # group_dims unfortunately created a bit of
        # overhead
        group_dims = ds.attrs.get("group_dims")
        if group_dims is None:
            group_dims = None
        elif isinstance(group_dims, str):
            group_dims = (group_dims,)
        else:
            group_dims = tuple(group_dims)

        obj = cls(
            transformer=transformer_cls(),
            sample_dim=ds.attrs["sample_dim"],
            feature_dim=ds.attrs["feature_dim"],
            group_dims=group_dims,
        )

        obj._transformers = {}

        # reconstruct transformers per group
        if obj.group_dims is None:
            keys = [None]
        else:
            group_coords = [ds.coords[d].values for d in obj.group_dims]
            keys = list(itertools.product(*group_coords))

        for key in keys:
            transformer = transformer_cls()

            for var in ds.data_vars:
                values = ds[var]
                if key is not None:
                    sel_dict = dict(zip(obj.group_dims, key))
                    values = values.sel(sel_dict)

                setattr(transformer, var, values.values)

            obj._transformers[key] = transformer

        obj._feature_coords = ds[obj.feature_dim]

        return obj
