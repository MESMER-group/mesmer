# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M-TP
"""

import numpy as np
import xarray as xr
from sklearn.linear_model import GammaRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
# import importlib
# import itertools
from joblib import Parallel, delayed
from mesmer._core.utils import _ignore_warnings

class FeaturewiseRuleGLM:
    """
    Feature-wise GLM with dynamic design rule for X per feature.
    Supports multiple feature dimensions or none.
    """
    def __init__(self, design_rule, alpha_grid, sample_dim, feature_dims=None,
                 cv=5, distribution="gamma", n_jobs=-1):
        self.design_rule = design_rule
        self.alpha_grid = alpha_grid
        self.sample_dim = sample_dim
        self.feature_dims = feature_dims or []  # list of dims, can be empty
        self.cv = cv
        self.distribution = distribution
        self.n_jobs = n_jobs
        self.models_ = {}  # dict: feature_index_tuple -> fitted model

    def _flatten_feature_coords(self, y):
        """Return list of feature coordinate tuples and y flattened along features"""
        if not self.feature_dims:
            return [()], y  # scalar feature
        # Flatten feature dims
        stacked = y.stack(_feature=self.feature_dims)
        return list(stacked._feature.values), stacked

    def _make_model(self, alpha):
        if self.distribution == "gamma":
            return GammaRegressor(alpha=alpha, max_iter=1000, tol=1e-6, fit_intercept = True)
        elif self.distribution == "lognormal":
            return Ridge(alpha=alpha, fit_intercept = True)
        else:
            raise ValueError("distribution must be 'gamma' or 'lognormal'")

    def fit(self, y, **data):
        feature_index_list, y_flat = self._flatten_feature_coords(y)
        n_samples = y_flat[self.sample_dim].size

        def _fit_one(f_idx):
            y_np = y_flat.sel(_feature=f_idx).transpose(self.sample_dim).values
            X_np, _ = self.design_rule(f_idx, **data)  # (n_samples, n_covariates)
            if X_np.shape[0] != n_samples:
                raise ValueError(f"Design matrix for feature {f_idx} has wrong number of samples")

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            best_score = np.inf
            best_model = None

            for alpha in self.alpha_grid:
                model = self._make_model(alpha)
                scores = []
                converged = False

                for train_idx, val_idx in kf.split(X_np):
                    Xtr, Xval = X_np[train_idx], X_np[val_idx]
                    ytr, yval = y_np[train_idx], y_np[val_idx]

                    if self.distribution == "lognormal":
                        if np.any(ytr <= 0):
                            continue
                        ytr = np.log(ytr)

                    try:
                        model.fit(Xtr, ytr)
                        converged = True
                    except Exception:
                        continue

                    ypred = model.predict(Xval)
                    if self.distribution == "lognormal":
                        ypred = np.exp(ypred)
                    scores.append(mean_squared_error(yval, ypred))

                if converged and scores:
                    score = np.mean(scores)
                    if score < best_score:
                        best_score = score
                        best_model = clone(model)

            if best_model is None:
                raise RuntimeError(f"No converged model for feature {f_idx}")

            # Refit full data
            y_full = y_np if self.distribution == "gamma" else np.log(y_np)
            best_model.fit(X_np, y_full)
            return f_idx, best_model

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(f_idx) for f_idx in feature_index_list
        )

        for f_idx, model in results:
            self.models_[f_idx] = model

        return self

    def predict(self, **data):
        feature_index_list = list(self.models_.keys())
        preds = []

        sample_coords = None  # optional, preserve coordinates

        for f_idx in feature_index_list:
            X_np, sample_coords = self.design_rule(f_idx, **data)
            ypred = self.models_[f_idx].predict(X_np)
            if self.distribution == "lognormal":
                ypred = np.exp(ypred)
            preds.append(ypred)

        mu = np.stack(preds, axis=1)  # (n_samples, n_flat_features)

        if not self.feature_dims:
            return xr.DataArray(mu[:, 0],
                                dims=(self.sample_dim,),
                                coords={self.sample_dim: sample_coords})

        # build unique coordinates per feature dim
        coords_dict = {self.sample_dim: sample_coords}
        for i, dim in enumerate(self.feature_dims):
            coords_dict[dim] = sorted(set(idx[i] for idx in feature_index_list))

        # reshape to (n_samples, *feature_dims)
        shape = [mu.shape[0]] + [len(coords_dict[d]) for d in self.feature_dims]
        mu = mu.reshape(shape)

        return xr.DataArray(mu, dims=[self.sample_dim] + self.feature_dims, coords=coords_dict)

    def to_dataset(self) -> xr.Dataset:
        if not self.models_:
            raise RuntimeError("Model is not fitted.")

        feature_keys = list(self.models_.keys())

        # Scalar feature case
        if not self.feature_dims:
            model = next(iter(self.models_.values()))
            ds = xr.Dataset(
                data_vars={
                    "coef_": (["covariate"], model.coef_),
                    "intercept_": model.intercept_,
                },
                coords={"covariate": np.arange(len(model.coef_))},
            )

        else:
            # Determine coordinate values per feature dim
            coords_per_dim = {
                dim: sorted(set(key[i] for key in feature_keys))
                for i, dim in enumerate(self.feature_dims)
            }

            # Determine sizes
            shape = [len(coords_per_dim[d]) for d in self.feature_dims]
            n_covariates = len(next(iter(self.models_.values())).coef_)

            coef_array = np.zeros(shape + [n_covariates])
            intercept_array = np.zeros(shape)

            # Fill arrays
            for key, model in self.models_.items():
                idx = tuple(
                    coords_per_dim[dim].index(key[i])
                    for i, dim in enumerate(self.feature_dims)
                )
                coef_array[idx] = model.coef_
                intercept_array[idx] = model.intercept_

            ds = xr.Dataset(
                data_vars={
                    "coef_": (
                        self.feature_dims + ["covariate"],
                        coef_array,
                    ),
                    "intercept_": (
                        self.feature_dims,
                        intercept_array,
                    ),
                },
                coords={
                    **coords_per_dim,
                    "covariate": np.arange(n_covariates),
                },
            )

        ds.attrs.update(
            {
                "sample_dim": self.sample_dim,
                "feature_dims": list(self.feature_dims),
                "distribution": self.distribution,
            }
        )

        return ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, design_rule):
        obj = cls(
            design_rule=design_rule,
            alpha_grid=[0.0],  # dummy; not used after fitting
            sample_dim=ds.attrs["sample_dim"],
            feature_dims=ds.attrs["feature_dims"],
            distribution=ds.attrs["distribution"],
        )

        obj.models_ = {}

        def _initialize_model_with_state(coef_vals, intercept_val):
            model = obj._make_model(alpha=0.0)

            n_cov = coef_vals.shape[-1]

            # --- IMPORTANT: dummy fit to create sklearn internals ---
            X_dummy = np.zeros((1, n_cov))
            y_dummy = np.ones(1)

            model.fit(X_dummy, y_dummy)

            # overwrite learned parameters
            model.coef_ = coef_vals.copy()
            model.intercept_ = float(intercept_val)

            # ensure feature count consistency
            model.n_features_in_ = n_cov

            return model

        # ---- Scalar feature case ----
        if not obj.feature_dims:
            coef_vals = ds["coef_"].values
            intercept_val = ds["intercept_"].values
            obj.models_[()] = _initialize_model_with_state(
                coef_vals, intercept_val
            )
            return obj

        # ---- Multi-feature case ----
        dim_sizes = [ds.sizes[d] for d in obj.feature_dims]

        for idx in np.ndindex(*dim_sizes):
            key = tuple(
                ds.coords[dim].values[i]
                for dim, i in zip(obj.feature_dims, idx)
            )

            coef_vals = ds["coef_"].isel(
                dict(zip(obj.feature_dims, idx))
            ).values

            intercept_val = ds["intercept_"].isel(
                dict(zip(obj.feature_dims, idx))
            ).values

            obj.models_[key] = _initialize_model_with_state(
                coef_vals, intercept_val
            )

        return obj