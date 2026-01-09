# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M
"""


import numpy as np
import xarray as xr
from sklearn.base import clone
from sklearn.neighbors import KernelDensity


class GroupedKDEXarray:
    def __init__(
        self,
        kde: KernelDensity,
        sample_dim: str,
        feature_dim: str,
        group_dim: str,
    ):
        self.kde_prototype = kde
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.group_dim = group_dim

        self.kdes: dict = {}
        self.feature_coords = None

    # -------------------------
    # sklearn-style API
    # -------------------------

    def get_params(self, deep=True):
        return self.kde_prototype.get_params(deep=deep)

    def set_params(self, **params):
        self.kde_prototype.set_params(**params)
        return self

    # -------------------------
    # fitting
    # -------------------------

    def fit(self, da: xr.DataArray):
        self.kdes = {}

        self.feature_coords = da.coords[self.feature_dim]

        for g in da[self.group_dim].values:
            da_g = da.sel({self.group_dim: g})

            X = da_g.transpose(self.sample_dim, self.feature_dim).values

            kde = clone(self.kde_prototype)
            kde.fit(X)
            self.kdes[g] = kde

        return self

    # -------------------------
    # scoring
    # -------------------------

    def score_samples(self, da: xr.DataArray) -> xr.DataArray:
        scores = []

        for g in da[self.group_dim].values:
            da_g = da.sel({self.group_dim: g})

            X = da_g.transpose(self.sample_dim, self.feature_dim).values

            logp = self.kdes[g].score_samples(X)

            scores.append(
                xr.DataArray(
                    logp,
                    dims=[self.sample_dim],
                    coords={self.sample_dim: da_g[self.sample_dim]},
                )
            )

        return xr.concat(scores, dim=self.group_dim).assign_coords(
            {self.group_dim: da[self.group_dim]}
        )

    # -------------------------
    # sampling
    # -------------------------

    def sample(
        self,
        n_samples: int,
        random_state: int | None = None,
    ) -> xr.DataArray:
        """
        Draw samples from each group-specific KDE.
        """
        samples = []

        for g, kde in self.kdes.items():
            Xs = kde.sample(n_samples, random_state=random_state)

            samples.append(
                xr.DataArray(
                    Xs,
                    dims=["sample", self.feature_dim],
                    coords={
                        "sample": np.arange(n_samples),
                        self.feature_dim: self.feature_coords,
                    },
                )
            )

        return xr.concat(samples, dim=self.group_dim).assign_coords(
            {self.group_dim: list(self.kdes.keys())}
        )

    # -------------------------
    # introspection helpers
    # -------------------------

    def get_kdes_as_xarray(self) -> xr.DataArray:
        """
        KDE objects indexed by group (object dtype).
        """
        return xr.DataArray(
            data=[self.kdes[g] for g in self.kdes],
            dims=[self.group_dim],
            coords={self.group_dim: list(self.kdes.keys())},
        )
