# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M
"""


import numpy as np
import statsmodels.api as sm
import xarray as xr
from joblib import Parallel, delayed

from mesmer._core.utils import _ignore_warnings


# haven't properly commented this yet - WIP
class GammaGLMXarray:

    def __init__(self, alphas, l1_wt=0.0001, n_jobs=-1):
        """
        Gamma GLM (log link), fitted independently for each (gridcell, month).

        alpha : list of scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        L1_wt  : float
            Must be in [0, 1].  The L1 penalty has weight L1_wt and the
            L2 penalty has weight 1 - L1_wt.
        """

        self.alphas = alphas
        self.l1_wt = l1_wt
        self.n_jobs = n_jobs
        self.params_ = None

    @_ignore_warnings(
        [
            "Elastic net fitting did not converge",
            "divide by zero encountered",
            "invalid value encountered",
        ]
    )
    def _fit_single(self, X, y):
        y_max = y.max()

        family = sm.families.Gamma
        link = sm.families.links.Log()

        glm = sm.GLM(y, X, family=family(link=link))

        last_res = None

        for alpha in self.alphas:
            try:
                res = glm.fit_regularized(
                    alpha=alpha,
                    L1_wt=self.l1_wt,
                    refit=False,
                )
                last_res = res
            except Exception:
                continue

            resid = res.fittedvalues - y
            if np.all(resid <= 0.4 * y_max):
                return res.params

        # safe fallback
        if last_res is None:
            return np.full(X.shape[1], np.nan)

        return last_res.params

    def fit(self, tas, tas_sq, pr, closest_locations):
        """
        Estimate regression coefficients.

        Parameters
        ----------
        tas, tas_sq, pr:
            DataArrays with dims (gridcell, year, month)

        closest_locations:
            DataArray with dims (gridcell, closest_gridcells)
        """

        gridcells = tas.gridcell.values
        months = tas.month.values

        n_years = tas.sizes["year"]
        n_closest = closest_locations.sizes["closest_gridcells"]
        n_cov = 1 + 2 * n_closest

        def _compute(i_grid, mon):
            nbrs = closest_locations.sel(gridcell=i_grid).values

            X = np.c_[
                np.ones(n_years),
                tas.sel(gridcell=nbrs, month=mon).T.values,
                tas_sq.sel(gridcell=nbrs, month=mon).T.values,
            ]

            y = pr.sel(gridcell=i_grid, month=mon).values
            return self._fit_single(X, y)

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_compute)(i_grid, mon) for mon in months for i_grid in gridcells
        )

        params = (
            np.stack(results)
            .reshape(len(months), len(gridcells), n_cov)
            .transpose(1, 0, 2)
        )

        covariates = (
            ["intercept"]
            + [f"tas_{i}" for i in range(n_closest)]
            + [f"tas_sq_{i}" for i in range(n_closest)]
        )

        self.params_ = xr.DataArray(
            params,
            dims=("gridcell", "month", "covariate"),
            coords={
                "gridcell": tas.gridcell,
                "month": tas.month,
                "covariate": covariates,
            },
            name="params",
        )

        return self

    def predict(self, tas, tas_sq, closest_locations):
        """
        Compute μ = exp(Xβ)
        """

        if self.params_ is None:
            raise RuntimeError("Model must be fitted first.")

        gridcells = tas.gridcell.values
        months = tas.month.values
        n_years = tas.sizes["year"]

        mu = np.empty((len(gridcells), n_years, len(months)))

        for ig, i_grid in enumerate(gridcells):
            nbrs = closest_locations.sel(gridcell=i_grid).values

            for im, mon in enumerate(months):
                beta = self.params_.sel(gridcell=i_grid, month=mon).values

                X = np.c_[
                    np.ones(n_years),
                    tas.sel(gridcell=nbrs, month=mon).T.values,
                    tas_sq.sel(gridcell=nbrs, month=mon).T.values,
                ]

                mu[ig, :, im] = np.exp(X @ beta)

        return xr.DataArray(
            mu,
            dims=("gridcell", "year", "month"),
            coords={
                "gridcell": tas.gridcell,
                "year": tas.year,
                "month": tas.month,
                "lat": tas.lat,
                "lon": tas.lon,
            },
            name="mu",
        )

    def residuals(self, pr, mu):
        return (np.log(pr / mu)).rename("residuals")
