import numpy as np
import pytest
import xarray as xr
from statsmodels.tsa.arima_process import ArmaProcess

import mesmer


def generate_ar_samples(ar, std=1, n_timesteps=100, n_ens=4):

    np.random.seed(0)

    data = ArmaProcess(ar, 1).generate_sample([n_timesteps, n_ens], scale=std)

    ens = np.arange(n_ens)

    da = xr.DataArray(data, dims=("time", "ens"), coords={"ens": ens})

    return da


def test_select_ar_order_scen_ens_one_scen():

    da = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)

    result = mesmer.stats._select_ar_order_scen_ens(
        da, dim="time", ens_dim="ens", maxlag=5
    )

    expected = xr.DataArray(3, coords={"quantile": 0.5})

    xr.testing.assert_equal(result, expected)


def test_select_ar_order_scen_ens_multi_scen():

    da1 = generate_ar_samples([1, 0.5, 0.3], n_timesteps=100, n_ens=4)
    da2 = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)

    result = mesmer.stats._select_ar_order_scen_ens(
        da1, da2, dim="time", ens_dim="ens", maxlag=5
    )

    expected = xr.DataArray(2, coords={"quantile": 0.5})

    xr.testing.assert_equal(result, expected)


def test_select_ar_order_scen_ens_no_ens_dim():

    da = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)

    result = mesmer.stats._select_ar_order_scen_ens(
        da, dim="time", ens_dim=None, maxlag=5
    )

    ens = [0, 1, 2, 3]
    expected = xr.DataArray(
        [3, 1, 3, 3], dims="ens", coords={"quantile": 0.5, "ens": ens}
    )

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("std", [1, 0.1, 0.5])
def test_fit_auto_regression_scen_ens_one_scen(std):

    n_timesteps = 100
    da = generate_ar_samples([1, 0.5, 0.3, 0.4], std, n_timesteps=n_timesteps, n_ens=4)

    result = mesmer.stats._fit_auto_regression_scen_ens(
        da, dim="time", ens_dim="ens", lags=3
    )

    expected = mesmer.stats.fit_auto_regression(da, dim="time", lags=3)
    expected = expected.mean("ens")

    xr.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(np.sqrt(result.variance), std, rtol=1e-1)


def test_fit_auto_regression_scen_ens_multi_scen():
    n_ens = xr.DataArray([4, 5], dims="scen")
    da1 = generate_ar_samples([1, 0.5, 0.3], n_timesteps=100, n_ens=4)
    da2 = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=5)

    result = mesmer.stats._fit_auto_regression_scen_ens(
        da1, da2, dim="time", ens_dim="ens", lags=3
    )

    da = xr.concat([da1, da2], dim="scen")
    da = da.stack(scen_ens=("scen", "ens")).dropna("scen_ens")
    expected = mesmer.stats.fit_auto_regression(da, dim="time", lags=3)
    expected = expected.unstack("scen_ens")
    expected = expected.mean("ens").mean("scen")

    xr.testing.assert_equal(result, expected)


def test_fit_auto_regression_scen_ens_no_ens_dim():

    da = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)

    # simply fits each ens individually, no averaging
    result = mesmer.stats._fit_auto_regression_scen_ens(
        da, dim="time", ens_dim=None, lags=3
    )

    expected = mesmer.stats.fit_auto_regression(da, dim="time", lags=3)

    xr.testing.assert_allclose(result, expected)
