from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from datatree import DataTree
from statsmodels.tsa.arima_process import ArmaProcess

import mesmer
import mesmer.stats._auto_regression


def generate_ar_samples(ar, std=1, n_timesteps=100, n_ens=4):

    np.random.seed(0)

    data = ArmaProcess(ar, 1).generate_sample([n_timesteps, n_ens], scale=std)  # type: ignore[attr-defined] not typed in statsmodels

    ens = np.arange(n_ens)

    da = xr.DataArray(data, dims=("time", "ens"), coords={"ens": ens})

    return da


def _prepare_data(*dataarrays, data_format):
    if data_format == "tuple":
        # Return the data arrays as a tuple
        return dataarrays
    elif data_format == "dict":
        # Use provided data arrays to create a dictionary
        return {f"scen{i}": da for i, da in enumerate(dataarrays, 1)}
    elif data_format == "datatree":
        # Use provided data arrays to create a DataTree
        data = {
            f"scen{i}": xr.Dataset({"tas": da}) for i, da in enumerate(dataarrays, 1)
        }
        return DataTree.from_dict(data)  # type: ignore[attr-defined] not typed in datatree
    else:
        raise ValueError(f"Unknown format_type: {data_format}")


def _maybe_unpack_data_and_call(func: Callable, data_format: str, data, **kwargs):
    if data_format == "tuple":
        return func(*data, **kwargs)
    else:
        return func(data, **kwargs)


@pytest.mark.parametrize("data_format", ["tuple", "dict", "datatree"])
def test_select_ar_order_scen_ens_one_scen(data_format):

    da = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)
    data = _prepare_data(da, data_format=data_format)

    result = _maybe_unpack_data_and_call(
        mesmer.stats.select_ar_order_scen_ens,
        data_format,
        data,
        dim="time",
        ens_dim="ens",
        maxlag=5,
    )

    expected = xr.DataArray(3, coords={"quantile": 0.5})

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("data_format", ["tuple", "dict", "datatree"])
def test_select_ar_order_scen_ens_multi_scen_tuple(data_format):

    da1 = generate_ar_samples([1, 0.5, 0.3], n_timesteps=100, n_ens=4)
    da2 = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)

    data = _prepare_data(da1, da2, data_format=data_format)

    result = _maybe_unpack_data_and_call(
        mesmer.stats.select_ar_order_scen_ens,
        data_format,
        data,
        dim="time",
        ens_dim="ens",
        maxlag=5,
    )

    expected = xr.DataArray(2, coords={"quantile": 0.5})

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("data_format", ["tuple", "dict", "datatree"])
def test_select_ar_order_scen_ens_no_ens_dim(data_format):

    da = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)
    data = _prepare_data(da, data_format=data_format)

    result = _maybe_unpack_data_and_call(
        mesmer.stats.select_ar_order_scen_ens,
        data_format,
        data,
        dim="time",
        ens_dim=None,
        maxlag=5,
    )

    ens = [0, 1, 2, 3]
    expected = xr.DataArray(
        [3, 1, 3, 3], dims="ens", coords={"quantile": 0.5, "ens": ens}
    )

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("data_format", ["tuple", "dict", "datatree"])
@pytest.mark.parametrize("std", [1, 0.1, 0.5])
def test_fit_auto_regression_scen_ens_one_scen(data_format, std):

    n_timesteps = 100
    da = generate_ar_samples([1, 0.5, 0.3, 0.4], std, n_timesteps=n_timesteps, n_ens=4)
    data = _prepare_data(da, data_format=data_format)

    result = _maybe_unpack_data_and_call(
        mesmer.stats.fit_auto_regression_scen_ens,
        data_format,
        data,
        dim="time",
        ens_dim="ens",
        lags=3,
    )

    expected = mesmer.stats.fit_auto_regression(da, dim="time", lags=3)
    expected = expected.mean("ens")

    xr.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(np.sqrt(result.variance), std, rtol=1e-1)


@pytest.mark.parametrize("data_format", ["tuple", "dict", "datatree"])
def test_fit_auto_regression_scen_ens_multi_scen(data_format):
    da1 = generate_ar_samples([1, 0.5, 0.3], n_timesteps=100, n_ens=4)
    da2 = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=5)

    data = _prepare_data(da1, da2, data_format=data_format)

    result = _maybe_unpack_data_and_call(
        mesmer.stats.fit_auto_regression_scen_ens,
        data_format,
        data,
        dim="time",
        ens_dim="ens",
        lags=3,
    )

    da = xr.concat([da1, da2], dim="scen")
    da = da.stack(scen_ens=("scen", "ens")).dropna("scen_ens")
    expected = mesmer.stats.fit_auto_regression(da, dim="time", lags=3)
    expected = expected.unstack("scen_ens")
    expected = expected.mean("ens").mean("scen")

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("data_format", ["tuple", "dict", "datatree"])
def test_fit_auto_regression_scen_ens_no_ens_dim(data_format):

    da = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)
    data = _prepare_data(da, data_format=data_format)

    # simply fits each ens individually, no averaging
    result = _maybe_unpack_data_and_call(
        mesmer.stats.fit_auto_regression_scen_ens,
        data_format,
        data,
        dim="time",
        ens_dim=None,
        lags=3,
    )

    expected = mesmer.stats.fit_auto_regression(da, dim="time", lags=3)

    xr.testing.assert_allclose(result, expected)


def test_fit_auto_regression_scen_ens_errors():
    da = generate_ar_samples([1, 0.5, 0.3, 0.4], n_timesteps=100, n_ens=4)

    # data tree
    data = {"scen1": xr.Dataset({"tas": da})}
    dt1 = DataTree.from_dict(data)  # type: ignore[attr-defined] not typed in datatree
    dt2 = dt1.copy()

    with pytest.raises(ValueError, match="Only one DataTree can be passed."):
        mesmer.stats.fit_auto_regression_scen_ens(
            dt1, dt2, dim="time", ens_dim="ens", lags=3
        )

    data = {"scen1": xr.Dataset({"tas": da, "tas2": da})}
    dt = DataTree.from_dict(data)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Dataset must have only one data variable."):
        mesmer.stats.fit_auto_regression_scen_ens(dt, dim="time", ens_dim="ens", lags=3)

    # dict
    data = {"scen1": da, "scen2": da}
    with pytest.raises(ValueError, match="Only one dictionary can be passed."):
        mesmer.stats.fit_auto_regression_scen_ens(
            data, data, dim="time", ens_dim="ens", lags=3
        )

    # wrong datatype
    data = xr.Dataset({"tas": da})
    with pytest.raises(ValueError, match="Expected either"):
        mesmer.stats.fit_auto_regression_scen_ens(
            data, dim="time", ens_dim="ens", lags=3  # type: ignore[arg-type]
        )
