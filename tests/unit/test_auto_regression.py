from unittest import mock

import numpy as np
import pytest
import xarray as xr

import mesmer.stats.auto_regression
from mesmer.core.utils import _check_dataarray_form, _check_dataset_form
from mesmer.testing import trend_data_1D, trend_data_2D, trend_data_3D


def test_select_ar_order_xr_1d():

    data = trend_data_1D()

    result = mesmer.stats.auto_regression._select_ar_order_xr(data, "time", 4)

    _check_dataarray_form(result, "selected_ar_order", ndim=0, shape=())


@pytest.mark.parametrize("n_lon", [1, 2])
@pytest.mark.parametrize("n_lat", [3, 4])
def test_select_ar_order_xr_3d(n_lon, n_lat):

    data = trend_data_3D(n_lat=n_lat, n_lon=n_lon)

    result = mesmer.stats.auto_regression._select_ar_order_xr(data, "time", 1)

    _check_dataarray_form(
        result,
        "selected_ar_order",
        ndim=2,
        required_dims={"lon", "lat"},
        shape=(n_lat, n_lon),
    )


def test_select_ar_order_xr_dim():

    data = trend_data_3D(n_timesteps=4, n_lon=5)
    result = mesmer.stats.auto_regression._select_ar_order_xr(data, "lon", 1)

    _check_dataarray_form(
        result, "selected_ar_order", ndim=2, required_dims={"time", "lat"}, shape=(4, 3)
    )


def test_select_ar_order_xr():

    data = trend_data_2D()

    result = mesmer.stats.auto_regression._select_ar_order_xr(data, "time", 4)

    coords = data.drop_vars("time").coords

    expected = xr.DataArray([4.0, 2.0, 2.0, 3.0, 4.0, 2.0], dims="cells", coords=coords)

    xr.testing.assert_equal(result, expected)


def test_select_ar_order_np():

    rng = np.random.default_rng(seed=0)
    data = rng.normal(size=100)

    result = mesmer.stats.auto_regression._select_ar_order_np(data, 2)
    assert np.isnan(result)

    result = mesmer.stats.auto_regression._select_ar_order_np(data[:10], 2)
    assert result == 2

    with pytest.raises(ValueError):
        mesmer.stats.auto_regression._select_ar_order_np(data[:6], 5)


@pytest.mark.parametrize("ar_order", [1, 8])
@pytest.mark.parametrize("n_cells", [1, 10])
@pytest.mark.parametrize("n_samples", [2, 5])
@pytest.mark.parametrize("n_ts", [3, 7])
def test_draw_auto_regression_correlated_np_shape(ar_order, n_cells, n_samples, n_ts):

    intercept = np.zeros(n_cells)
    coefs = np.ones((ar_order, n_cells))
    covariance = np.ones((n_cells, n_cells))

    result = mesmer.stats.auto_regression._draw_auto_regression_correlated_np(
        intercept=intercept,
        coeffs=coefs,
        covariance=covariance,
        n_samples=n_samples,
        n_ts=n_ts,
        seed=0,
        buffer=10,
    )

    expected_shape = (n_samples, n_ts, n_cells)

    assert result.shape == expected_shape


@pytest.mark.parametrize("intercept", [0, 1, 3.14])
def test_draw_auto_regression_deterministic_intercept(intercept):

    result = mesmer.stats.auto_regression._draw_auto_regression_correlated_np(
        intercept=intercept,
        coeffs=np.array([[0]]),
        covariance=[0],
        n_samples=1,
        n_ts=3,
        seed=0,
        buffer=10,
    )

    expected = np.full((1, 3, 1), intercept)

    np.testing.assert_equal(result, expected)

    result = mesmer.stats.auto_regression._draw_auto_regression_correlated_np(
        intercept=np.array([[0, intercept]]),
        coeffs=np.array([[0, 0]]),
        covariance=np.zeros((2, 2)),
        n_samples=1,
        n_ts=1,
        seed=0,
        buffer=10,
    )

    expected = np.array([0, intercept]).reshape(1, 1, 2)

    np.testing.assert_equal(result, expected)


def test_draw_auto_regression_deterministic_coefs_buffer():

    result = mesmer.stats.auto_regression._draw_auto_regression_correlated_np(
        intercept=1,
        coeffs=np.array([[1]]),
        covariance=[0],
        n_samples=1,
        n_ts=4,
        seed=0,
        buffer=1,
    )

    expected = np.arange(4).reshape(1, -4, 1)

    np.testing.assert_equal(result, expected)

    expected = np.array([0, 1, 1.5, 1.75, 1.875]).reshape(1, -1, 1)

    for i, buffer in enumerate([1, 2]):
        result = mesmer.stats.auto_regression._draw_auto_regression_correlated_np(
            intercept=1,
            coeffs=np.array([[0.5]]),
            covariance=[0],
            n_samples=1,
            n_ts=4,
            seed=0,
            buffer=buffer,
        )

        np.testing.assert_allclose(result, expected[:, i : i + 4])


def test_draw_auto_regression_random():

    result = mesmer.stats.auto_regression._draw_auto_regression_correlated_np(
        intercept=1,
        coeffs=np.array([[0.375], [0.125]]),
        covariance=0.5,
        n_samples=1,
        n_ts=4,
        seed=0,
        buffer=3,
    )

    expected = np.array([2.58455078, 3.28976946, 1.86569258, 2.78266986])
    expected = expected.reshape(1, 4, 1)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("obj", [xr.Dataset(), None])
def test_fit_auto_regression_xr_errors(obj):

    with pytest.raises(TypeError, match="Expected a `xr.DataArray`"):
        mesmer.stats.auto_regression._fit_auto_regression_xr(obj, "dim", lags=1)


def test_fit_auto_regression_xr_1D_values():
    # values obtained by running the example - to ensure there are no changes in
    # statsmodels.tsa.ar_model.AutoReg

    data = trend_data_1D()
    result = mesmer.stats.auto_regression._fit_auto_regression_xr(data, "time", lags=1)

    expected = xr.Dataset(
        {
            "intercept": 1.04728995,
            "coeffs": ("lags", [0.99682459]),
            "covariance": 1.05381192,
            "lags": [1],
        }
    )

    xr.testing.assert_allclose(result, expected)


def test_fit_auto_regression_xr_1D_values_lags():
    # values obtained by running the example - to ensure there are no changes in
    # statsmodels.tsa.ar_model.AutoReg

    data = trend_data_1D()
    result = mesmer.stats.auto_regression._fit_auto_regression_xr(
        data, "time", lags=[2]
    )

    expected = xr.Dataset(
        {
            "intercept": 2.08295035,
            "coeffs": ("lags", [0.99318256]),
            "covariance": 1.18712735,
            "lags": [2],
        }
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lags", [1, 2, [2]])
def test_fit_auto_regression_xr_1D(lags):

    data = trend_data_1D()
    res = mesmer.stats.auto_regression._fit_auto_regression_xr(data, "time", lags=lags)

    lags = lags if not np.ndim(lags) == 0 else np.arange(lags) + 1

    _check_dataset_form(
        res,
        "_fit_auto_regression_result",
        required_vars=["intercept", "coeffs", "covariance"],
    )

    _check_dataarray_form(res.intercept, "intercept", ndim=0, shape=())
    _check_dataarray_form(
        res.coeffs, "coeffs", ndim=1, required_dims={"lags"}, shape=(len(lags),)
    )
    _check_dataarray_form(res.covariance, "covariance", ndim=0, shape=())

    expected = xr.DataArray(lags, coords={"lags": lags})

    xr.testing.assert_allclose(res.lags, expected)


@pytest.mark.parametrize("lags", [1, 2])
def test_fit_auto_regression_xr_2D(lags):

    data = trend_data_2D()
    res = mesmer.stats.auto_regression._fit_auto_regression_xr(data, "time", lags=lags)

    (n_cells,) = data.cells.shape

    _check_dataset_form(
        res,
        "_fit_auto_regression_result",
        required_vars=["intercept", "coeffs", "covariance"],
    )

    _check_dataarray_form(res.intercept, "intercept", ndim=1, shape=(n_cells,))
    _check_dataarray_form(
        res.coeffs,
        "coeffs",
        ndim=2,
        required_dims={"cells", "lags"},
        shape=(n_cells, lags),
    )
    _check_dataarray_form(res.covariance, "covariance", ndim=1, shape=(n_cells,))


@pytest.mark.parametrize("lags", [1, 2])
def test_fit_auto_regression_np(lags):

    data = np.array([0, 1, 3.14])

    mock_auto_regressor = mock.Mock()
    mock_auto_regressor.params = np.array([0.1, 0.25])
    mock_auto_regressor.sigma2 = 3.14

    with mock.patch(
        "statsmodels.tsa.ar_model.AutoReg"
    ) as mocked_auto_regression, mock.patch(
        "statsmodels.tsa.ar_model.AutoRegResults"
    ) as mocked_auto_regression_result:

        mocked_auto_regression.return_value = mocked_auto_regression_result
        mocked_auto_regression_result.return_value = mock_auto_regressor

        mesmer.stats.auto_regression._fit_auto_regression_np(data, lags=lags)

        mocked_auto_regression.assert_called_once()
        mocked_auto_regression.assert_called_with(data, lags=lags, old_names=False)

        mocked_auto_regression_result.fit.assert_called_once()
        mocked_auto_regression_result.fit.assert_called_with()
