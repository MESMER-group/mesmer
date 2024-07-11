from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mesmer
from mesmer.core.utils import LinAlgWarning, _check_dataarray_form, _check_dataset_form
from mesmer.testing import trend_data_1D, trend_data_2D, trend_data_3D


def test_select_ar_order_1d():

    data = trend_data_1D()

    result = mesmer.stats.select_ar_order(data, "time", 4)

    _check_dataarray_form(result, "selected_ar_order", ndim=0, shape=())


@pytest.mark.parametrize("n_lon", [1, 2])
@pytest.mark.parametrize("n_lat", [3, 4])
def test_select_ar_order_3d(n_lon, n_lat):

    data = trend_data_3D(n_lat=n_lat, n_lon=n_lon)

    result = mesmer.stats.select_ar_order(data, "time", 1)

    _check_dataarray_form(
        result,
        "selected_ar_order",
        ndim=2,
        required_dims={"lon", "lat"},
        shape=(n_lat, n_lon),
    )


def test_select_ar_order_dim():

    data = trend_data_3D(n_timesteps=4, n_lon=5)
    result = mesmer.stats.select_ar_order(data, "lon", 1)

    _check_dataarray_form(
        result, "selected_ar_order", ndim=2, required_dims={"time", "lat"}, shape=(4, 3)
    )


def test_select_ar_order():

    data = trend_data_2D()

    result = mesmer.stats.select_ar_order(data, "time", 4)

    coords = data.drop_vars("time").coords

    expected = xr.DataArray([4.0, 2.0, 2.0, 3.0, 4.0, 2.0], dims="cells", coords=coords)

    xr.testing.assert_equal(result, expected)


def test_select_ar_order_np():

    rng = np.random.default_rng(seed=0)
    data = rng.normal(size=100)

    result = mesmer.stats._auto_regression._select_ar_order_np(data, 2)
    assert np.isnan(result)

    result = mesmer.stats._auto_regression._select_ar_order_np(data[:10], 2)
    assert result == 2

    with pytest.raises(ValueError):
        mesmer.stats._auto_regression._select_ar_order_np(data[:6], 5)


@pytest.fixture
def ar_params_1D():

    intercept = xr.DataArray(0)
    coeffs = xr.DataArray([0], dims="lags")
    variance = xr.DataArray(0.5)
    ar_params = xr.Dataset(
        {"intercept": intercept, "coeffs": coeffs, "variance": variance}
    )

    return ar_params


@pytest.fixture
def ar_params_2D():

    intercept = xr.DataArray([0, 0], dims="gridcell")
    coeffs = xr.DataArray([[0, 0]], dims=("lags", "gridcell"))
    variance = xr.DataArray([0.5, 0.3], dims="gridcell")
    ar_params = xr.Dataset(
        {"intercept": intercept, "coeffs": coeffs, "variance": variance}
    )

    return ar_params


@pytest.fixture
def covariance():

    data = np.eye(2)

    covariance = xr.DataArray(data, dims=("gridcell_i", "gridcell_j"))

    return covariance


@pytest.mark.parametrize("drop", ("intercept", "coeffs", "variance"))
def test_draw_auto_regression_uncorrelated_wrong_input(ar_params_1D, drop):

    ar_params = ar_params_1D.drop_vars(drop)

    with pytest.raises(
        ValueError, match=f"ar_params is missing the required data_vars: {drop}"
    ):

        mesmer.stats.draw_auto_regression_uncorrelated(
            ar_params,
            time=1,
            realisation=1,
            seed=0,
            buffer=0,
        )


def test_draw_auto_regression_uncorrelated_2D_errors(ar_params_2D):

    with pytest.raises(
        ValueError,
        match="``_draw_auto_regression_uncorrelated`` can currently only handle single points",
    ):

        mesmer.stats.draw_auto_regression_uncorrelated(
            ar_params_2D,
            time=1,
            realisation=1,
            seed=0,
            buffer=0,
        )


@pytest.mark.parametrize("time", (3, 5))
@pytest.mark.parametrize("realization", (2, 4))
@pytest.mark.parametrize("time_dim", ("time", "ts"))
@pytest.mark.parametrize("realization_dim", ("realization", "sample"))
def test_draw_auto_regression_uncorrelated(
    ar_params_1D, time, realization, time_dim, realization_dim
):

    result = mesmer.stats.draw_auto_regression_uncorrelated(
        ar_params_1D,
        time=time,
        realisation=realization,
        seed=0,
        buffer=0,
        time_dim=time_dim,
        realisation_dim=realization_dim,
    )

    _check_dataarray_form(
        result,
        "result",
        ndim=2,
        required_dims={time_dim, realization_dim},
        shape=(time, realization),
    )


@pytest.mark.parametrize("dim", ("time", "realisation"))
@pytest.mark.parametrize("wrong_coords", (None, 2.0, np.array([1, 2]), xr.Dataset()))
def test_draw_auto_regression_uncorrelated_wrong_coords(
    ar_params_1D, dim, wrong_coords
):

    coords = {"time": 2, "realisation": 3}

    coords[dim] = wrong_coords

    with pytest.raises(
        TypeError,
        match=f"expected '{dim}' to be an `int`, pandas or xarray Index or a `DataArray`",
    ):
        mesmer.stats.draw_auto_regression_uncorrelated(
            ar_params_1D,
            **coords,
            seed=0,
            buffer=0,
        )


@pytest.mark.parametrize("dim", ("time", "realisation"))
def test_draw_auto_regression_uncorrelated_wrong_coords_2D(ar_params_1D, dim):

    coords = {"time": 2, "realisation": 3}

    coords[dim] = xr.DataArray([[1, 2]])

    with pytest.raises(ValueError, match="Coords must be 1D but have 2 dimensions"):
        mesmer.stats.draw_auto_regression_uncorrelated(
            ar_params_1D,
            **coords,
            seed=0,
            buffer=0,
        )


@pytest.mark.parametrize("dim", ("time", "realisation"))
@pytest.mark.parametrize("coords", (xr.DataArray([3, 5]), pd.Index([1, 2, 3])))
def test_draw_auto_regression_uncorrelated_coords(ar_params_1D, dim, coords):

    coords_ = {"time": 2, "realisation": 3}

    coords_[dim] = coords

    result = mesmer.stats.draw_auto_regression_uncorrelated(
        ar_params_1D,
        **coords_,
        seed=0,
        buffer=0,
    )

    assert result[dim].size == coords.size
    np.testing.assert_equal(result[dim].values, coords.values)


@pytest.mark.parametrize("drop", ("intercept", "coeffs"))
def test_draw_auto_regression_correlated_wrong_input(ar_params_2D, covariance, drop):

    ar_params = ar_params_2D.drop_vars(drop)

    with pytest.raises(
        ValueError, match=f"ar_params is missing the required data_vars: {drop}"
    ):

        mesmer.stats.draw_auto_regression_correlated(
            ar_params,
            covariance,
            time=1,
            realisation=1,
            seed=0,
            buffer=0,
        )


@pytest.mark.parametrize("time", (3, 5))
@pytest.mark.parametrize("realization", (2, 4))
@pytest.mark.parametrize("time_dim", ("time", "ts"))
@pytest.mark.parametrize("realization_dim", ("realization", "sample"))
def test_draw_auto_regression_correlated(
    ar_params_2D, covariance, time, realization, time_dim, realization_dim
):

    result = mesmer.stats.draw_auto_regression_correlated(
        ar_params_2D,
        covariance,
        time=time,
        realisation=realization,
        seed=0,
        buffer=0,
        time_dim=time_dim,
        realisation_dim=realization_dim,
    )

    n_gridcells = ar_params_2D.intercept.size

    _check_dataarray_form(
        result,
        "result",
        ndim=3,
        required_dims={time_dim, "gridcell", realization_dim},
        shape=(time, n_gridcells, realization),
    )


@pytest.mark.parametrize("dim", ("time", "realisation"))
@pytest.mark.parametrize("wrong_coords", (None, 2.0, np.array([1, 2]), xr.Dataset()))
def test_draw_auto_regression_correlated_wrong_coords(
    ar_params_2D, covariance, dim, wrong_coords
):

    coords = {"time": 2, "realisation": 3}

    coords[dim] = wrong_coords

    with pytest.raises(
        TypeError,
        match=f"expected '{dim}' to be an `int`, pandas or xarray Index or a `DataArray`",
    ):
        mesmer.stats.draw_auto_regression_correlated(
            ar_params_2D,
            covariance,
            **coords,
            seed=0,
            buffer=0,
        )


@pytest.mark.parametrize("dim", ("time", "realisation"))
def test_draw_auto_regression_correlated_wrong_coords_2D(ar_params_2D, covariance, dim):

    coords = {"time": 2, "realisation": 3}

    coords[dim] = xr.DataArray([[1, 2]])

    with pytest.raises(ValueError, match="Coords must be 1D but have 2 dimensions"):
        mesmer.stats.draw_auto_regression_correlated(
            ar_params_2D,
            covariance,
            **coords,
            seed=0,
            buffer=0,
        )


@pytest.mark.parametrize("dim", ("time", "realisation"))
@pytest.mark.parametrize("coords", (xr.DataArray([3, 5]), pd.Index([1, 2, 3])))
def test_draw_auto_regression_correlated_coords(ar_params_2D, covariance, dim, coords):

    coords_ = {"time": 2, "realisation": 3}

    coords_[dim] = coords

    result = mesmer.stats.draw_auto_regression_correlated(
        ar_params_2D,
        covariance,
        **coords_,
        seed=0,
        buffer=0,
    )

    assert result[dim].size == coords.size
    np.testing.assert_equal(result[dim].values, coords.values)


@pytest.mark.parametrize("ar_order", [1, 8])
@pytest.mark.parametrize("n_cells", [1, 10])
@pytest.mark.parametrize("n_samples", [2, 5])
@pytest.mark.parametrize("n_ts", [3, 7])
def test_draw_auto_regression_correlated_np_shape(ar_order, n_cells, n_samples, n_ts):

    intercept = np.zeros(n_cells)
    coefs = np.ones((ar_order, n_cells))
    variance = np.eye(n_cells)

    result = mesmer.stats._auto_regression._draw_auto_regression_correlated_np(
        intercept=intercept,
        coeffs=coefs,
        covariance=variance,
        n_samples=n_samples,
        n_ts=n_ts,
        seed=0,
        buffer=10,
    )

    expected_shape = (n_samples, n_ts, n_cells)

    assert result.shape == expected_shape


@pytest.mark.filterwarnings(
    "ignore:Covariance matrix is not positive definite, using eigh instead of cholesky."
)
@pytest.mark.parametrize("intercept", [0, 1, 3.14])
def test_draw_auto_regression_deterministic_intercept(intercept):

    result = mesmer.stats._auto_regression._draw_auto_regression_correlated_np(
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

    result = mesmer.stats._auto_regression._draw_auto_regression_correlated_np(
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


@pytest.mark.filterwarnings(
    "ignore:Covariance matrix is not positive definite, using eigh instead of cholesky."
)
def test_draw_auto_regression_deterministic_coefs_buffer():

    result = mesmer.stats._auto_regression._draw_auto_regression_correlated_np(
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
        result = mesmer.stats._auto_regression._draw_auto_regression_correlated_np(
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

    result = mesmer.stats._auto_regression._draw_auto_regression_correlated_np(
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


def test_draw_auto_regression_correlated_eigh():
    # test that the function uses eigh when the covariance matrix is not positive definite
    with pytest.warns(
        LinAlgWarning, match="Covariance matrix is not positive definite"
    ):
        result = mesmer.stats._auto_regression._draw_auto_regression_correlated_np(
            intercept=1,
            coeffs=np.array([[0.5, 0.7], [0.3, 0.2]]),
            covariance=np.zeros((2, 2)),
            n_samples=1,
            n_ts=4,
            seed=0,
            buffer=3,
        )

    expected = np.array([[[1.0, 1.0], [1.5, 1.7], [2.05, 2.39], [2.475, 3.013]]])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("obj", [xr.Dataset(), None])
def test_fit_auto_regression_xr_errors(obj):

    with pytest.raises(TypeError, match="Expected a `xr.DataArray`"):
        mesmer.stats.fit_auto_regression(obj, "dim", lags=1)


def test_fit_auto_regression_xr_1D_values():
    # values obtained by running the example - to ensure there are no changes in
    # statsmodels.tsa.ar_model.AutoReg

    data = trend_data_1D()
    lags = 1
    result = mesmer.stats.fit_auto_regression(data, "time", lags=lags)

    expected = xr.Dataset(
        {
            "intercept": 1.04728995,
            "coeffs": ("lags", [0.99682459]),
            "variance": 1.05381192,
            "lags": [1],
            "nobs": data["time"].size - lags,
        }
    )

    xr.testing.assert_allclose(result, expected)


def test_fit_auto_regression_xr_1D_values_lags():
    # values obtained by running the example - to ensure there are no changes in
    # statsmodels.tsa.ar_model.AutoReg

    data = trend_data_1D()
    lags = 2
    result = mesmer.stats.fit_auto_regression(data, "time", lags=[lags])

    expected = xr.Dataset(
        {
            "intercept": 2.08295035,
            "coeffs": ("lags", [0.99318256]),
            "variance": 1.18712735,
            "lags": [2],
            "nobs": data["time"].size - lags,
        }
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lags", [1, 2, [2]])
def test_fit_auto_regression_xr_1D(lags):

    data = trend_data_1D()
    res = mesmer.stats.fit_auto_regression(data, "time", lags=lags)

    lags = lags if not np.ndim(lags) == 0 else np.arange(lags) + 1

    _check_dataset_form(
        res,
        "_fit_auto_regression_result",
        required_vars=["intercept", "coeffs", "variance"],
    )

    _check_dataarray_form(res.intercept, "intercept", ndim=0, shape=())
    _check_dataarray_form(
        res.coeffs, "coeffs", ndim=1, required_dims={"lags"}, shape=(len(lags),)
    )
    _check_dataarray_form(res.variance, "variance", ndim=0, shape=())

    expected = xr.DataArray(lags, coords={"lags": lags})

    xr.testing.assert_allclose(res.lags, expected)


@pytest.mark.parametrize("lags", [1, 2])
def test_fit_auto_regression_xr_2D(lags):

    data = trend_data_2D()
    res = mesmer.stats.fit_auto_regression(data, "time", lags=lags)

    (n_cells,) = data.cells.shape

    _check_dataset_form(
        res,
        "_fit_auto_regression_result",
        required_vars=["intercept", "coeffs", "variance"],
    )

    _check_dataarray_form(res.intercept, "intercept", ndim=1, shape=(n_cells,))
    _check_dataarray_form(
        res.coeffs,
        "coeffs",
        ndim=2,
        required_dims={"cells", "lags"},
        shape=(n_cells, lags),
    )
    _check_dataarray_form(res.variance, "variance", ndim=1, shape=(n_cells,))


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

        mesmer.stats._auto_regression._fit_auto_regression_np(data, lags=lags)

        mocked_auto_regression.assert_called_once()
        mocked_auto_regression.assert_called_with(data, lags=lags)

        mocked_auto_regression_result.fit.assert_called_once()
        mocked_auto_regression_result.fit.assert_called_with()


@pytest.mark.parametrize("intercept", [1.0, -4.0])
@pytest.mark.parametrize("slope", [0.2, -0.3])
def test_fit_autoregression_monthly_np(slope, intercept):
    # test if autoregrerssion can fit using previous month as independent variable
    # and current month as dependent variable
    # NOTE: the fit for the slope is bounded between -1 and 1
    np.random.seed(0)
    prev_month = np.random.normal(size=100)
    cur_month = prev_month * slope + intercept

    result = mesmer.stats._auto_regression._fit_auto_regression_monthly_np(
        cur_month, prev_month
    )
    expected = (slope, intercept)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("slope", [2.0, -2])
def test_fit_autoregression_monthly_np_outside_bounds(slope):
    # test that given a slope outside the bounds of -1 and 1, the slope is clipped
    np.random.seed(0)
    prev_month = np.random.normal(size=100)
    cur_month = prev_month * slope + 3

    result = mesmer.stats._auto_regression._fit_auto_regression_monthly_np(
        cur_month, prev_month
    )
    expected_slope = 1.0 * np.sign(slope)
    np.testing.assert_allclose(result[0], expected_slope)


def test_fit_auto_regression_monthly():
    n_years = 20
    n_gridcells = 10
    np.random.seed(0)
    data = xr.DataArray(
        np.random.normal(size=(n_years * 12, n_gridcells)),
        dims=("time", "gridcell"),
        coords={
            "time": pd.date_range("2000-01-01", periods=n_years * 12, freq="M"),
            "gridcell": np.arange(n_gridcells),
        },
    )

    result = mesmer.stats.fit_auto_regression_monthly(data)

    _check_dataset_form(result, "result", required_vars={"slope", "intercept"})
    _check_dataarray_form(
        result.slope,
        "slope",
        ndim=2,
        required_dims={"month", "gridcell"},
        shape=(12, n_gridcells),
    )
    _check_dataarray_form(
        result.intercept,
        "intercept",
        ndim=2,
        required_dims={"month", "gridcell"},
        shape=(12, n_gridcells),
    )

    with pytest.raises(TypeError, match="Expected a `xr.DataArray`"):
        mesmer.stats.fit_auto_regression_monthly(data.values)


def test_predict_auto_regression_monthly_intercept():
    n_gridcells = 2
    slope = np.zeros((12, n_gridcells))
    slope = xr.DataArray(
        slope,
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    intercept = np.tile(np.arange(1, 13), n_gridcells).reshape(n_gridcells, 12).T
    intercept = xr.DataArray(
        intercept,
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    ar_params = xr.Dataset({"intercept": intercept, "slope": slope})

    n_years = 10
    time = pd.date_range("2000-01-01", periods=12 * n_years, freq="M")
    time = xr.DataArray(time, dims="time", coords={"time": time})
    result = mesmer.stats.predict_auto_regression_monthly(ar_params, time, 0)

    expected = (
        np.tile(np.arange(1, 13), n_gridcells * n_years)
        .reshape(n_gridcells, 12 * n_years)
        .T
    )
    expected = xr.DataArray(
        expected,
        dims=("time", "gridcell"),
        coords={"time": time, "gridcell": np.arange(n_gridcells)},
    )
    np.testing.assert_allclose(result, expected)


def test_predict_auto_regression_monthly():
    n_gridcells = 10
    n_years = 10
    np.random.seed(0)
    slopes = xr.DataArray(
        np.random.normal(-0.99, 0.99, size=(12, n_gridcells)),
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    intercepts = xr.DataArray(
        np.random.normal(-10, 10, size=(12, n_gridcells)),
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    ar_params = xr.Dataset({"intercept": intercepts, "slope": slopes})

    time = pd.date_range("2000-01-01", periods=n_years * 12, freq="M")
    time = xr.DataArray(time, dims="time", coords={"time": time})

    result = mesmer.stats.predict_auto_regression_monthly(ar_params, time, 10)

    _check_dataarray_form(
        result,
        "result",
        ndim=2,
        required_dims={"time", "gridcell"},
        shape=(len(time), n_gridcells),
    )


def test_fit_predict_autoregression_monthly_roundtrip():
    n_gridcells = 10
    n_years = 150
    buffer = 30 * 12
    np.random.seed(0)
    slopes = xr.DataArray(
        np.random.uniform(-0.99, 0.99, size=(12, n_gridcells)),
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    intercepts = xr.DataArray(
        np.random.normal(-10, 10, size=(12, n_gridcells)),
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    ar_params = xr.Dataset({"intercept": intercepts, "slope": slopes})

    time = pd.date_range("2000-01-01", periods=n_years * 12, freq="M")
    time = xr.DataArray(time, dims="time", coords={"time": time})

    data = mesmer.stats.predict_auto_regression_monthly(
        ar_params, time, buffer
    )
    AR_fit = mesmer.stats.fit_auto_regression_monthly(data)
    predicted = mesmer.stats.predict_auto_regression_monthly(
        ar_params, data.time, buffer
    )

    np.testing.assert_allclose(predicted, data)


@pytest.mark.parametrize("buffer", [2, 12, 12 * 5])
def test_draw_auto_regression_monthly_np_buffer(buffer):
    n_realisations = 1
    n_gridcells = 10
    seed = 0
    slope = np.random.normal(size=(12, n_gridcells))
    intercept = np.ones((12, n_gridcells))
    covariance = np.tile(np.eye(n_gridcells), 12).reshape(12, n_gridcells, n_gridcells)
    n_ts = 120

    res_wo_buffer = mesmer.stats._auto_regression._draw_auto_regression_monthly_np(
        intercept, slope, covariance, n_realisations, n_ts, seed, buffer=0
    )
    res_w_buffer = mesmer.stats._auto_regression._draw_auto_regression_monthly_np(
        intercept, slope, covariance, n_realisations, n_ts, seed, buffer=buffer
    )

    np.testing.assert_allclose(
        res_wo_buffer[:, buffer:, :], res_w_buffer[:, :-buffer, :]
    )


def test_draw_auto_regression_monthly():
    n_gridcells = 10
    n_realisations = 5
    n_years = 10
    seed = 0
    buffer = 10
    np.random.seed(0)
    slopes = xr.DataArray(
        np.random.normal(-0.99, 0.99, size=(12, n_gridcells)),
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    intercepts = xr.DataArray(
        np.random.normal(-10, 10, size=(12, n_gridcells)),
        dims=("month", "gridcell"),
        coords={"month": np.arange(1, 13), "gridcell": np.arange(n_gridcells)},
    )
    ar_params = xr.Dataset({"intercept": intercepts, "slope": slopes})

    covariance = xr.DataArray(
        np.tile(np.eye(n_gridcells), 12).T.reshape(12, n_gridcells, n_gridcells),
        dims=("month", "gridcell_i", "gridcell_j"),
        coords={
            "month": np.arange(1, 13),
            "gridcell_i": np.arange(n_gridcells),
            "gridcell_j": np.arange(n_gridcells),
        },
    )

    time = pd.date_range("2000-01-01", periods=n_years * 12, freq="M")
    time = xr.DataArray(time, dims="time", coords={"time": time})

    result = mesmer.stats.draw_auto_regression_monthly(
        ar_params,
        covariance,
        time,
        n_realisations,
        seed,
        buffer,
    )

    _check_dataarray_form(
        result,
        "result",
        ndim=3,
        required_dims={"time", "gridcell", "realisation"},
        shape=(len(time), n_gridcells, n_realisations),
    )
