import numpy as np
import pytest
import scipy as sp
import xarray as xr
from sklearn.preprocessing import PowerTransformer

from mesmer.core.utils import _check_dataarray_form
from mesmer.stats._power_transformer import (
    YeoJohnsonTransformer,
    _yeo_johnson_inverse_transform_np,
    # _yeo_johnson_optimize_lambda_np,
    _yeo_johnson_transform_np,
)
from mesmer.testing import trend_data_2D


@pytest.mark.parametrize(
    "coeffs, t, expected",
    [
        ([1, 0.1], -np.inf, 2.0),
        ([1, 0.1], np.inf, 0.0),
        ([1, -0.1], -np.inf, 0.0),
        ([1, -0.1], np.inf, 2.0),
        ([0, 0], 1, 2),
        ([0, 1], 1, 2),
        ([1, 0], 1, 1),
        ([2, 0], 1, 2 / 3),
        ([1, 1], np.log(9), 2 / 10),
    ],
)
def test_lambda_function(coeffs, t, expected):

    yj_transformer = YeoJohnsonTransformer("logistic")
    result = yj_transformer.lambda_function(coeffs, t)
    np.testing.assert_allclose(result, expected)


def test_yeo_johnson_optimize_lambda_np_normal():
    # with enough random normal data points the fit should be close to 1 and 0
    np.random.seed(0)
    gridcells = 1
    n_months = 100_000
    monthly_residuals = np.random.standard_normal((n_months, gridcells)) * 10
    yearly_T = np.ones((n_months, gridcells))

    yj_transformer = YeoJohnsonTransformer("logistic")
    result = yj_transformer._optimize_lambda_np(monthly_residuals, yearly_T)
    # to test viability
    expected = np.array([1, 0])
    np.testing.assert_allclose(result, expected, atol=1e-2)

    # to test numerical stability
    expected_exact = np.array([9.976913e-01, -1.998520e-05])
    np.testing.assert_allclose(result, expected_exact, atol=1e-7)


@pytest.mark.parametrize(
    "skew, bounds",
    [
        (-2, [1, 2]),  # left skewed data
        (2, [0, 1]),  # right skewed data
        (-5, [1, 2]),  # more skew
        (5, [0, 1]),  # more skew
        (-0.5, [1, 2]),  # less skew
        (0.5, [0, 1]),  # less skew
        (0, [0.9, 1.1]),  # no skew
    ],
)
def test_yeo_johnson_optimize_lambda_np(skew, bounds):
    np.random.seed(0)
    n_years = 100_000

    yearly_T = np.random.randn(n_years)
    local_monthly_residuals = sp.stats.skewnorm.rvs(skew, size=n_years)

    yj_transformer = YeoJohnsonTransformer("logistic")
    coeffs = yj_transformer._optimize_lambda_np(local_monthly_residuals, yearly_T)
    lmbda = yj_transformer.lambda_function(coeffs, yearly_T)
    transformed = _yeo_johnson_transform_np(local_monthly_residuals, lmbda)

    assert (lmbda >= bounds[0]).all() & (lmbda <= bounds[1]).all()
    np.testing.assert_allclose(sp.stats.skew(transformed), 0, atol=0.1)


def test_yeo_johnson_transform_np_trivial():
    # NOTE: testing trivial transform with lambda = 1
    n_ts = 20

    lambdas = np.tile([1], (n_ts))

    monthly_residuals = np.ones(n_ts)

    result = _yeo_johnson_transform_np(monthly_residuals, lambdas)
    expected = np.ones(n_ts)

    np.testing.assert_equal(result, expected)


def test_yeo_johnson_transform_np_all():

    # test all possible combinations of local_monthly_residuals and lambdas
    local_monthly_residuals = np.array([0.0, 1.0, 0.0, 1.0, -1.0, -1.0])
    lambdas = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 2.0])

    result = _yeo_johnson_transform_np(local_monthly_residuals, lambdas)
    expected = np.array([0.0, 1.0, 0.0, np.log1p(1.0), -1.0, -np.log1p(1.0)])

    np.testing.assert_equal(result, expected)


def test_yeo_johnson_transform_np_sklearn():
    # test if our power transform is the same as sklearns
    np.random.seed(0)
    n_ts = 20

    lambdas = np.tile([2.0], (n_ts))

    monthly_residuals = sp.stats.skewnorm.rvs(2, size=n_ts)
    result = _yeo_johnson_transform_np(monthly_residuals, lambdas)

    pt_sklearn = PowerTransformer(method="yeo-johnson", standardize=False)
    pt_sklearn.lambdas_ = np.array([2.0])
    expected = pt_sklearn.transform(monthly_residuals.reshape(-1, 1))

    # only approximately the same due to #494
    np.testing.assert_allclose(result, expected.reshape(-1))


def test_transform_roundtrip():
    n_ts = 5
    # dummy data, having negative and positive values
    monthly_residuals = np.sin(np.linspace(0, 2 * np.pi, n_ts))

    # we have all cases for lambdas 0 and 2 (special cases), 1 (identity case)
    # lambda between 0 and 1 and lambda between 1 and 2 for concave and convex cases
    lambdas = np.array([0, 1, 2, 0.5, 1.5])

    transformed = _yeo_johnson_transform_np(monthly_residuals, lambdas)
    result = _yeo_johnson_inverse_transform_np(transformed, lambdas)

    np.testing.assert_allclose(result, monthly_residuals, atol=1e-7)


@pytest.mark.parametrize("value", (-2, -1, 0, 1, 2))
@pytest.mark.parametrize("lmbda", (0, 1, 2))
@pytest.mark.parametrize("lambda_delta", (-4e-16, 0, 4e-16))
def test_transform_roundtrip_special_cases(value, lmbda, lambda_delta):
    # test yeo_johnson for lambdas close to the boundary points

    x = np.array([value], dtype=float)

    lambdas = np.array([lmbda + lambda_delta], dtype=float)

    transformed = _yeo_johnson_transform_np(x, lambdas)
    result = _yeo_johnson_inverse_transform_np(transformed, lambdas)

    np.testing.assert_allclose(result, x)


def test_yeo_johnson_inverse_transform_np_sklearn():
    # test if our inverse power transform is the same as sklearn's for constant lambda
    np.random.seed(0)
    n_ts = 20

    lambdas = np.tile([2.0], (n_ts))

    monthly_residuals = sp.stats.skewnorm.rvs(2, size=n_ts)
    result = _yeo_johnson_inverse_transform_np(monthly_residuals, lambdas)

    pt_sklearn = PowerTransformer(method="yeo-johnson", standardize=False)
    pt_sklearn.lambdas_ = np.array([2.0])
    expected = pt_sklearn.inverse_transform(monthly_residuals.reshape(-1, 1))

    # only approximately the same due to #494
    np.testing.assert_allclose(result, expected.reshape(-1))


def test_yeo_johnson_optimize_lambda_sklearn():
    # test if our fit is the same as sklearns
    np.random.seed(0)
    n_ts = 100
    yearly_T_value = np.array(2.0)

    yearly_T = np.ones(n_ts) * yearly_T_value
    local_monthly_residuals = sp.stats.skewnorm.rvs(2, size=n_ts)

    yj_transformer = YeoJohnsonTransformer("logistic")
    ourfit = yj_transformer._optimize_lambda_np(local_monthly_residuals, yearly_T)
    result = yj_transformer.lambda_function(ourfit, yearly_T_value)
    sklearnfit = PowerTransformer(method="yeo-johnson", standardize=False).fit(
        local_monthly_residuals.reshape(-1, 1), yearly_T.reshape(-1, 1)
    )
    expected = sklearnfit.lambdas_

    np.testing.assert_allclose(np.array([result]), expected, atol=1e-5)


def skewed_data_2D(n_timesteps=30, n_lat=3, n_lon=2, stack=False):
    """
    Generate a 2D dataset with skewed data in time for each cell.
    The skewness of the data can be random for each cell when skew="random"
    or the same for all cells when skew is a number.
    """

    n_cells = n_lat * n_lon
    time = xr.date_range(
        start="2000-01-01", periods=n_timesteps, freq="MS", use_cftime=True
    )

    ts_array = np.empty((n_cells, n_timesteps))
    rng = np.random.default_rng(0)

    # create random data with a skew
    skew = rng.uniform(-5, 5, size=(n_cells, 1))

    ts_array = sp.stats.skewnorm.rvs(skew, size=(n_cells, n_timesteps))

    LON, LAT = np.meshgrid(np.arange(n_lon), np.arange(n_lat))
    coords = {
        "time": time,
        "lon": ("cells", LON.flatten()),
        "lat": ("cells", LAT.flatten()),
    }

    data = xr.DataArray(ts_array, dims=("cells", "time"), coords=coords, name="data")

    if stack:
        data = data.stack(sample=["time"], create_index=False)

    return data


@pytest.mark.parametrize("stack", (False, True))
def test_power_transformer_xr(stack):
    n_years = 100
    n_lon, n_lat = 2, 3
    n_gridcells = n_lat * n_lon

    monthly_residuals = skewed_data_2D(
        n_timesteps=n_years * 12, n_lat=n_lat, n_lon=n_lon, stack=stack
    )
    yearly_T = trend_data_2D(n_timesteps=n_years, n_lat=n_lat, n_lon=n_lon, scale=2)
    if stack:
        yearly_T = yearly_T.stack(sample=["time"], create_index=False)

    month = np.arange(1, 13)
    expected_month = xr.DataArray(month, coords={"month": month}, name="month")

    # 0. create instance
    yj_transformer = YeoJohnsonTransformer("logistic")

    # 1 - fitting
    pt_coefficients = yj_transformer.fit(yearly_T, monthly_residuals)
    # 2 - transformation
    transformed = yj_transformer.transform(yearly_T, monthly_residuals, pt_coefficients)
    # 3 - back-transformation
    inverse_transformed = yj_transformer.inverse_transform(
        yearly_T, transformed.transformed, pt_coefficients
    )

    sample_dim = "sample" if stack else "time"

    xr.testing.assert_allclose(
        inverse_transformed.inverted, monthly_residuals, atol=1e-5
    )

    _check_dataarray_form(
        transformed.transformed,
        name="transformed",
        ndim=2,
        required_dims={"cells", sample_dim},
        required_coords="time",
        shape=(n_gridcells, n_years * 12),
    )

    _check_dataarray_form(
        transformed.lambdas,
        name="lambdas",
        ndim=2,
        required_dims={"cells", sample_dim},
        required_coords="time",
        shape=(n_gridcells, n_years * 12),
    )
    xr.testing.assert_equal(monthly_residuals.time, transformed.time)

    _check_dataarray_form(
        inverse_transformed.inverted,
        name="inverted",
        ndim=2,
        required_dims={"cells", sample_dim},
        required_coords="time",
        shape=(n_gridcells, n_years * 12),
    )
    _check_dataarray_form(
        pt_coefficients,
        name="lambda_coeffs",
        ndim=3,
        required_dims={"cells", "coeff", "month"},
        shape=(12, n_gridcells, 2),
    )
    assert "month" in pt_coefficients.coords
    xr.testing.assert_equal(expected_month, pt_coefficients.month)
