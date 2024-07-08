import numpy as np
import pytest
import scipy as sp
import xarray as xr
from sklearn.preprocessing import PowerTransformer

from mesmer.core.utils import _check_dataarray_form, _check_dataset_form
from mesmer.mesmer_m import power_transformer
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

    result = power_transformer.lambda_function(coeffs[0], coeffs[1], t)
    np.testing.assert_allclose(result, expected)


def test_yeo_johnson_optimize_lambda_np_normal():
    # with enough random normal data points the fit should be close to 1 and 0
    np.random.seed(0)
    gridcells = 1
    n_months = 100_000
    monthly_residuals = np.random.standard_normal((n_months, gridcells)) * 10
    yearly_T = np.ones((n_months, gridcells))

    result = power_transformer._yeo_johnson_optimize_lambda_np(
        monthly_residuals, yearly_T
    )
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

    coeffs = power_transformer._yeo_johnson_optimize_lambda_np(
        local_monthly_residuals, yearly_T
    )
    lmbda = power_transformer.lambda_function(coeffs[0], coeffs[1], yearly_T)
    transformed = power_transformer._yeo_johnson_transform_np(
        local_monthly_residuals, lmbda
    )

    assert (lmbda >= bounds[0]).all() & (lmbda <= bounds[1]).all()
    np.testing.assert_allclose(sp.stats.skew(transformed), 0, atol=0.1)


def test_yeo_johnson_transform_np_trivial():
    # NOTE: testing trivial transform with lambda = 1
    n_ts = 20

    lambdas = np.tile([1], (n_ts))

    monthly_residuals = np.ones((n_ts))

    result = power_transformer._yeo_johnson_transform_np(monthly_residuals, lambdas)
    expected = np.ones((n_ts))

    np.testing.assert_equal(result, expected)


def test_yeo_johnson_transform_np_all():

    # test all possible combinations of local_monthly_residuals and lambdas
    local_monthly_residuals = np.array([0.0, 1.0, 0.0, 1.0, -1.0, -1.0])
    lambdas = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 2.0])

    result = power_transformer._yeo_johnson_transform_np(
        local_monthly_residuals, lambdas
    )
    expected = np.array([0.0, 1.0, 0.0, np.log1p(1.0), -1.0, -np.log1p(1.0)])

    np.testing.assert_equal(result, expected)


def test_yeo_johnson_transform_np_sklearn():
    # test if our power trasform is the same as sklearns
    np.random.seed(0)
    n_ts = 20

    lambdas = np.tile([2.0], (n_ts))

    monthly_residuals = sp.stats.skewnorm.rvs(2, size=n_ts)
    result = power_transformer._yeo_johnson_transform_np(monthly_residuals, lambdas)

    pt_sklearn = PowerTransformer(method="yeo-johnson", standardize=False)
    pt_sklearn.lambdas_ = np.array([2.0])
    expected = pt_sklearn.transform(monthly_residuals.reshape(-1, 1))

    np.testing.assert_equal(result, expected.reshape(-1))


def test_transform_roundtrip():
    n_ts = 5
    # dummy data, having negative and positive values
    monthly_residuals = np.sin(np.linspace(0, 2 * np.pi, n_ts))

    # we have all cases for lambdas 0 and 2 (special cases), 1 (identity case)
    # lambda between 0 and 1 and lambda between 1 and 2 for concave and convex cases
    lambdas = np.array([0, 1, 2, 0.5, 1.5])

    transformed = power_transformer._yeo_johnson_transform_np(
        monthly_residuals, lambdas
    )
    result = power_transformer._yeo_johnson_inverse_transform_np(transformed, lambdas)

    np.testing.assert_allclose(result, monthly_residuals, atol=1e-7)


def test_yeo_johnson_inverse_transform_np_sklearn():
    # test if our inverse power trasform is the same as sklearn's for constant lambda
    np.random.seed(0)
    n_ts = 20

    lambdas = np.tile([2.0], (n_ts))

    monthly_residuals = sp.stats.skewnorm.rvs(2, size=n_ts)
    result = power_transformer._yeo_johnson_inverse_transform_np(
        monthly_residuals, lambdas
    )

    pt_sklearn = PowerTransformer(method="yeo-johnson", standardize=False)
    pt_sklearn.lambdas_ = np.array([2.0])
    expected = pt_sklearn.inverse_transform(monthly_residuals.reshape(-1, 1))

    np.testing.assert_equal(result, expected.reshape(-1))


def test_yeo_johnson_optimize_lambda_sklearn():
    # test if our fit is the same as sklearns
    np.random.seed(0)
    n_ts = 100
    yearly_T_value = 2

    yearly_T = np.ones(n_ts) * yearly_T_value
    local_monthly_residuals = sp.stats.skewnorm.rvs(2, size=n_ts)

    ourfit = power_transformer._yeo_johnson_optimize_lambda_np(
        local_monthly_residuals, yearly_T
    )
    result = power_transformer.lambda_function(ourfit[0], ourfit[1], yearly_T_value)
    sklearnfit = PowerTransformer(method="yeo-johnson", standardize=False).fit(
        local_monthly_residuals.reshape(-1, 1), yearly_T.reshape(-1, 1)
    )
    expected = sklearnfit.lambdas_

    np.testing.assert_allclose(np.array([result]), expected, atol=1e-5)


def skewed_data_2D(n_timesteps=30, n_lat=3, n_lon=2):
    """
    Generate a 2D dataset with skewed data in time for each cell.
    The skewness of the data can be random for each cell when skew="random"
    or the same for all cells when skew is a number.
    """

    n_cells = n_lat * n_lon
    time = xr.cftime_range(start="2000-01-01", periods=n_timesteps, freq="MS")

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

    return xr.DataArray(ts_array, dims=("cells", "time"), coords=coords, name="data")


def test_power_transformer_xr():
    n_years = 100
    n_lon, n_lat = 2, 3
    n_gridcells = n_lat * n_lon

    monthly_residuals = skewed_data_2D(
        n_timesteps=n_years * 12, n_lat=n_lat, n_lon=n_lon
    )
    yearly_T = trend_data_2D(n_timesteps=n_years, n_lat=n_lat, n_lon=n_lon, scale=2)

    # new method
    pt_coefficients = power_transformer.fit_yeo_johnson_transform(
        monthly_residuals, yearly_T
    )
    transformed = power_transformer.yeo_johnson_transform(
        monthly_residuals, pt_coefficients, yearly_T
    )
    inverse_transformed = power_transformer.inverse_yeo_johnson_transform(
        transformed.transformed, pt_coefficients, yearly_T
    )
    xr.testing.assert_allclose(
        inverse_transformed.inverted, monthly_residuals, atol=1e-5
    )

    _check_dataarray_form(
        transformed.transformed,
        name="transformed",
        ndim=2,
        required_dims=("cells", "time"),
        shape=(n_gridcells, n_years * 12),
    )
    _check_dataarray_form(
        inverse_transformed.inverted,
        name="inverted",
        ndim=2,
        required_dims=("cells", "time"),
        shape=(n_gridcells, n_years * 12),
    )
    _check_dataset_form(
        pt_coefficients, name="pt_coefficients", required_vars=("xi_0", "xi_1")
    )
    _check_dataarray_form(
        pt_coefficients.xi_0,
        name="xi_0",
        ndim=2,
        required_dims=("cells",),
        shape=(12, n_gridcells),
    )
