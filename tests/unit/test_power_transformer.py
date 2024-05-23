import numpy as np
import pytest
import scipy as sp

from mesmer.mesmer_m.power_transformer import (
    PowerTransformerVariableLambda,
    lambda_function,
)


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

    result = lambda_function(coeffs, t)
    np.testing.assert_allclose(result, expected)


def test_fit_power_transformer():
    # with enough random normal data points the fit should be close to 1 and 0
    np.random.seed(0)
    gridcells = 1
    n_months = 100_000
    monthly_residuals = np.random.standard_normal((n_months, gridcells)) * 10
    yearly_T = np.ones((n_months, gridcells))

    pt = PowerTransformerVariableLambda(standardize=False)
    # standardize false speeds up the fit and does not impact the coefficients
    pt.fit(monthly_residuals, yearly_T, gridcells)

    result = pt.coeffs_
    # to test viability
    expected = np.array([[1, 0]])
    np.testing.assert_allclose(result, expected, atol=1e-2)

    # to test numerical stability
    expected_exact = np.array([[9.976913e-01, -1.998520e-05]])
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
def test_yeo_johnson_optimize_lambda(skew, bounds):
    np.random.seed(0)
    n_years = 100_000

    yearly_T = np.random.randn(n_years)
    local_monthly_residuals = sp.stats.skewnorm.rvs(skew, size=n_years)

    pt = PowerTransformerVariableLambda(standardize=False)
    pt.coeffs_ = pt._yeo_johnson_optimize_lambda(local_monthly_residuals, yearly_T)
    lmbda = lambda_function(pt.coeffs_, yearly_T)
    transformed = pt._yeo_johnson_transform(local_monthly_residuals, lmbda)

    assert (lmbda >= bounds[0]).all() & (lmbda <= bounds[1]).all()
    np.testing.assert_allclose(sp.stats.skew(transformed), 0, atol=0.1)


def test_transform():
    # NOTE: testing trivial transform with lambda = 1
    n_ts = 20
    n_gridcells = 10

    pt = PowerTransformerVariableLambda(standardize=False)
    pt.coeffs_ = np.tile([1, 0], (n_gridcells, 1))

    monthly_residuals = np.ones((n_ts, n_gridcells))
    yearly_T = np.zeros((n_ts, n_gridcells))

    result = pt.transform(monthly_residuals, yearly_T)
    expected = np.ones((n_ts, n_gridcells))

    np.testing.assert_equal(result, expected)


def test_yeo_johnson_transform():
    pt = PowerTransformerVariableLambda(standardize=False)

    # test all possible combinations of local_monthly_residuals and lambdas
    local_monthly_residuals = np.array([0.0, 1.0, 0.0, 1.0, -1.0, -1.0])
    lambdas = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 2.0])

    result = pt._yeo_johnson_transform(local_monthly_residuals, lambdas)
    expected = np.array([0.0, 1.0, 0.0, np.log1p(1.0), -1.0, -np.log1p(1.0)])

    np.testing.assert_equal(result, expected)


def test_transform_roundtrip():
    n_ts = 20
    n_gridcells = 5
    # dummy seasonal cylce, having negative and positive values
    monthly_residuals = np.sin(np.linspace(0, 2 * np.pi, n_ts * n_gridcells)).reshape(
        n_ts, n_gridcells
    )
    yearly_T = np.zeros((n_ts, n_gridcells))

    pt = PowerTransformerVariableLambda(standardize=False)
    # dummy lambdas (since yearly_T is zero lambda comes out to be second coefficient)
    # we have all cases for lambdas 0 and 2 (special cases), 1 (identity case)
    # lambda between 0 and 1 and lambda between 1 and 2 for concave and convex cases
    pt.coeffs_ = np.array([[0, 0], [0, 1], [0, 2], [0, 0.5], [0, 1.5]])
    pt.mins_ = np.amin(monthly_residuals, axis=0)
    pt.maxs_ = np.amax(monthly_residuals, axis=0)

    transformed = pt.transform(monthly_residuals, yearly_T)
    result = pt.inverse_transform(transformed, yearly_T)

    np.testing.assert_allclose(result, monthly_residuals, atol=1e-7)


def test_standard_scaler():
    # generate random data with mean different from 0 and std different from one
    np.random.seed(0)
    n_ts = 100_000
    n_gridcells = 1
    monthly_residuals = np.random.randn(n_ts, n_gridcells) * 10 + 5
    yearly_T = np.zeros((n_ts, n_gridcells))

    # fit the transformer
    pt = PowerTransformerVariableLambda(standardize=True)
    pt.fit(monthly_residuals, yearly_T, n_gridcells)
    transformed = pt.transform(monthly_residuals, yearly_T)

    # the transformed data should have mean close to 0 and std close to 1
    np.testing.assert_allclose(np.mean(transformed), 0, atol=1e-2)
    np.testing.assert_allclose(np.std(transformed), 1, atol=1e-2)

    # inverse transform should give back the original data
    result = pt.inverse_transform(transformed, yearly_T)
    np.testing.assert_allclose(result, monthly_residuals, atol=1e-7)
