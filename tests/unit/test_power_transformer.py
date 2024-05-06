import numpy as np

from mesmer.mesmer_m.power_transformer import (
    PowerTransformerVariableLambda,
    lambda_function,
)


def test_lambda_function():
    # Note that we test with normally distributed data
    # which should make lambda close to 1 and
    # the coefficients close to 1 and 0
    # but for the sake of testing, we set the coefficients differently
    coeffs = [1, 0.1]
    local_yearly_T_test_data = np.random.normal(0, 1, 10)

    # even for random numbers, the lambdas should always be between 0 and 2
    lambdas = lambda_function(coeffs, local_yearly_T_test_data)

    assert np.all(lambdas > 0) and np.all(lambdas < 2)


def test_fit_power_transformer():
    # with enough random data points the fit should be close to 1 and 0
    # here we test with uniform random data because it is quicker to fit
    # Uniform data is also symmetrically distributed so coefficients
    # should be close to 1 and 0 as well

    gridcells = 1
    years = 100000
    monthly_residuals = np.random.rand(years, gridcells) * 10
    yearly_T = np.ones((years, gridcells))

    pt = PowerTransformerVariableLambda()
    pt.fit(monthly_residuals, yearly_T, gridcells)

    result = pt.coeffs_
    expected = np.array([[1, 0]])

    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_transform():
    n_years = 20
    n_gridcells = 10

    pt = PowerTransformerVariableLambda()
    pt.standardize = False
    pt.coeffs_ = np.tile([1, 0], (n_gridcells, 1))

    monthly_residuals = np.ones((n_years, n_gridcells))
    yearly_T = np.zeros((n_years, n_gridcells))

    result = pt.transform(monthly_residuals, yearly_T)
    expected = np.ones((n_years, n_gridcells))

    np.testing.assert_equal(result, expected)


def test_yeo_johnson_transform():
    pt = PowerTransformerVariableLambda()

    # test all possible combinations of local_monthly_residuals and lambdas
    local_monthly_residuals = np.array([0.0, 1.0, 0.0, 1.0, -1.0, -1.0])
    lambdas = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 2.0])

    result = pt._yeo_johnson_transform(local_monthly_residuals, lambdas)
    expected = np.array([0.0, 1.0, 0.0, np.log1p(1.0), -1.0, -np.log1p(1.0)])

    np.testing.assert_equal(result, expected)


def test_inverse_transform():
    n_ts = 20
    n_gridcells = 5
    # dummy seasonal cylce, having negative and positive values
    monthly_residuals = np.sin(
        np.linspace(0, 2 * np.pi, n_ts * n_gridcells)
    ).reshape(n_ts, n_gridcells)
    yearly_T = np.zeros((n_ts, n_gridcells))

    pt = PowerTransformerVariableLambda()
    pt.standardize = False
    # dummy lambdas (since yearly_T is zero lambda comes out to be second coefficient)
    # we have all cases for lambdas 0 and 2 (special cases), 1 (identity case)
    # lambda between 1 and 1 and lambda between 1 and 2 for concave and convex cases
    pt.coeffs_ = np.array([[0, 0], [0, 1], [0, 2], [0, 0.5], [0, 1.5]])
    pt.mins_ = np.amin(monthly_residuals, axis=0)
    pt.maxs_ = np.amax(monthly_residuals, axis=0)

    transformed = pt.transform(monthly_residuals, yearly_T)
    result = pt.inverse_transform(transformed, yearly_T)

    np.testing.assert_allclose(result, monthly_residuals, atol=1e-7)
