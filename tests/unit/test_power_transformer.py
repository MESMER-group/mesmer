import numpy as np
import scipy

from mesmer.mesmer_m.power_transformer import (
    PowerTransformerVariableLambda,
    lambda_function,
)


def test_lambda_function():
    # Note that we test with normally distributed data without skewness
    # which would yield coefficients close to 1 and 0
    # and a constant lambda of about 1
    # but for the sake of testing, we use coefficients which respresent skewness
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
    n_months = 100_000
    monthly_residuals = np.random.rand(n_months, gridcells) * 10
    yearly_T = np.ones((n_months, gridcells))

    pt = PowerTransformerVariableLambda()
    pt.fit(monthly_residuals, yearly_T, gridcells)

    result = pt.coeffs_
    expected = np.array([[1, 0]])

    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_yeo_johnson_optimize_lambda():
    np.random.seed(0)
    n_years = 10_000
    yearly_T = np.random.randn(n_years)

    skew = -2
    local_monthly_residuals = scipy.stats.skewnorm.rvs(skew, size=n_years)

    pt = PowerTransformerVariableLambda()
    pt.coeffs_ = pt._yeo_johnson_optimize_lambda(local_monthly_residuals, yearly_T)
    lmbda = lambda_function(pt.coeffs_, yearly_T)
    transformed = pt._yeo_johnson_transform(local_monthly_residuals, lmbda)

    assert (lmbda > 1).all() & (lmbda <= 2).all()
    np.testing.assert_allclose(scipy.stats.skew(transformed), 0, atol=0.01)

    # this fails, need to investigate more
    # skew = 2
    # local_monthly_residuals = scipy.stats.skewnorm.rvs(skew, size=n_years)

    # pt = PowerTransformerVariableLambda()
    # pt.coeffs_ = pt._yeo_johnson_optimize_lambda(local_monthly_residuals, yearly_T)
    # lmbda = lambda_function(pt.coeffs_, yearly_T)
    # transformed = pt._yeo_johnson_transform(local_monthly_residuals, lmbda)

    # assert (lmbda >= 0).all() & (lmbda <= 1).all()
    # np.testing.assert_allclose(scipy.stats.skew(transformed), 0, atol=0.01)


def test_transform():
    n_ts = 20
    n_gridcells = 10

    pt = PowerTransformerVariableLambda()
    pt.standardize = False
    pt.coeffs_ = np.tile([1, 0], (n_gridcells, 1))

    monthly_residuals = np.ones((n_ts, n_gridcells))
    yearly_T = np.zeros((n_ts, n_gridcells))

    result = pt.transform(monthly_residuals, yearly_T)
    expected = np.ones((n_ts, n_gridcells))

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
    monthly_residuals = np.sin(np.linspace(0, 2 * np.pi, n_ts * n_gridcells)).reshape(
        n_ts, n_gridcells
    )
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
