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
    pt.coeffs_

    result = pt.coeffs_[0]
    expected = np.array([1, 0])

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_transform():
    pt = PowerTransformerVariableLambda()
    pt.standardize = False
    pt.coeffs_ = np.tile([1, 0], (10, 1))

    monthly_residuals = np.ones((10, 10))
    yearly_T = np.zeros((10, 10))

    result = pt.transform(monthly_residuals, yearly_T)
    expected = np.ones((10, 10))

    np.testing.assert_equal(result, expected)


def test_yeo_johnson_transform():
    pt = PowerTransformerVariableLambda()

    # test all possible combinations of local_monthly_residuals and lambdas
    local_monthly_residuals = np.array([0, 1, 0, 1, -1, -1])
    lambdas = np.array([1, 1, 0, 0, 1, 2])

    result = pt._yeo_johnson_transform(local_monthly_residuals, lambdas)
    expected = np.array([0, 1, 0, 0, -1, 0])

    np.testing.assert_equal(result, expected)


def test_inverse_transform():
    monthly_residuals = np.ones((10, 10))
    yearly_T = np.zeros((10, 10))

    pt = PowerTransformerVariableLambda()
    pt.standardize = False
    pt.coeffs_ = np.tile([1, 0], (10, 1))
    pt.mins_ = np.amin(monthly_residuals, axis=0)
    pt.maxs_ = np.amax(monthly_residuals, axis=0)

    transformed = pt.transform(monthly_residuals, yearly_T)
    result = pt.inverse_transform(transformed, yearly_T)

    np.testing.assert_equal(result, monthly_residuals)
