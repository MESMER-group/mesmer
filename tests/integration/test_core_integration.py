import numpy as np
import numpy.testing as npt

import mesmer.core.linear_regression


def test_basic_regression():
    res = mesmer.core.linear_regression.linear_regression([[0], [1], [2]], [0, 2, 4])

    npt.assert_allclose(res, [0, 2], atol=1e-10)


def test_basic_regression_with_weights():
    res = mesmer.core.linear_regression.linear_regression(
        [[0], [1], [2], [3]], [0, 2, 4, 5], [10, 10, 10, 0.1]
    )

    npt.assert_allclose(res, [0.0065, 1.99], atol=1e-3)


def test_basic_regression_multidimensional():
    res = mesmer.core.linear_regression.linear_regression(
        [[0, 1], [1, 3], [2, 4]], [2, 7, 8]
    )

    # intercept before coefficients, in same order as columns of
    # predictors
    npt.assert_allclose(res, [-2, -3, 4])


def test_regression_order():
    x = np.array([[0, 1], [1, 3], [2, 4]])
    y = np.array([2, 7, 10])

    res_original = mesmer.core.linear_regression.linear_regression(x, y)

    res_reversed = mesmer.core.linear_regression.linear_regression(
        np.flip(x, axis=1), y
    )

    npt.assert_allclose(res_original[0], res_reversed[0], atol=1e-10)
    npt.assert_allclose(res_original[1:], res_reversed[-1:0:-1])


def test_regression_order_with_weights():
    x = np.array([[0, 1], [1, 3], [2, 4], [1, 1]])
    y = np.array([2, 7, 8, 0])
    weights = [10, 10, 10, 0.1]

    res_original = mesmer.core.linear_regression.linear_regression(
        x, y, weights=weights
    )
    res_reversed = mesmer.core.linear_regression.linear_regression(
        np.flip(x, axis=1), y, weights=weights
    )

    npt.assert_allclose(res_original[0], -1.89, atol=1e-2)
    npt.assert_allclose(res_original[0], res_reversed[0])
    npt.assert_allclose(res_original[1:], res_reversed[-1:0:-1])
