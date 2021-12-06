import numpy as np
import numpy.testing as npt
import pytest

import mesmer.core.linear_regression


def test_basic_regression():
    res = mesmer.core.linear_regression.linear_regression([[0], [1], [2]], [0, 2, 4])

    npt.assert_allclose(res, [[0, 2]], atol=1e-10)


def test_basic_regression_two_targets():
    res = mesmer.core.linear_regression.linear_regression(
        [[0], [1], [2]],
        [[0, 1], [2, 3], [4, 5]]
    )

    npt.assert_allclose(res, [[0, 2], [1, 2]], atol=1e-10)


def test_basic_regression_three_targets():
    res = mesmer.core.linear_regression.linear_regression(
        [[0], [1], [2]],
        [[0, 1, 2], [2, 3, 7], [4, 5, 12]]
    )

    # each target gets its own row in the results
    npt.assert_allclose(res, [[0, 2], [1, 2], [2, 5]], atol=1e-10)


def test_basic_regression_with_weights():
    res = mesmer.core.linear_regression.linear_regression(
        [[0], [1], [2], [3]], [0, 2, 4, 5], [10, 10, 10, 0.1]
    )

    npt.assert_allclose(res, [[0.0065, 1.99]], atol=1e-3)


def test_basic_regression_multidimensional():
    res = mesmer.core.linear_regression.linear_regression(
        [[0, 1], [1, 3], [2, 4]], [2, 7, 8]
    )

    # intercept before coefficients, in same order as columns of
    # predictors
    npt.assert_allclose(res, [[-2, -3, 4]])


def test_basic_regression_multidimensional_multitarget():
    res = mesmer.core.linear_regression.linear_regression(
        [[0, 1], [1, 3], [2, 4]], [[2, 0], [7, 0], [8, 5]]
    )

    # intercept before coefficients, in same order as columns of
    # predictors, rows in same order as columns of target
    npt.assert_allclose(res, [[-2, -3, 4], [5, 10, -5]])


def test_regression_with_weights_multidimensional_multitarget():
    res = mesmer.core.linear_regression.linear_regression(
        [[0, 1], [1, 3], [2, 4], [3, 5]],
        [[2, 0], [7, 0], [8, 5], [11, 11]],
        # extra point with low weight alters results in a minor way
        weights=[10, 10, 10, 1e-3]
    )

    # intercept before coefficients, in same order as columns of
    # predictors, rows in same order as columns of target
    npt.assert_allclose(res, [[-2, -3, 4], [5, 10, -5]], atol=1e-2)


def test_regression_order():
    x = np.array([[0, 1], [1, 3], [2, 4]])
    y = np.array([2, 7, 10])

    res_original = mesmer.core.linear_regression.linear_regression(x, y)

    res_reversed = mesmer.core.linear_regression.linear_regression(
        np.flip(x, axis=1), y
    )

    npt.assert_allclose(res_original[0][0], res_reversed[0][0], atol=1e-10)
    npt.assert_allclose(res_original[0][1:], res_reversed[0][-1:0:-1])


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

    npt.assert_allclose(res_original[0][0], -1.89, atol=1e-2)
    npt.assert_allclose(res_original[0][0], res_reversed[0][0])
    npt.assert_allclose(res_original[0][1:], res_reversed[0][-1:0:-1])



@pytest.mark.parametrize("x,y,exp_output_shape", (
    # one predictor
    (
        np.array([[0], [1], [2]]), # (3, 1)
        np.array([2, 7, 10]), # (3,)
        (1, 2)
    ),
    (
        np.array([[0], [1], [2]]), # (3, 1)
        np.array([[2], [7], [10]]), # (3, 1)
        (1, 2)
    ),
    (
        np.array([[0], [1], [2]]), # (3, 1)
        np.array([[2, 4], [7, 14], [10, 20]]), # (3, 2)
        (2, 2)
    ),
    # two predictors
    (
        np.array([[0, 1], [1, 3], [2, 4]]), # (3, 2)
        np.array([2, 7, 10]), # (3, )
        (1, 3)
    ),
    (
        np.array([[0, 1], [1, 3], [2, 4]]),  # (3, 2)
        np.array([[2], [7], [10]]),  # (3, 1)
        (1, 3)
    ),
    (
        np.array([[0, 1], [1, 3], [2, 4]]),  # (3, 2)
        np.array([[2, 4], [7, 14], [10, 20]]), # (3, 2)
        (2, 3)
    ),
))
def test_linear_regression_output_shape(x, y, exp_output_shape):
    res = mesmer.core.linear_regression.linear_regression(x, y)

    assert res.shape == exp_output_shape

