from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest

import mesmer.core.linear_regression


@pytest.mark.parametrize(
    "predictors,target,weight",
    (
        ([[1], [2], [3]], [1, 2, 2], None),
        ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1]),
    ),
)
def test_linear_regression(predictors, target, weight):
    # This testing is really nasty because the function is (deliberately)
    # written without proper dependency injection. See e.g.
    # https://stackoverflow.com/a/46865495 which recommends against this
    # approach. At the moment, I can't see how to write a suitably simple
    # function for regressions that uses proper dependency injection and
    # doesn't make the interface more complicated.
    mock_regressor = mock.Mock()
    mock_regressor.intercept_ = 12
    mock_regressor.coef_ = [123, -38]

    with mock.patch(
        "mesmer.core.linear_regression.LinearRegression"
    ) as mocked_linear_regression:
        mocked_linear_regression.return_value = mock_regressor

        if weight is None:
            # check that the default behaviour is to pass None to `fit`
            # internally
            expected_weights = None
            res = mesmer.core.linear_regression.linear_regression(predictors, target)
        else:
            # check that the intended weights are indeed passed to `fit`
            # internally
            expected_weights = weight
            res = mesmer.core.linear_regression.linear_regression(
                predictors, target, weight
            )

        mocked_linear_regression.assert_called_once()
        mocked_linear_regression.assert_called_with()
        mock_regressor.fit.assert_called_once()
        mock_regressor.fit.assert_called_with(
            X=predictors, y=target, sample_weight=expected_weights
        )

    intercepts = np.atleast_2d(mock_regressor.intercept_).T
    coefficients = np.atleast_2d(mock_regressor.coef_)
    npt.assert_allclose(res, np.hstack([intercepts, coefficients]))


@pytest.mark.parametrize(
    "predictors,target",
    (
        ([[1], [2], [3]], [1, 2]),
        ([[1, 2, 3], [2, 4, 0]], [1, 2, 2]),
    ),
)
def test_bad_shape(predictors, target):
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        mesmer.core.linear_regression.linear_regression(predictors, target)


@pytest.mark.parametrize(
    "predictors,target,weight",
    (
        ([[1], [2], [3]], [1, 2, 2], [1, 10]),
        ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1, 1]),
    ),
)
def test_bad_shape_weights(predictors, target, weight):
    with pytest.raises(ValueError, match="sample_weight.shape.*expected"):
        mesmer.core.linear_regression.linear_regression(predictors, target, weight)


def test_basic_regression():
    res = mesmer.core.linear_regression.linear_regression([[0], [1], [2]], [0, 2, 4])

    npt.assert_allclose(res, [[0, 2]], atol=1e-10)


def test_basic_regression_two_targets():
    res = mesmer.core.linear_regression.linear_regression(
        [[0], [1], [2]], [[0, 1], [2, 3], [4, 5]]
    )

    npt.assert_allclose(res, [[0, 2], [1, 2]], atol=1e-10)


def test_basic_regression_three_targets():
    res = mesmer.core.linear_regression.linear_regression(
        [[0], [1], [2]], [[0, 1, 2], [2, 3, 7], [4, 5, 12]]
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
        weights=[10, 10, 10, 1e-3],
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


@pytest.mark.parametrize(
    "x,y,exp_output_shape",
    (
        # one predictor
        (np.array([[0], [1], [2]]), np.array([2, 7, 10]), (1, 2)),  # (3, 1)  # (3,)
        (
            np.array([[0], [1], [2]]),  # (3, 1)
            np.array([[2], [7], [10]]),  # (3, 1)
            (1, 2),
        ),
        (
            np.array([[0], [1], [2]]),  # (3, 1)
            np.array([[2, 4], [7, 14], [10, 20]]),  # (3, 2)
            (2, 2),
        ),
        # two predictors
        (
            np.array([[0, 1], [1, 3], [2, 4]]),  # (3, 2)
            np.array([2, 7, 10]),  # (3, )
            (1, 3),
        ),
        (
            np.array([[0, 1], [1, 3], [2, 4]]),  # (3, 2)
            np.array([[2], [7], [10]]),  # (3, 1)
            (1, 3),
        ),
        (
            np.array([[0, 1], [1, 3], [2, 4]]),  # (3, 2)
            np.array([[2, 4], [7, 14], [10, 20]]),  # (3, 2)
            (2, 3),
        ),
    ),
)
def test_linear_regression_output_shape(x, y, exp_output_shape):
    res = mesmer.core.linear_regression.linear_regression(x, y)

    assert res.shape == exp_output_shape
