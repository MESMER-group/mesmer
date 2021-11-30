from unittest import mock
import pytest

import mesmer.core.linear_regression


@pytest.mark.parametrize("predictors,target,weight", (
    ([[1], [2], [3]], [1, 2, 2], None),
    ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1]),
))
def test_linear_regression(predictors, target, weight):
    with mock.patch("mesmer.core.linear_regression.LinearRegression") as mocked_linear_regression:

        if weight is None:
            expected_weights = None
            mesmer.core.linear_regression.linear_regression(predictors, target)
        else:
            expected_weights = weight
            mesmer.core.linear_regression.linear_regression(predictors, target, weight)

        mocked_linear_regression.assert_called_once()
        mocked_linear_regression.assert_called_with()
        mocked_linear_regression.fit.assert_called_once()
        mocked_linear_regression.fit.assert_called_with(
            X=predictors, y=target, sample_weight=expected_weights
        )


@pytest.mark.parametrize("predictors,target", (
    ([[1], [2], [3]], [1, 2]),
    ([[1, 2, 3], [2, 4, 0]], [1, 2, 2]),
))
def test_bad_shape(predictors, target):
    with pytest.raises(ValueError):
        mesmer.core.linear_regression.linear_regression(predictors, target)


@pytest.mark.parametrize("predictors,target,weight", (
    ([[1], [2], [3]], [1, 2, 2], [1, 10]),
    ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1, 1]),
))
def test_bad_shape_weights(predictors, target, weight):
    with pytest.raises(ValueError):
        mesmer.core.linear_regression.linear_regression(predictors, target, weight)

# - results (integration)
# - results with weights (integration)
