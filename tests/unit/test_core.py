from unittest import mock
import pytest

import mesmer.core.linear_regression


@pytest.mark.parametrize("predictors,target,weight", (
    ([[1], [2], [3]], [1, 2, 2], None),
    ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1]),
))
def test_linear_regression(predictors, target, weight):
    # This testing is really nasty because the function is (deliberately)
    # written without proper dependency injection. See e.g.
    # https://stackoverflow.com/a/46865495 which recommends against this
    # approach. At the moment, I can't see how to write a suitably simple
    # function for regressions without making the interface more complicated.
    mock_regressor = mock.Mock()
    with mock.patch("mesmer.core.linear_regression.LinearRegression") as mocked_linear_regression:
        mocked_linear_regression.return_value = mock_regressor

        if weight is None:
            expected_weights = None
            mesmer.core.linear_regression.linear_regression(predictors, target)
        else:
            expected_weights = weight
            mesmer.core.linear_regression.linear_regression(predictors, target, weight)

        mocked_linear_regression.assert_called_once()
        mocked_linear_regression.assert_called_with()
        mock_regressor.fit.assert_called_once()
        mock_regressor.fit.assert_called_with(
            X=predictors, y=target, sample_weight=expected_weights
        )
        mock_regressor.get_params.assert_called_once()


@pytest.mark.parametrize("predictors,target", (
    ([[1], [2], [3]], [1, 2]),
    ([[1, 2, 3], [2, 4, 0]], [1, 2, 2]),
))
def test_bad_shape(predictors, target):
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        mesmer.core.linear_regression.linear_regression(predictors, target)


@pytest.mark.parametrize("predictors,target,weight", (
    ([[1], [2], [3]], [1, 2, 2], [1, 10]),
    ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1, 1]),
))
def test_bad_shape_weights(predictors, target, weight):
    with pytest.raises(ValueError, match="sample_weight.shape.*expected"):
        mesmer.core.linear_regression.linear_regression(predictors, target, weight)

# - results (integration)
# - results with weights (integration)
# - results order of coefficients swaps if inputs swap
