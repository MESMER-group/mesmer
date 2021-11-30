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
            expected_weights = None
            res = mesmer.core.linear_regression.linear_regression(predictors, target)
        else:
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

    npt.assert_allclose(
        res, np.hstack([mock_regressor.intercept_, mock_regressor.coef_])
    )


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
