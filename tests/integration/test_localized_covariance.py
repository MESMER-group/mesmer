import numpy as np
import pytest

from mesmer.core.localized_covariance import (
    _adjust_ecov_ar1_np,
    _minimize_local_discrete,
)


def test_ecov_crossvalidation():

    import numpy as np

    from mesmer.core.localized_covariance import _ecov_crossvalidation

    data = np.random.rand(5, 3)
    weights = np.array([1, 1, 1, 1, 1])

    # trivial localizer 1
    localizer = {250: np.diag(np.ones(3))}

    # trivial localizer 2
    localizer = {250: np.ones((3, 3))}

    # nontrivial localizer (symmetric and diag == 1) between 0..1
    loc = np.random.uniform(size=(3, 3))
    ls = loc * loc.T
    ls[np.diag_indices(3)] = 1
    localizer = {250: ls}

    _ecov_crossvalidation(
        250, data=data, weights=weights, localizer=localizer, k_folds=2
    )


def test_adjust_ecov_ar1_np():

    np.random.seed(0)
    data = np.random.rand(3, 5)
    ar_coefs = np.random.randn(3)
    cov = np.cov(data)
    result = _adjust_ecov_ar1_np(cov, ar_coefs)

    expected = np.array(
        [
            [0.00853598, 0.00151781, 0.01149936],
            [0.00151781, 0.04518936, 0.05038083],
            [0.01149936, 0.05038083, 0.10299335],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "values, expected", [((5, 4, 3, 2, 3, 0), 3), ((0, 1, 2), 0), ((3, 2, 1), 2)]
)
def test_minimize_local_discrete(values, expected):

    data_dict = {key: value for key, value in enumerate(values)}

    def func(i):
        return data_dict[i]

    result = _minimize_local_discrete(func, data_dict.keys())

    assert result == expected


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_minimize_local_discrete_error(value):
    def func(i):
        return value

    with pytest.raises(ValueError, match=r"`fun` returned `\+/\-inf`"):
        _minimize_local_discrete(func, [0])
