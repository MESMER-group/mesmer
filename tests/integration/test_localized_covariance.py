import numpy as np
import pytest

from mesmer.core.localized_covariance import (
    _adjust_ecov_ar1_np,
    _ecov_crossvalidation,
    _find_localized_empirical_covariance_np,
    _get_neg_loglikelihood,
)
from mesmer.core.utils import LinAlgWarning


@pytest.fixture
def random_data_5x3():

    np.random.seed(0)
    data = np.random.rand(5, 3)
    return data


def test_find_localized_empirical_covariance_np(random_data_5x3):

    n_samples = 20
    n_gridpoints = 3
    np.random.seed(0)
    data = np.random.rand(n_samples, n_gridpoints)

    localizer = dict()
    # for i, crosscov in enumerate(np.arange(0, 1.01, 0.01)):
    for i, crosscov in enumerate(np.arange(0, 1.01, 0.1)):

        loc = np.full((n_gridpoints, n_gridpoints), fill_value=crosscov)
        loc[np.diag_indices(n_gridpoints)] = 1
        localizer[i] = loc

    weights = np.full(n_samples, 1)

    result, __, __ = _find_localized_empirical_covariance_np(
        data, weights, localizer, k_folds=2
    )
    expected = 0
    assert result == expected

    result, __, __ = _find_localized_empirical_covariance_np(
        data, weights, localizer, k_folds=3
    )
    expected = 1
    assert result == expected

    result, __, __ = _find_localized_empirical_covariance_np(
        data, weights, localizer, k_folds=8
    )
    expected = 6
    assert result == expected


def test_ecov_crossvalidation_k_folds(random_data_5x3):

    weights = np.array([1, 1, 1, 1, 1])

    # trivial localizer
    localizer = {250: np.diag(np.ones(3))}

    result = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=2
    )
    expected = 204.5516663440938
    np.testing.assert_allclose(result, expected)

    result = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=3
    )
    expected = 183.32294464558134
    np.testing.assert_allclose(result, expected)

    # there is a maximum of 5 folds for 5 samples -> same result for larger k_folds
    result5 = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=5
    )
    result6 = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=6
    )

    np.testing.assert_allclose(result5, result6)


def test_ecov_crossvalidation_localizer(random_data_5x3):

    weights = np.array([1, 1, 1, 1, 1])

    # trivial localizer 1
    localizer = {250: np.diag(np.ones(3))}

    result = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=5
    )
    expected = 133.975629
    np.testing.assert_allclose(result, expected)

    # trivial localizer 2
    localizer = {250: np.ones((3, 3))}

    result = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=5
    )
    expected = 985.073313
    np.testing.assert_allclose(result, expected)

    # nontrivial localizer (symmetric and diag == 1) between 0..1
    np.random.seed(0)
    loc = np.random.uniform(size=(3, 3))
    loc = loc * loc.T  # make it symmetric
    loc[np.diag_indices(3)] = 1
    localizer = {250: loc}

    result = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=5
    )
    expected = 181.360159
    np.testing.assert_allclose(result, expected)


def test_ecov_crossvalidation_weights(random_data_5x3):

    # trivial localizer
    localizer = {250: np.diag(np.ones(3))}

    weights = np.array([1, 1, 1, 1, 1])
    result = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=2
    )
    expected = 204.5516663440938
    np.testing.assert_allclose(result, expected)

    weights = np.array([0.5, 0.5, 0.5, 1, 1])
    result = _ecov_crossvalidation(
        250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=2
    )
    expected = 187.682058
    np.testing.assert_allclose(result, expected)


def test_ecov_crossvalidation_singular(random_data_5x3):

    weights = np.array([1, 1, 1, 1, 1])

    # trivial localizer
    localizer = {250: np.ones((3, 3))}

    with pytest.warns(LinAlgWarning, match="Singular matrix"):
        result = _ecov_crossvalidation(
            250, data=random_data_5x3, weights=weights, localizer=localizer, k_folds=2
        )
    expected = float("inf")
    np.testing.assert_allclose(result, expected)


def test_get_neg_loglikelihood(random_data_5x3):

    covariance = np.cov(random_data_5x3, rowvar=False)

    weights = np.full(5, fill_value=1)
    result = _get_neg_loglikelihood(random_data_5x3, covariance, weights)
    expected = 343.29088073
    np.testing.assert_allclose(result, expected)

    weights = np.array([0.5, 0.2, 0.3, 0.7, 1])
    result = _get_neg_loglikelihood(random_data_5x3, covariance, weights)
    expected = 340.387267
    np.testing.assert_allclose(result, expected)


def test_get_neg_loglikelihood_singular(random_data_5x3):

    # select data that leads to singular covariance matrix
    data = random_data_5x3[1::2]
    covariance = np.cov(data, rowvar=False)

    with pytest.raises(np.linalg.LinAlgError):
        _get_neg_loglikelihood(random_data_5x3, covariance, None)


def test_adjust_ecov_ar1_np_errors():

    cov = np.ones((3, 3))
    ar_coefs = np.ones((3, 1))
    with pytest.raises(ValueError, match="`ar_coefs` must be 1D"):
        _adjust_ecov_ar1_np(cov, ar_coefs)

    ar_coefs = np.ones(4)
    with pytest.raises(ValueError, match="`ar_coefs` must be 1D"):
        _adjust_ecov_ar1_np(cov, ar_coefs)


def test_adjust_ecov_ar1_np(random_data_5x3):

    ar_coefs = np.random.randn(3)
    cov = np.cov(random_data_5x3, rowvar=False)
    result = _adjust_ecov_ar1_np(cov, ar_coefs)

    expected = np.array(
        [
            [0.005061, -0.00323, -0.010508],
            [-0.00323, 0.026648, -0.011914],
            [-0.010508, -0.011914, 0.099646],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=1e-6)
