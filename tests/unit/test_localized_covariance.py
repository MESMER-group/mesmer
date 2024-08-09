import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mesmer
from mesmer.core.utils import LinAlgWarning, _check_dataarray_form
from mesmer.stats._localized_covariance import (
    _adjust_ecov_ar1_np,
    _ecov_crossvalidation,
    _find_localized_empirical_covariance_np,
    _get_neg_loglikelihood,
)


@pytest.fixture
def random_data_5x3():

    np.random.seed(0)
    data = np.random.rand(5, 3)
    return data


def get_random_data(n_samples, n_gridpoints):

    np.random.seed(0)
    data = np.random.rand(n_samples, n_gridpoints)

    return xr.DataArray(data, dims=("samples", "cell"))


def get_localizer_dict(n_gridpoints, as_dataarray):

    localizer = dict()
    dims = ("cell_i", "cell_j")
    for i, crosscov in enumerate(np.arange(0, 1.01, 0.1)):

        loc = np.full((n_gridpoints, n_gridpoints), fill_value=crosscov)
        loc[np.diag_indices(n_gridpoints)] = 1
        loc = xr.DataArray(loc, dims=dims) if as_dataarray else loc
        localizer[i] = loc

    return localizer


def get_weights(n_samples):

    weights = np.full(n_samples, 1)
    return xr.DataArray(weights, dims="samples")


@pytest.mark.filterwarnings("ignore:First element is local minimum.")
def test_find_localized_empirical_covariance():

    n_samples = 20
    n_gridpoints = 3

    data = get_random_data(n_samples, n_gridpoints)
    localizer = get_localizer_dict(n_gridpoints, as_dataarray=True)
    weights = get_weights(n_samples)

    required_form = {
        "ndim": 2,
        "required_dims": ("cell_i", "cell_j"),
        "shape": (n_gridpoints, n_gridpoints),
    }

    result = mesmer.stats.find_localized_empirical_covariance(
        data, weights, localizer, dim="samples", k_folds=2
    )

    assert result.localization_radius == 0
    _check_dataarray_form(result.covariance, "covariance", **required_form)
    _check_dataarray_form(result.covariance, "localized_covariance", **required_form)

    # ensure it works if data is transposed
    result = mesmer.stats.find_localized_empirical_covariance(
        data.T, weights, localizer, dim="samples", k_folds=3
    )

    assert result.localization_radius == 1
    _check_dataarray_form(result.covariance, "covariance", **required_form)
    _check_dataarray_form(
        result.localized_covariance, "localized_covariance", **required_form
    )

    # ensure can pass equal_dim_suffixes
    result = mesmer.stats.find_localized_empirical_covariance(
        data,
        weights,
        localizer,
        dim="samples",
        k_folds=3,
        equal_dim_suffixes=(":j", ":i"),
    )

    required_form["required_dims"] = ("cell:j", "cell:i")

    assert result.localization_radius == 1
    _check_dataarray_form(result.covariance, "covariance", **required_form)
    _check_dataarray_form(
        result.localized_covariance, "localized_covariance", **required_form
    )


@pytest.mark.filterwarnings("ignore:First element is local minimum.")
def test_find_localized_empirical_covariance_monthly():

    n_samples = 20 * 12
    n_gridpoints = 60
    time = pd.date_range("2000-01-01", periods=n_samples, freq="MS")

    data = get_random_data(n_samples, n_gridpoints)
    data = data.assign_coords({"samples": time})

    localizer = get_localizer_dict(n_gridpoints, as_dataarray=True)

    weights = get_weights(n_samples)
    weights = weights.assign_coords({"samples": time})

    required_form = {
        "ndim": 3,
        "required_dims": ("month", "cell_i", "cell_j"),
        "shape": (12, n_gridpoints, n_gridpoints),
    }

    result = mesmer.stats.find_localized_empirical_covariance_monthly(
        data, weights, localizer, dim="samples", k_folds=2
    )

    np.testing.assert_equal(result.localization_radius.values, np.zeros(12))
    _check_dataarray_form(result.covariance, "covariance", **required_form)
    _check_dataarray_form(
        result.localized_covariance, "localized_covariance", **required_form
    )

    # ensure it works if data is transposed
    result = mesmer.stats.find_localized_empirical_covariance_monthly(
        data.T, weights, localizer, dim="samples", k_folds=3
    )

    np.testing.assert_equal(result.localization_radius.values, np.zeros(12))
    _check_dataarray_form(result.covariance, "covariance", **required_form)
    _check_dataarray_form(
        result.localized_covariance, "localized_covariance", **required_form
    )

    # ensure can pass equal_dim_suffixes
    result = mesmer.stats.find_localized_empirical_covariance_monthly(
        data,
        weights,
        localizer,
        dim="samples",
        k_folds=3,
        equal_dim_suffixes=(":j", ":i"),
    )

    required_form["required_dims"] = ("cell:j", "cell:i")

    np.testing.assert_equal(result.localization_radius.values, np.zeros(12))
    _check_dataarray_form(result.covariance, "covariance", **required_form)
    _check_dataarray_form(
        result.localized_covariance, "localized_covariance", **required_form
    )


@pytest.mark.filterwarnings("ignore:First element is local minimum.")
def test_find_localized_empirical_covariance_np():

    n_samples = 20
    n_gridpoints = 3

    data = get_random_data(n_samples, n_gridpoints).values
    localizer = get_localizer_dict(n_gridpoints, as_dataarray=False)
    weights = get_weights(n_samples).values

    result, cov, loc_cov = _find_localized_empirical_covariance_np(
        data, weights, localizer, k_folds=2
    )
    expected = 0
    assert result == expected
    assert cov.shape == (n_gridpoints, n_gridpoints)
    assert loc_cov.shape == (n_gridpoints, n_gridpoints)
    np.testing.assert_allclose(np.diag(cov), np.diag(loc_cov))

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


def normal_data_50x30(covariance):
    rng = np.random.default_rng(0)
    data = rng.multivariate_normal(
        mean=np.zeros(30),
        cov=covariance,
        size=[50],
        method="eigh",
    ).reshape(50, 30)
    return data


def test_find_localized_empirical_covariance_method(random_data_5x3):
    cov = np.full((30, 30), fill_value=0.1)
    data = normal_data_50x30(cov)
    weights = np.full(50, fill_value=1)

    localizer = {100: np.ones(30), 200: np.eye(30), 300: np.eye(30)}
    with pytest.warns(LinAlgWarning, match="Singular matrix"):
        result, __, __ = _find_localized_empirical_covariance_np(
            data, weights, localizer, 3
        )
    assert result == 200


def test_ecov_crossvalidation_k_folds(random_data_5x3):

    weights = np.array([1, 1, 1, 1, 1])

    # trivial localizer
    localizer = {250: np.diag(np.ones(3))}
    method = "cholesky"

    result, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=2,
    )
    expected = 204.5516663440938
    np.testing.assert_allclose(result, expected)

    result, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=3,
    )
    expected = 183.32294464558134
    np.testing.assert_allclose(result, expected)

    # there is a maximum of 5 folds for 5 samples -> same result for larger k_folds
    result5, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=5,
    )
    result6, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=6,
    )

    np.testing.assert_allclose(result5, result6)


def test_ecov_crossvalidation_localizer(random_data_5x3):

    weights = np.array([1, 1, 1, 1, 1])

    # trivial localizer 1
    localizer = {250: np.diag(np.ones(3))}
    method = "cholesky"

    result, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=5,
    )
    expected = 133.975629
    np.testing.assert_allclose(result, expected)

    # trivial localizer 2
    localizer = {250: np.ones((3, 3))}

    result, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=5,
    )
    expected = 985.073313
    np.testing.assert_allclose(result, expected)

    # nontrivial localizer (symmetric and diag == 1) between 0..1
    np.random.seed(0)
    loc = np.random.uniform(size=(3, 3))
    loc = loc * loc.T  # make it symmetric
    loc[np.diag_indices(3)] = 1
    localizer = {250: loc}

    result, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=5,
    )
    expected = 181.360159
    np.testing.assert_allclose(result, expected)


def test_ecov_crossvalidation_weights(random_data_5x3):

    # trivial localizer
    localizer = {250: np.diag(np.ones(3))}
    method = "cholesky"

    weights = np.array([1, 1, 1, 1, 1])
    result, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=2,
    )
    expected = 204.5516663440938
    np.testing.assert_allclose(result, expected)

    weights = np.array([0.5, 0.5, 0.5, 1, 1])
    result, _ = _ecov_crossvalidation(
        250,
        method,
        data=random_data_5x3,
        weights=weights,
        localizer=localizer,
        k_folds=2,
    )
    expected = 187.682058
    np.testing.assert_allclose(result, expected)


def test_ecov_crossvalidation_singular(random_data_5x3):

    weights = np.array([1, 1, 1, 1, 1])
    method = "cholesky"

    # trivial localizer
    localizer = {250: np.ones((3, 3))}

    with pytest.warns(LinAlgWarning, match="Singular matrix"):
        result, method = _ecov_crossvalidation(
            250,
            method,
            data=random_data_5x3,
            weights=weights,
            localizer=localizer,
            k_folds=2,
        )
    expected = float("inf")
    np.testing.assert_allclose(result, expected)
    assert method == "eigh"


def test_get_neg_loglikelihood(random_data_5x3):

    covariance = np.cov(random_data_5x3, rowvar=False)

    weights = np.full(5, fill_value=1)
    result = _get_neg_loglikelihood(
        random_data_5x3, covariance, weights, method="cholesky"
    )
    expected = 343.29088073
    np.testing.assert_allclose(result, expected)

    # test with non-uniform weights
    weights = np.array([0.5, 0.2, 0.3, 0.7, 1])
    result = _get_neg_loglikelihood(
        random_data_5x3, covariance, weights, method="cholesky"
    )
    expected = 340.387267
    np.testing.assert_allclose(result, expected)

    # test if method='eigh' gives same result
    result = _get_neg_loglikelihood(random_data_5x3, covariance, weights, method="eigh")
    np.testing.assert_allclose(result, expected)


def test_get_neg_loglikelihood_singular(random_data_5x3):

    # select data that leads to singular covariance matrix
    data = random_data_5x3[1::2]
    covariance = np.cov(data, rowvar=False)
    weights = np.full(2, fill_value=1)

    with pytest.raises(np.linalg.LinAlgError):
        _get_neg_loglikelihood(data, covariance, weights, method="cholesky")

    # works with eigh
    _get_neg_loglikelihood(data, covariance, weights, method="eigh")


def test_adjust_ecov_ar1_np_errors():

    cov = np.ones((3, 3))
    ar_coefs = np.ones((3, 2))
    with pytest.raises(ValueError, match="`ar_coefs` must be 1D"):
        _adjust_ecov_ar1_np(cov, ar_coefs)

    ar_coefs = np.ones(4)
    with pytest.raises(ValueError, match=".*have length equal"):
        _adjust_ecov_ar1_np(cov, ar_coefs)


@pytest.mark.parametrize("shape", [(3,), (3, 1), (1, 3)])
def test_adjust_ecov_ar1_np(random_data_5x3, shape):

    ar_coefs = np.random.randn(*shape)
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


@pytest.mark.parametrize(
    "shape, dims",
    [((3,), "cells"), ((3, 1), ("cells", "lags")), ((1, 3), ("lags", "cells"))],
)
def test_adjust_covariance_ar1(random_data_5x3, shape, dims):

    ar_coefs = np.random.randn(*shape)
    ar_coefs = xr.DataArray(ar_coefs, dims=dims)

    cov = np.cov(random_data_5x3, rowvar=False)
    cov = xr.DataArray(cov, dims=("cell_i", "cell_j"))

    result = mesmer.stats.adjust_covariance_ar1(cov, ar_coefs)

    expected = np.array(
        [
            [0.005061, -0.00323, -0.010508],
            [-0.00323, 0.026648, -0.011914],
            [-0.010508, -0.011914, 0.099646],
        ]
    )

    expected = xr.DataArray(expected, dims=("cell_i", "cell_j"))
    xr.testing.assert_allclose(result, expected, atol=1e-6)
