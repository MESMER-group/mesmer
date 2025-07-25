import numpy as np

from mesmer.mesmer_x import Expression
from mesmer.mesmer_x._distrib_checks import _validate_coefficients


def test_validate_coefficients():
    n = 10

    rng = np.random.default_rng(42)
    expr = Expression(
        "truncnorm(loc=c1 * __tas__, scale=c2, a=-10, b=10)",
        expr_name="exp1",
        boundaries_coeffs={"c1": (0.0, 1.0), "c2": (0.0, 1.0)},
        boundaries_params={"loc": (0.0, 1.0), "scale": (0.0, 1.0)},
    )

    pred = {"tas": np.ones(n)}
    c1, c2 = 1, 0.1
    targ = rng.normal(loc=c1, scale=c2, size=n)

    thresh = 1.0e-4

    # Test with valid coefficients
    coeffs = [c1, c2]

    coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params = (
        _validate_coefficients(expr, pred, targ, coeffs, thresh)
    )
    assert coeffs_in_bounds is True
    assert params_in_bounds is True
    assert params_in_support is True
    assert test_proba is np.True_
    np.testing.assert_equal(params["loc"], np.ones(n))  # type: ignore
    assert params["scale"] == c2  # type: ignore

    # Test with coefficients out of bounds
    coeffs_out_of_bounds = [1.5, -0.5]
    coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params = (
        _validate_coefficients(expr, pred, targ, coeffs_out_of_bounds, thresh)
    )

    assert coeffs_in_bounds is False
    assert params_in_bounds is False
    assert params_in_support is False
    assert test_proba is False
    assert params == {}

    # Test with preds that lead to out of bounds params
    pred_out_of_bounds = {"tas": np.arange(n)}
    coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params = (
        _validate_coefficients(expr, pred_out_of_bounds, targ, coeffs, thresh)
    )

    assert coeffs_in_bounds is True
    assert params_in_bounds is False
    assert params_in_support is False
    assert test_proba is False
    np.testing.assert_equal(params["loc"], np.arange(n))  # type: ignore
    assert params["scale"] == c2  # type: ignore

    # Test with targ that lies outside of distrib support
    targ_out_of_support = rng.uniform(low=-20 * c2, high=20 * c2, size=n)
    coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params = (
        _validate_coefficients(expr, pred, targ_out_of_support, coeffs, thresh)
    )

    assert coeffs_in_bounds is True
    assert params_in_bounds is True
    assert params_in_support is False
    assert test_proba is False
    np.testing.assert_equal(params["loc"], np.ones(n))  # type: ignore
    assert params["scale"] == c2  # type: ignore

    # Test with target that leads to low probability values
    targ_low_prob = rng.normal(loc=c1, scale=c2 * 5, size=n)
    coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params = (
        _validate_coefficients(expr, pred, targ_low_prob, coeffs, thresh)
    )
    assert coeffs_in_bounds is True
    assert params_in_bounds is True
    assert params_in_support is True
    assert test_proba is np.False_
    np.testing.assert_equal(params["loc"], np.ones(n))  # type: ignore
    assert params["scale"] == c2  # type: ignore

    # if thresh is none, test_proba is always True
    thresh_none = None
    coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params = (
        _validate_coefficients(expr, pred, targ_low_prob, coeffs, thresh_none)
    )
    assert coeffs_in_bounds is True
    assert params_in_bounds is True
    assert params_in_support is True
    assert test_proba is True
    np.testing.assert_equal(params["loc"], np.ones(n))  # type: ignore
    assert params["scale"] == c2  # type: ignore
