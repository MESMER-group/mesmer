import numpy as np
import pytest
from scipy.stats import genextreme

from mesmer.mesmer_x import Expression, distrib_cov


def test_first_guess_standard_normal():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    targ = rng.normal(loc=0, scale=1, size=n)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)

    dist.find_fg()
    result = dist.fg_coeffs

    np.testing.assert_allclose(result, [0.0, 1.0], atol=0.02)


def test_first_guess_provided():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    targ = rng.normal(loc=0, scale=1, size=n)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")
    first_guess = [0.0, 1.0]

    dist = distrib_cov(targ, {"tas": pred}, expression, first_guess=first_guess)

    dist.find_fg()
    result = dist.fg_coeffs
    expected = np.array([-0.005093, 1.015267])

    np.testing.assert_allclose(result, expected)


def test_first_guess_dist_more_params():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)

    loc = 0
    scale = 1
    shape = 0.5
    # distribution with loc, scale, and shape parameters
    targ = genextreme.rvs(c=shape, loc=loc, scale=scale, size=n, random_state=rng)

    expression = Expression("genextreme(loc=c1, scale=c2, c=c3)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.find_fg()
    result = dist.fg_coeffs
    expected = [loc, scale, shape]

    # test right order of magnitude
    np.testing.assert_allclose(result, expected, atol=0.2, rtol=0.2)

    # any difference if we provide a first guess?
    dist2 = distrib_cov(
        targ, {"tas": pred}, expression, first_guess=[loc, scale, shape]
    )
    dist2.find_fg()
    result2 = dist.fg_coeffs
    np.testing.assert_equal(result2, result)  # No


def test_first_guess_with_bounds():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)

    loc = 0
    loc_bounds = (-1, 1)
    scale = 1
    scale_bounds = (0.5, 1.5)

    targ = rng.normal(loc=loc, scale=scale, size=n)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds}
    dist = distrib_cov(
        targ, {"tas": pred}, expression, boundaries_coeffs=boundaries_coeffs
    )

    dist.find_fg()
    result = dist.fg_coeffs

    # test result is within bounds
    assert loc_bounds[0] <= result[0] <= loc_bounds[1]
    assert scale_bounds[0] <= result[1] <= scale_bounds[1]
    expected = np.array([-0.005093, 1.015267])
    np.testing.assert_allclose(result, expected)

    # test with wrong bounds
    scale_bounds_wrong = (0.5, 0.8)
    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds_wrong}
    dist = distrib_cov(
        targ, {"tas": pred}, expression, boundaries_coeffs=boundaries_coeffs
    )

    dist.find_fg()
    result = dist.fg_coeffs
    # still finds a fg because we do not enforce the bounds on the fg
    # however the fg is significantly worse on the param with the wrong bounds
    # in contrast to the above this also runs step 6: fit on CDF or LL^n -> implications?
    expected = np.array([-0.016552, 1.520612])
    np.testing.assert_allclose(result, expected)

    # fails if we enforce the bounds
    options_solver = {"fg_with_global_opti": True}
    dist = distrib_cov(
        targ,
        {"tas": pred},
        expression,
        boundaries_coeffs=boundaries_coeffs,
        options_solver=options_solver,
    )

    with pytest.raises(ValueError, match="Global optimization for first guess failed,"):
        dist.find_fg()

    # when does step 7 actually succeed?
    # when we do enforce a global optimum but the bounds are good
    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds}

    dist = distrib_cov(
        targ,
        {"tas": pred},
        expression,
        boundaries_coeffs=boundaries_coeffs,
        options_solver=options_solver,
    )
    dist.find_fg()
    result = dist.fg_coeffs
    expected = np.array([-0.005093, 1.015267])
    np.testing.assert_allclose(result, expected)
