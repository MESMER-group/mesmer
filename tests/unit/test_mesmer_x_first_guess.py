import numpy as np
import pytest
from scipy.stats import genextreme

from mesmer.mesmer_x import Expression, distrib_cov


def test_first_guess_standard_normal():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    targ = rng.normal(loc=0, scale=1, size=n)

    expression = Expression("norm(loc=c1, scale=c3)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)

    dist.find_fg()
    result = dist.fg_coeffs

    np.testing.assert_allclose(result, [0.0, 1.0], atol=0.02)


def test_first_guess_standard_normal_including_pred():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(1, n, n)
    c1 = 2.0
    c2 = 5.0
    c3 = 1.0
    targ = rng.normal(loc=c1 * pred + c2, scale=c3, size=n)

    expression = Expression("norm(loc=c1*__tas__+c2, scale=c3)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)

    dist.find_fg()
    result = dist.fg_coeffs

    np.testing.assert_allclose(result, [c1, c2, c3], rtol=0.1)


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
    expected = np.array([-0.005093822, 1.015267311])

    np.testing.assert_allclose(result, expected, rtol=1e-5)


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
    expected = np.array([-0.005093813, 1.015267311])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

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
    expected = np.array([-0.016552528, 1.520612114])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

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
    expected = np.array([-0.005093817, 1.015267298])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.xfail(reason="https://github.com/MESMER-group/mesmer/issues/581")
def test_fg_func_deriv01():
    # test that for a rather smooth target, fg_func_deriv01 returns small loss for the right coefficients
    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(0, n - 1, n)
    c1 = 2.0
    c2 = 0.1
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)

    expression = Expression("norm(loc=c1*__tas__, scale=c2)", expr_name="exp1")
    dist = distrib_cov(targ, {"tas": pred}, expression)

    smooth_targ = dist._smooth_data(targ)
    # smooth_targ = scipy.ndimage.uniform_filter1d(targ, 10)
    # from statsmodels.nonparametric.smoothers_lowess import lowess
    # smooth_targ = lowess(
    #     targ, np.arange(n), frac=10 / n, return_sorted=False
    # )  # works best

    mean_smooth_targ, std_smooth_targ = np.mean(smooth_targ), np.std(smooth_targ)

    mean_minus_one_std = mean_smooth_targ - std_smooth_targ
    mean_plus_one_std = mean_smooth_targ + std_smooth_targ

    ind_targ_low = np.where(smooth_targ < mean_minus_one_std)[0]
    ind_targ_high = np.where(smooth_targ > mean_plus_one_std)[0]

    mean_low_preds = {
        p: np.mean(dist.data_pred[p][ind_targ_low]) for p in dist.data_pred
    }
    mean_high_preds = {
        p: np.mean(dist.data_pred[p][ind_targ_high]) for p in dist.data_pred
    }

    mean_low_targs = np.mean(smooth_targ[ind_targ_low])
    mean_high_targs = np.mean(smooth_targ[ind_targ_high])

    deriv_targ = {
        p: (mean_high_targs - mean_low_targs) / (mean_high_preds[p] - mean_low_preds[p])
        for p in dist.data_pred
    }

    result = dist._fg_fun_deriv01(
        [c1, c2], mean_high_preds, mean_low_preds, deriv_targ, mean_smooth_targ
    )

    np.testing.assert_allclose(result, 0, atol=1e-5)


def test_fg_fun_loc():
    # test for noiseless data, loss = 0
    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(0, n - 1, n)
    c1 = 2
    c2 = 0
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)

    expression = Expression("norm(loc=c1*__tas__, scale=c2)", expr_name="exp1")
    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.fg_coeffs = [0, 0]
    dist.fg_ind_loc = np.array([0])

    result = dist._fg_fun_loc(c1, targ)
    np.testing.assert_equal(result, 0)


def test_fg_fun_sca():
    # test for noiseless data, loss = 0
    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(0, n - 1, n)
    c1 = 2
    c2 = 1
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)

    expression = Expression("norm(loc=c1*__tas__, scale=c2)", expr_name="exp1")
    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.fg_coeffs = [2, 0]
    dist.fg_ind_sca = np.array([1])

    result = dist._fg_fun_sca(c2)
    np.testing.assert_allclose(result, 0, atol=0.4)