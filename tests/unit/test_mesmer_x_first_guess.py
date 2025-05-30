import numpy as np
import pytest
from scipy.stats import beta, binom, gamma, genextreme, hypergeom, laplace, truncnorm

from mesmer.mesmer_x import Expression, distrib_cov
from mesmer.mesmer_x.train_l_distrib_mesmerx import _smooth_data


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

    np.testing.assert_allclose(result, [c1, c2, c3], rtol=0.03)


@pytest.mark.parametrize("first_guess", [[1.0, 1.0], [1.0, 2.0], [-1, 0.5], [10, 7]])
def test_first_guess_provided(first_guess):
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc, scale = 1, 1
    targ = rng.normal(loc=loc, scale=scale, size=n)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression, first_guess=first_guess)

    dist.find_fg()
    result = dist.fg_coeffs

    np.testing.assert_allclose(result, [loc, scale], rtol=0.02)


@pytest.mark.parametrize("shape", [0.5, -0.5, 0.1])
def test_first_guess_GEV(shape):
    # NOTE: shape is difficult to estimate
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)

    loc = 1.0
    scale = 0.5
    # distribution with loc, scale, and shape parameters
    targ = genextreme.rvs(c=shape, loc=loc, scale=scale, size=n, random_state=rng)

    expression = Expression("genextreme(loc=c1, scale=c2, c=c3)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.find_fg()
    result = dist.fg_coeffs
    expected = [loc, scale, shape]

    # test right order of magnitude
    np.testing.assert_allclose(result, expected, rtol=0.4)

    # any difference if we provide a first guess?
    dist2 = distrib_cov(
        targ, {"tas": pred}, expression, first_guess=[loc, scale, shape]
    )
    dist2.find_fg()
    result2 = dist.fg_coeffs
    # NOTE: leads to the same result as without first guess
    np.testing.assert_equal(result2, result)


def test_first_guess_GEV_including_pred():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.arange(n)

    c1 = 2.0
    scale = 1.0
    shape = 0.5
    # distribution with loc, scale, and shape parameters
    targ = genextreme.rvs(c=shape, loc=pred**c1, scale=scale, size=n, random_state=rng)

    expression = Expression(
        "genextreme(loc=__tas__**c1, scale=c2, c=c3)", expr_name="exp1"
    )

    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.find_fg()
    result = dist.fg_coeffs
    expected = [c1, scale, shape]

    # test right order of magnitude
    np.testing.assert_allclose(result, expected, rtol=0.2)


def test_first_guess_truncnorm():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)

    loc = 1.0
    scale = 0.1
    a = -1.2  # nr of stds from loc at which to truncate
    b = 1.2
    targ = truncnorm.rvs(loc=loc, scale=scale, a=a, b=b, size=n, random_state=rng)
    expression = Expression("truncnorm(loc=c1, scale=c2, a=c3, b=c4)", expr_name="exp1")

    # NOTE: this is an interesting case to test because the fact that the distribution is truncated
    # makes the optimization for the scale biased in step 3: here we fit the scale to be close to the
    # deviations from the location, which is smaller for the truncated normal than for the not truncated normal
    # thus the first fit for the scale is too small, then the fit for the bounds is too large, but step 5
    # does a good job in fixing this.

    # needs first guess different from 0, 0 for a and b, degenerate otherwise, also degenerate if a == b
    first_guess = [0.0, 1.0, -1, 2.0]
    dist = distrib_cov(
        targ,
        {"tas": pred},
        expression,
        first_guess=first_guess,
        threshold_min_proba=None,
    )
    dist.find_fg()

    result = dist.fg_coeffs
    expected = [loc, scale, a, b]

    np.testing.assert_allclose(result, expected, rtol=0.52)


def test_fg_binom():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n) * 0.5

    n_trials = 10
    p = 0.5
    targ = binom.rvs(n=n_trials, p=p * pred, random_state=rng, size=n)

    expression = Expression("binom(n=c1, p=c2*__tas__, loc=0)", expr_name="exp1")

    first_guess = [11, 0.4]
    dist = distrib_cov(targ, {"tas": pred}, expression, first_guess=first_guess)
    dist.find_fg()
    result = dist.fg_coeffs
    expected = [n_trials, p]

    np.testing.assert_allclose(result, expected, rtol=0.2)


def test_fg_hypergeom():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.full(n, fill_value=2, dtype=int)

    M = 100
    n_draws = 10
    n_success = 2
    targ = hypergeom.rvs(M=M, n=n_draws, N=n_success * pred, random_state=rng, size=n)

    expression = Expression(
        "hypergeom(M=c1, n=c2, N=c3*__tas__, loc=0)", expr_name="exp1"
    )

    first_guess = [99, 9, 1]
    dist = distrib_cov(targ, {"tas": pred}, expression, first_guess=first_guess)
    dist.find_fg()
    result = dist.fg_coeffs  # 99, 11, 2
    expected = [M, n_draws, n_success]

    np.testing.assert_allclose(result, expected, rtol=0.1)

    # test with bounds
    # NOTE: here we add a bound on c2 that is smaller than the negative loglikelihood fit (step 5)
    # this leads to `test_coeff` being False and another fit with NLL*4 is being done (step 6)
    # this leads to coverage of _fg_fun_LL_n() also for the discrete case
    # NOTE: sadly this does not improve the fit

    boundaries_coeffs = {"c1": (80, 110), "c2": (5, 10), "c3": (1, 3)}
    dist = distrib_cov(
        targ,
        {"tas": pred},
        expression,
        first_guess=first_guess,
        boundaries_coeffs=boundaries_coeffs,
    )
    dist.find_fg()
    result_with_bounds = dist.fg_coeffs

    np.testing.assert_equal(result, result_with_bounds)


def test_first_guess_beta():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)

    a = 2
    b = 2
    loc = 0
    scale = 1
    targ = beta.rvs(a, b, loc, scale, size=n, random_state=rng)

    expression = Expression("beta(loc=0, scale=1, a=c3, b=c4)", expr_name="exp1")

    # we need a first guess here because our default first guess is zeros, which leads
    # to a degenerate distribution in the case of a beta distribution
    first_guess = [1.0, 1.0]
    options_solver = {"fg_with_global_opti": True}
    dist = distrib_cov(
        targ,
        {"tas": pred},
        expression,
        first_guess=first_guess,
        options_solver=options_solver,
    )
    dist.find_fg()

    # NOTE: for the beta distribution the support does not change for loc = 0 and scale = 1
    # it is always (0, 1), thus the optimization with _fg_fun_others does not do anything
    # NOTE: Step 7 (fg_with_global_opti) leads to a improvement of the first guess at the 6th digit after the comma, i.e. very small
    result = dist.fg_coeffs
    expected = [a, b]

    np.testing.assert_allclose(result, expected, rtol=0.5)


def test_first_guess_gamma():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)

    a = 2
    loc = 0
    scale = 1
    targ = gamma.rvs(a, loc, scale, size=n, random_state=rng)

    expression = Expression("gamma(loc=0, scale=1, a=c1)", expr_name="exp1")

    # we need a first guess different from zero for gamma distribution
    first_guess = [1.0]
    options_solver = {"fg_with_global_opti": True}
    dist = distrib_cov(
        targ,
        {"tas": pred},
        expression,
        first_guess=first_guess,
        options_solver=options_solver,
    )
    dist.find_fg()
    result = dist.fg_coeffs
    expected = [a]

    np.testing.assert_allclose(result, expected, rtol=0.02)


def test_fg_fun_scale_laplace():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc = 2
    scale = 1
    targ = laplace.rvs(loc=loc, scale=scale, size=n, random_state=rng)

    expression = Expression("laplace(loc=c1, scale=c2)", expr_name="exp1")
    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.find_fg()

    result = dist.fg_coeffs
    expected = [loc, scale]

    np.testing.assert_allclose(result, expected, rtol=0.1)


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

    # test with bounds outside true value
    scale_bounds_wrong = (0.5, 0.8)
    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds_wrong}
    dist = distrib_cov(
        targ, {"tas": pred}, expression, boundaries_coeffs=boundaries_coeffs
    )

    dist.find_fg()
    result = dist.fg_coeffs
    expected = np.array([-0.016552528, 1.520612114])
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    # ^ still finds a fg because we do not enforce the bounds on the fg
    # however the fg is significantly worse on the param with the wrong bounds
    # in contrast to the above the test below also runs step 6: fit on CDF or LL^n -> implications?

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
    rng = np.random.default_rng(0)
    n = 251
    pred = np.arange(n)
    c1 = 2.0
    c2 = 0.1
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)

    expression = Expression("norm(loc=c1*__tas__, scale=c2)", expr_name="exp1")
    dist = distrib_cov(targ, {"tas": pred}, expression)

    smooth_targ = _smooth_data(targ)

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

    loss_at_toolow = dist._fg_fun_deriv01(
        [c1 / 2, c2], mean_high_preds, mean_low_preds, deriv_targ, mean_smooth_targ
    )
    loss_at_truesolution = dist._fg_fun_deriv01(
        [c1, c2], mean_high_preds, mean_low_preds, deriv_targ, mean_smooth_targ
    )
    loss_at_toohigh = dist._fg_fun_deriv01(
        [c1 * 2, c2], mean_high_preds, mean_low_preds, deriv_targ, mean_smooth_targ
    )

    assert loss_at_toolow > loss_at_truesolution
    assert loss_at_toohigh > loss_at_truesolution

    np.testing.assert_allclose(loss_at_truesolution, 0, atol=1e-5)


def test_fg_fun_loc():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.arange(np.float64(n))
    c1 = 2.0
    c2 = 0.1
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)

    expression = Expression("norm(loc=c1*__tas__, scale=c2)", expr_name="exp1")
    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.fg_coeffs = [0.0, 0.0]
    dist.fg_ind_loc = np.array([0])

    # test local minima at true coefficients
    delta = 1.0e-4
    loss_at_toolow = dist._fg_fun_loc(c1 - delta, targ)
    loss_at_truesolution = dist._fg_fun_loc(c1, targ)
    loss_at_toohigh = dist._fg_fun_loc(c1 + delta, targ)

    assert loss_at_toolow > loss_at_truesolution
    assert loss_at_toohigh > loss_at_truesolution


def test_fg_fun_sca():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc = 0.0
    scale = 1.0

    # test normal
    targ = rng.normal(loc=loc, scale=scale, size=n)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")
    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.fg_coeffs = [loc, scale]
    dist.fg_ind_sca = np.array([1])

    result = dist._fg_fun_sca(scale)
    np.testing.assert_allclose(result, 0, atol=0.4)

    # test GEV
    targ = genextreme.rvs(c=1, loc=loc, scale=scale, size=n, random_state=rng)

    expression = Expression("genextreme(loc=c1, scale=c2, c=c3)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.fg_coeffs = [loc, scale, 1.0]
    dist.fg_ind_sca = np.array([1])

    # test local minimum at true value
    delta = 1.0  # don't get closer than this since for a GEV the variance is not necessarily close to the scale
    loss_at_toolow = dist._fg_fun_sca(scale - delta)
    loss_at_truesolution = dist._fg_fun_sca(scale)
    loss_at_toohigh = dist._fg_fun_sca(scale + delta)

    assert loss_at_toolow > loss_at_truesolution
    assert loss_at_toohigh > loss_at_truesolution


@pytest.mark.xfail(reason="https://github.com/MESMER-group/mesmer/issues/582")
def test_fg_fun_others():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc = 0
    scale = 1
    a = -1.2  # nr of stds from loc at which to truncate
    b = 1.2
    targ = truncnorm.rvs(loc=loc, scale=scale, a=a, b=b, size=n, random_state=rng)
    expression = Expression("truncnorm(loc=c1, scale=c2, a=c3, b=c4)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)
    dist.fg_coeffs = [loc, scale, a, b]
    dist.fg_ind_others = np.array([2, 3])

    # test local minimum at true value
    delta = 1
    loss_at_toolow_a = dist._fg_fun_others([a - delta, b])
    loss_at_toohigh_a = dist._fg_fun_others([a + delta, b])

    loss_at_toolow_b = dist._fg_fun_others([a, b - delta])
    loss_at_toohigh_b = dist._fg_fun_others([a, b + delta])

    loss_at_truesolution = dist._fg_fun_others([a, b])

    min_loss = np.min(
        [loss_at_toolow_a, loss_at_toohigh_a, loss_at_toolow_b, loss_at_toohigh_b]
    )
    np.testing.assert_equal(loss_at_truesolution, min_loss)
