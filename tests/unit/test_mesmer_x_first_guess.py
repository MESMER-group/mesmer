import numpy as np
import pytest
from scipy.stats import beta, binom, gamma, genextreme, hypergeom, laplace, truncnorm

import mesmer.mesmer_x
from mesmer.mesmer_x.train_l_distrib_mesmerx import _finite_difference, _smooth_data


def test_first_guess_standard_normal():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    targ = rng.normal(loc=0, scale=1, size=n)

    expression = mesmer.mesmer_x.Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    np.testing.assert_allclose(result, [0.0, 1.0], atol=0.02)


def test_first_guess_standard_normal_including_pred():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(1, n, n)
    c1 = 2.0
    c2 = 5.0
    c3 = 1.0
    targ = rng.normal(loc=c1 * pred + c2, scale=c3, size=n)

    expression = mesmer.mesmer_x.Expression(
        "norm(loc=c1*__tas__+c2, scale=c3)", expr_name="exp1"
    )

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    np.testing.assert_allclose(result, [c1, c2, c3], rtol=0.03)


@pytest.mark.parametrize("first_guess", [[1.0, 1.0], [1.0, 2.0], [-1, 0.5], [10, 7]])
def test_first_guess_provided(first_guess):
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc, scale = 1, 1
    targ = rng.normal(loc=loc, scale=scale, size=n)

    expression = mesmer.mesmer_x.Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

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

    expression = mesmer.mesmer_x.Expression(
        "genextreme(loc=c1, scale=c2, c=c3)", expr_name="exp1"
    )

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    expected = [loc, scale, shape]

    # test right order of magnitude
    np.testing.assert_allclose(result, expected, rtol=0.4)

    # any difference if we provide a first guess?
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, [loc, scale, shape], None
    )
    result2 = fg_mx._find_fg_np(pred, targ, weights)
    # NOTE: leads to the same result as without first guess
    np.testing.assert_allclose(result2, result, rtol=1.0e-3)


def test_first_guess_GEV_including_pred():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.arange(n)

    c1 = 2.0
    scale = 1.0
    shape = 0.5
    # distribution with loc, scale, and shape parameters
    targ = genextreme.rvs(c=shape, loc=pred**c1, scale=scale, size=n, random_state=rng)

    expression = mesmer.mesmer_x.Expression(
        "genextreme(loc=__tas__**c1, scale=c2, c=c3)", expr_name="exp1"
    )

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

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
    expression = mesmer.mesmer_x.Expression(
        "truncnorm(loc=c1, scale=c2, a=c3, b=c4)", expr_name="exp1"
    )

    # NOTE: this is an interesting case to test because the fact that the distribution is truncated
    # makes the optimization for the scale biased in step 3: here we fit the scale to be close to the
    # deviations from the location, which is smaller for the truncated normal than for the not truncated normal
    # thus the first fit for the scale is too small, then the fit for the bounds is too large, but step 5
    # does a good job in fixing this.

    # needs first guess different from 0, 0 for a and b, degenerate otherwise, also degenerate if a == b

    first_guess = [0.0, 1.0, -1, 2.0]

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, first_guess, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    expected = [loc, scale, a, b]

    np.testing.assert_allclose(result, expected, rtol=0.3)


def test_fg_binom():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n) * 0.5

    n_trials = 10
    p = 0.5
    targ = binom.rvs(n=n_trials, p=p * pred, random_state=rng, size=n)

    expression = mesmer.mesmer_x.Expression(
        "binom(n=c1, p=c2*__tas__, loc=0)", expr_name="exp1"
    )

    first_guess = [11, 0.4]

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, first_guess, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

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

    expression = mesmer.mesmer_x.Expression(
        "hypergeom(M=c1, n=c2, N=c3*__tas__, loc=0)", expr_name="exp1"
    )

    first_guess = [99, 9, 1]

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, first_guess, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    expected = [M, n_draws, n_success]

    np.testing.assert_allclose(result, expected, rtol=0.1)

    # test with bounds
    # NOTE: here we add a bound on c2 that is smaller than the negative loglikelihood fit (step 5)
    # this leads to `test_coeff` being False and another fit with NLL*4 is being done (step 6)
    # this leads to coverage of _fg_fun_LL_n() also for the discrete case
    # NOTE: sadly this does not improve the fit

    first_guess = [99, 9, 1]
    boundaries_coeffs = {"c1": (80, 110), "c2": (5, 10), "c3": (1, 3)}

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expression, 1.0e-9, None, boundaries_coeffs
    )
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, first_guess, None
    )
    result_with_bounds = fg_mx._find_fg_np(pred, targ, weights)

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

    expression = mesmer.mesmer_x.Expression(
        "beta(loc=0, scale=1, a=c3, b=c4)", expr_name="exp1"
    )

    # we need a first guess here because our default first guess is zeros, which leads
    # to a degenerate distribution in the case of a beta distribution
    first_guess = [1.0, 1.0]
    options_solver = {"fg_with_global_opti": True}

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(
        expression, tests_mx, None, options_solver
    )
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, first_guess, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    # NOTE: for the beta distribution the support does not change for loc = 0 and scale = 1
    # it is always (0, 1), thus the optimization with _fg_fun_others does not do anything
    # NOTE: Step 7 (fg_with_global_opti) leads to a improvement of the first guess at the 6th digit after the comma, i.e. very small
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

    expression = mesmer.mesmer_x.Expression(
        "gamma(loc=0, scale=1, a=c1)", expr_name="exp1"
    )

    # we need a first guess different from zero for gamma distribution
    first_guess = [1.0]
    options_solver = {"fg_with_global_opti": True}

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(
        expression, tests_mx, None, options_solver
    )
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, first_guess, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    expected = [a]

    np.testing.assert_allclose(result, expected, rtol=0.02)


def test_fg_fun_scale_laplace():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc = 2
    scale = 1
    targ = laplace.rvs(loc=loc, scale=scale, size=n, random_state=rng)

    expression = mesmer.mesmer_x.Expression(
        "laplace(loc=c1, scale=c2)", expr_name="exp1"
    )

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

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

    expression = mesmer.mesmer_x.Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds}

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expression, 1.0e-9, None, boundaries_coeffs
    )
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    # test result is within bounds
    assert loc_bounds[0] <= result[0] <= loc_bounds[1]
    assert scale_bounds[0] <= result[1] <= scale_bounds[1]
    expected = np.array([loc, scale])
    np.testing.assert_allclose(result, expected, atol=0.1)

    # test with bounds outside true value
    scale_bounds_wrong = (0.5, 0.8)
    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds_wrong}

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expression, 1.0e-9, None, boundaries_coeffs
    )
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    expected = np.array([-0.016552528, 1.520612114])
    np.testing.assert_allclose(result, expected, atol=0.5)
    # ^ still finds a fg because we do not enforce the bounds on the fg
    # however the fg is significantly worse on the param with the wrong bounds
    # in contrast to the above the test below also runs step 6: fit on LL^n -> implications?

    # fails if we enforce the bounds
    options_solver = {"fg_with_global_opti": True}

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expression, 1.0e-9, None, boundaries_coeffs
    )
    optim_mx = mesmer.mesmer_x.distrib_optimizer(
        expression, tests_mx, None, options_solver
    )
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    with pytest.raises(ValueError, match="Global optimization for first guess failed,"):
        result = fg_mx._find_fg_np(pred, targ, weights)

    # when does step 7 actually succeed?
    # when we do enforce a global optimum but the bounds are good
    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds}

    weights = mesmer.mesmer_x.get_weights_uniform(targ, "tas", None)
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expression, 1.0e-9, None, boundaries_coeffs
    )
    optim_mx = mesmer.mesmer_x.distrib_optimizer(
        expression, tests_mx, None, options_solver
    )
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    result = fg_mx._find_fg_np(pred, targ, weights)

    expected = np.array([loc, scale])
    np.testing.assert_allclose(result, expected, atol=0.1)


@pytest.mark.xfail(reason="https://github.com/MESMER-group/mesmer/issues/581")
def test_fg_func_deriv01():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.arange(n)
    c1 = 2.0
    c2 = 0.1
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)

    expression = mesmer.mesmer_x.Expression(
        "norm(loc=c1*__tas__, scale=c2)", expr_name="exp1"
    )

    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )

    fg_mx.predictor_dim = fg_mx.expr_fit.inputs_list
    fg_mx.data_pred = {"tas": pred}
    fg_mx.data_targ = targ

    # smooting to help with location & scale
    fg_mx.l_smooth = 5
    fg_mx.smooth_targ = _smooth_data(fg_mx.data_targ, length=fg_mx.l_smooth)
    fg_mx.dev_smooth_targ = (
        fg_mx.data_targ[fg_mx.l_smooth : -fg_mx.l_smooth] - fg_mx.smooth_targ
    )
    fg_mx.smooth_pred = {
        pp: _smooth_data(fg_mx.data_pred[pp], length=fg_mx.l_smooth)
        for pp in fg_mx.predictor_dim
    }

    # preparation of coefficients
    fg_mx.fg_coeffs = np.zeros(fg_mx.n_coeffs)
    loc_coeffs = fg_mx.expr_fit.coefficients_dict.get("loc", [])
    fg_mx.fg_ind_loc = np.array(
        [fg_mx.expr_fit.coefficients_list.index(c) for c in loc_coeffs]
    )

    # preparation of derivatives
    m_smooth_targ, s_smooth_targ = np.mean(fg_mx.smooth_targ), np.std(fg_mx.smooth_targ)
    ind_targ_low = np.where(fg_mx.smooth_targ < m_smooth_targ - s_smooth_targ)[0]
    ind_targ_high = np.where(fg_mx.smooth_targ > m_smooth_targ + s_smooth_targ)[0]
    mean_high_preds = {
        pp: np.mean(fg_mx.smooth_pred[pp][ind_targ_high], axis=0)
        for pp in fg_mx.predictor_dim
    }
    mean_low_preds = {
        pp: np.mean(fg_mx.smooth_pred[pp][ind_targ_low], axis=0)
        for pp in fg_mx.predictor_dim
    }

    derivative_targ = {
        pp: _finite_difference(
            np.mean(fg_mx.smooth_targ[ind_targ_high]),
            np.mean(fg_mx.smooth_targ[ind_targ_low]),
            mean_high_preds[pp],
            mean_low_preds[pp],
        )
        for pp in fg_mx.predictor_dim
    }

    loss_at_toolow = fg_mx._fg_fun_deriv01(
        [c1 / 2], mean_high_preds, mean_low_preds, derivative_targ, m_smooth_targ
    )
    loss_at_truesolution = fg_mx._fg_fun_deriv01(
        [c1], mean_high_preds, mean_low_preds, derivative_targ, m_smooth_targ
    )
    loss_at_toohigh = fg_mx._fg_fun_deriv01(
        [c1 * 2], mean_high_preds, mean_low_preds, derivative_targ, m_smooth_targ
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

    expression = mesmer.mesmer_x.Expression(
        "norm(loc=c1*__tas__, scale=c2)", expr_name="exp1"
    )

    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )

    fg_mx.predictor_dim = fg_mx.expr_fit.inputs_list
    fg_mx.data_pred = {"tas": pred}
    fg_mx.smooth_targ = _smooth_data(targ)
    fg_mx.smooth_pred = {
        pp: _smooth_data(fg_mx.data_pred[pp]) for pp in fg_mx.predictor_dim
    }
    fg_mx.fg_coeffs = [0.0, 0.0]
    fg_mx.fg_ind_loc = np.array([0])

    # test local minima at true coefficients
    delta = 1.0e-4

    loss_at_toolow = fg_mx._fg_fun_loc(c1 - delta)
    loss_at_truesolution = fg_mx._fg_fun_loc(c1)
    loss_at_toohigh = fg_mx._fg_fun_loc(c1 + delta)

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

    expression = mesmer.mesmer_x.Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )

    fg_mx.data_targ = targ
    fg_mx.data_pred = {"tas": pred}
    fg_mx.l_smooth = 5
    fg_mx.smooth_targ = _smooth_data(fg_mx.data_targ, length=fg_mx.l_smooth)
    fg_mx.dev_smooth_targ = (
        fg_mx.data_targ[fg_mx.l_smooth : -fg_mx.l_smooth] - fg_mx.smooth_targ
    )
    fg_mx.fg_coeffs = [loc, scale]
    fg_mx.fg_ind_sca = np.array([1])

    result = fg_mx._fg_fun_sca(scale)
    np.testing.assert_allclose(result, 0, atol=0.1)

    # test GEV
    targ = genextreme.rvs(c=1, loc=loc, scale=scale, size=n, random_state=rng)

    expression = mesmer.mesmer_x.Expression(
        "genextreme(loc=c1, scale=c2, c=c3)", expr_name="exp1"
    )

    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )

    fg_mx.data_targ = targ
    fg_mx.data_pred = {"tas": pred}
    fg_mx.l_smooth = 5
    fg_mx.smooth_targ = _smooth_data(fg_mx.data_targ, length=fg_mx.l_smooth)
    fg_mx.dev_smooth_targ = (
        fg_mx.data_targ[fg_mx.l_smooth : -fg_mx.l_smooth] - fg_mx.smooth_targ
    )
    fg_mx.fg_coeffs = [loc, scale, 1.0]
    fg_mx.fg_ind_sca = np.array([1])

    # test local minimum at true value
    delta = 1.0  # don't get closer than this since for a GEV the variance is not necessarily close to the scale
    loss_at_toolow = fg_mx._fg_fun_sca(scale - delta)
    loss_at_truesolution = fg_mx._fg_fun_sca(scale)
    loss_at_toohigh = fg_mx._fg_fun_sca(scale + delta)

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
    expression = mesmer.mesmer_x.Expression(
        "truncnorm(loc=c1, scale=c2, a=c3, b=c4)", expr_name="exp1"
    )

    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )

    fg_mx.data_targ = targ
    fg_mx.data_pred = {"tas": pred}
    fg_mx.fg_coeffs = [loc, scale, a, b]
    fg_mx.fg_ind_others = np.array([2, 3])

    # test local minimum at true value
    delta = 1
    loss_at_toolow_a = fg_mx._fg_fun_others([a - delta, b])
    loss_at_toohigh_a = fg_mx._fg_fun_others([a + delta, b])

    loss_at_toolow_b = fg_mx._fg_fun_others([a, b - delta])
    loss_at_toohigh_b = fg_mx._fg_fun_others([a, b + delta])

    loss_at_truesolution = fg_mx._fg_fun_others([a, b])

    min_loss = np.min(
        [loss_at_toolow_a, loss_at_toohigh_a, loss_at_toolow_b, loss_at_toohigh_b]
    )

    assert min_loss >= loss_at_truesolution
