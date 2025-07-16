import numpy as np
import pytest
import scipy as sp

from mesmer.mesmer_x import Expression, MinimizeOptions
from mesmer.mesmer_x._first_guess import (
    _finite_difference,
    _FirstGuess,
    _smooth_data,
)


@pytest.fixture
def expr():
    return Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")


def get_weights_uniform(data):
    return np.ones_like(data)


def fg_default(n_coeffs):
    return np.zeros(n_coeffs)


def test_first_guess_init_easy(expr):
    n = 251
    pred = np.arange(n)
    targ = np.random.normal(size=n)
    weights = get_weights_uniform(targ)

    fg_coeffs = fg_default(2)

    fg = _FirstGuess(
        expression=expr,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=["tas"],
    )

    assert fg.expression == expr

    assert isinstance(fg.minimize_options, MinimizeOptions)
    assert fg.minimize_options.method == "Powell"
    assert fg.minimize_options.tol is None
    assert fg.minimize_options.options is None

    assert fg.predictor_names == ["tas"]
    np.testing.assert_equal(fg.fg_coeffs, fg_coeffs)
    np.testing.assert_equal(fg.data_pred["tas"], pred)
    np.testing.assert_equal(fg.data_targ, targ)
    assert fg.l_smooth == 5
    np.testing.assert_equal(fg.smooth_pred["tas"], _smooth_data(pred, 5))
    np.testing.assert_equal(fg.smooth_targ, _smooth_data(targ, 5))
    np.testing.assert_equal(fg.smooth_targ_dev_sq, (targ[5:-5] - fg.smooth_targ) ** 2)
    np.testing.assert_equal(fg.data_weights, weights)


def test_first_guess_init_fg_ceoffs_int(expr):
    n = 251
    pred = np.arange(n).reshape(n, 1)
    targ = np.random.normal(size=n)
    weights = get_weights_uniform(targ)

    fg_coeffs = np.array([np.int16(1), np.int16(2)])

    fg = _FirstGuess(
        expression=expr,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=["tas"],
    )

    assert type(fg.fg_coeffs[0]) is np.float64
    assert type(fg.fg_coeffs[0]) is np.float64


def test_fg_init_errors_validate_data(expr):
    n = 15  # must be > 10 for smoothing
    with pytest.raises(ValueError, match="nan values in predictor data"):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones(n) * np.nan,
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="infinite values in predictor data"):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones(n) * np.inf,
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="nan values in target data"):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n) * np.nan,
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="infinite values in target data"):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n) * np.inf,
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )


def test_fg_init_errors_fg_coeffs(expr):
    n = 15
    with pytest.raises(
        ValueError, match="The provided first guess does not have the correct shape:"
    ):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones(n),
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2, 3]),
        )


def test_fg_init_errors_predictor_names(expr):
    n = 15
    with pytest.raises(
        ValueError,
        match="If data_pred is provided, predictor_names must be provided as well.",
    ):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones(n),
            predictor_names=None,
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(
        ValueError,
        match="If predictor_names is provided, data_pred must be provided as well.",
    ):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=None,
            predictor_names=["tas"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )


def test_fg_init_errors_number_of_preds(expr):
    n = 15

    with pytest.raises(ValueError, match="data_pred must be 1D or a 2D array"):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones((n, 3)),
            predictor_names=["tas", "tas2"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="data_pred must be 1D or a 2D array"):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones((n, 2, 2)),
            predictor_names=["tas", "tas2"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )

    with pytest.raises(ValueError, match="data_pred must be 1D or a 2D array"):
        _FirstGuess(
            expr,
            minimize_options=MinimizeOptions(),
            data_pred=np.ones(n),
            predictor_names=["tas", "tas2"],
            data_targ=np.ones(n),
            data_weights=np.ones(n) / n,
            first_guess=np.array([1, 2]),
        )


def test_first_guess_standard_normal():
    rng = np.random.default_rng(0)
    n = 251
    targ = rng.normal(loc=0, scale=1, size=n)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    weights = get_weights_uniform(targ)
    fg_coeffs = fg_default(2)

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=None,
    )._find_fg()

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

    weights = get_weights_uniform(targ)
    fg_coeffs = fg_default(3)

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=["tas"],
    )._find_fg()

    np.testing.assert_allclose(result, [c1, c2, c3], rtol=0.03)


@pytest.mark.parametrize("first_guess", [[1.0, 1.0], [1.0, 2.0], [-1, 0.5], [10, 7]])
def test_first_guess_provided(first_guess):
    rng = np.random.default_rng(0)
    n = 251
    loc, scale = 1, 1
    targ = rng.normal(loc=loc, scale=scale, size=n)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")

    weights = get_weights_uniform(targ)
    fg_coeffs = fg_default(2)

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=None,
    )._find_fg()

    np.testing.assert_allclose(result, [loc, scale], rtol=0.02)


@pytest.mark.parametrize("shape", [0.5, -0.5, 0.1])
def test_first_guess_GEV(shape):
    # NOTE: shape is difficult to estimate
    rng = np.random.default_rng(0)
    n = 251

    loc = 1.0
    scale = 0.5
    # distribution with loc, scale, and shape parameters
    targ = sp.stats.genextreme.rvs(
        c=shape, loc=loc, scale=scale, size=n, random_state=rng
    )

    expression = Expression("genextreme(loc=c1, scale=c2, c=c3)", expr_name="exp1")

    weights = get_weights_uniform(targ)
    fg_coeffs = fg_default(3)

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=None,
    )._find_fg()

    expected = [loc, scale, shape]

    # test right order of magnitude
    np.testing.assert_allclose(result, expected, rtol=0.4)

    # any difference if we provide a close first guess?
    result2 = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=None,
    )._find_fg()

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
    targ = sp.stats.genextreme.rvs(
        c=shape, loc=pred**c1, scale=scale, size=n, random_state=rng
    )

    expression = Expression(
        "genextreme(loc=__tas__**c1, scale=c2, c=c3)", expr_name="exp1"
    )

    weights = get_weights_uniform(targ)
    fg_coeffs = fg_default(3)

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_coeffs,
        predictor_names=["tas"],
    )._find_fg()

    expected = [c1, scale, shape]

    # test right order of magnitude
    np.testing.assert_allclose(result, expected, rtol=0.2)


def test_first_guess_truncnorm():
    rng = np.random.default_rng(0)
    n = 251

    loc = 1.0
    scale = 0.1
    a = -1.2  # nr of stds from loc at which to truncate
    b = 1.2
    targ = sp.stats.truncnorm.rvs(
        loc=loc, scale=scale, a=a, b=b, size=n, random_state=rng
    )
    weights = get_weights_uniform(targ)

    # NOTE: this is an interesting case to test because the fact that the distribution is truncated
    # makes the optimization for the scale biased in step 3: here we fit the scale to be close to the
    # deviations from the location, which is smaller for the truncated normal than for the not truncated normal
    # thus the first fit for the scale is too small, then the fit for the bounds is too large, but step 5
    # does a good job in fixing this.

    expression = Expression("truncnorm(loc=c1, scale=c2, a=c3, b=c4)", expr_name="exp1")

    # needs first guess different from 0, 0 for a and b, degenerate otherwise, also degenerate if a == b
    first_guess = [0.0, 1.0, -1, 2.0]

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=first_guess,
        predictor_names=None,
    )._find_fg()

    expected = [loc, scale, a, b]

    np.testing.assert_allclose(result, expected, rtol=0.3)


def test_fg_binom():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n) * 0.5

    n_trials = 10
    p = 0.5
    targ = sp.stats.binom.rvs(n=n_trials, p=p * pred, random_state=rng, size=n)
    weights = get_weights_uniform(targ)

    expression = Expression("binom(n=c1, p=c2*__tas__, loc=0)", expr_name="exp1")

    first_guess = [11, 0.4]
    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=first_guess,
        predictor_names=["tas"],
    )._find_fg()

    expected = [n_trials, p]

    np.testing.assert_allclose(result, expected, rtol=0.2)


def test_fg_hypergeom():
    # TODO: remove this test?

    rng = np.random.default_rng(0)
    n = 251
    pred = np.full(n, fill_value=2, dtype=int)

    M = 100
    n_draws = 10
    n_success = 2
    targ = sp.stats.hypergeom.rvs(
        M=M, n=n_draws, N=n_success * pred, random_state=rng, size=n
    )
    weights = get_weights_uniform(targ)

    expr_str = "hypergeom(M=c1, n=c2, N=c3*__tas__, loc=0)"
    expression = Expression(expr_str, expr_name="exp1")

    first_guess = [99, 9, 1]
    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=first_guess,
        predictor_names=["tas"],
    )._find_fg()

    expected = [M, n_draws, n_success]

    np.testing.assert_allclose(result, expected, rtol=0.1)

    # test with bounds
    # NOTE: here we add a bound on c2 that is smaller than the negative loglikelihood fit (step 5)
    # this leads to `test_coeff` being False and another fit with NLL*3 is being done (step 6)
    # this leads to coverage of _fg_fun_nll_cubed() also for the discrete case

    # NOTE: the bounds are not respected, because _fg_fun_nll_cubed does not enforce
    # them, see also https://github.com/MESMER-group/mesmer/issues/596
    # not sure if the result is an improvement over the one without bounds

    expected = [102.0, 13.0, 3.0]
    first_guess = [99, 9, 1]
    boundaries_coeffs = {"c1": (80, 110), "c2": (5, 10), "c3": (1, 3)}
    expression = Expression(
        expr_str, expr_name="exp1", boundaries_coeffs=boundaries_coeffs
    )

    result_with_bounds = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=first_guess,
        predictor_names=["tas"],
    )._find_fg()

    np.testing.assert_equal(expected, result_with_bounds)


def test_first_guess_beta():
    rng = np.random.default_rng(0)
    n = 251

    a = 2
    b = 2
    loc = 0
    scale = 1
    targ = sp.stats.beta.rvs(a, b, loc, scale, size=n, random_state=rng)
    weights = get_weights_uniform(targ)

    expression = Expression("beta(loc=0, scale=1, a=c3, b=c4)", expr_name="exp1")

    # we need a first guess here because our default first guess is zeros, which leads
    # to a degenerate distribution in the case of a beta distribution
    first_guess = [1.0, 1.0]

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=first_guess,
        predictor_names=None,
    )._find_fg()

    # NOTE: for the beta distribution the support does not change for loc = 0 and scale = 1
    # it is always (0, 1), thus the optimization with _fg_fun_others does not do anything
    expected = [a, b]

    np.testing.assert_allclose(result, expected, rtol=0.5)


def test_first_guess_gamma():
    rng = np.random.default_rng(0)
    n = 251

    a = 2
    loc = 0
    scale = 1
    targ = sp.stats.gamma.rvs(a, loc, scale, size=n, random_state=rng)

    expression = Expression("gamma(loc=0, scale=1, a=c1)", expr_name="exp1")
    weights = get_weights_uniform(targ)

    # we need a first guess different from zero for gamma distribution
    first_guess = [1.0]
    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=first_guess,
        predictor_names=None,
    )._find_fg()

    expected = [a]

    np.testing.assert_allclose(result, expected, rtol=0.02)


def test_fg_fun_scale_laplace():
    rng = np.random.default_rng(0)
    n = 251
    loc = 2
    scale = 1
    targ = sp.stats.laplace.rvs(loc=loc, scale=scale, size=n, random_state=rng)
    weights = get_weights_uniform(targ)

    expression = Expression("laplace(loc=c1, scale=c2)", expr_name="exp1")
    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_default(2),
        predictor_names=None,
    )._find_fg()

    expected = [loc, scale]

    np.testing.assert_allclose(result, expected, rtol=0.1)


def test_first_guess_with_bounds():
    rng = np.random.default_rng(0)
    n = 251

    loc = 0
    loc_bounds = (-1, 1)
    scale = 1
    scale_bounds = (0.5, 1.5)

    targ = rng.normal(loc=loc, scale=scale, size=n)
    weights = get_weights_uniform(targ)

    expr_str = "norm(loc=c1, scale=c2)"
    expression = Expression(
        expr_str,
        expr_name="exp1",
        boundaries_coeffs={"c1": loc_bounds, "c2": scale_bounds},
    )

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_default(2),
        predictor_names=None,
    )._find_fg()

    # test result is within bounds
    assert loc_bounds[0] <= result[0] <= loc_bounds[1]
    assert scale_bounds[0] <= result[1] <= scale_bounds[1]
    expected = np.array([loc, scale])
    np.testing.assert_allclose(result, expected, atol=0.1)

    # test with bounds outside true value
    scale_bounds_wrong = (0.5, 0.8)
    expression = Expression(
        expr_str,
        expr_name="exp2",
        boundaries_coeffs={"c1": loc_bounds, "c2": scale_bounds_wrong},
    )

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_default(2),
        predictor_names=None,
    )._find_fg()

    expected = np.array([-0.016552528, 1.520612114])
    np.testing.assert_allclose(result, expected, atol=0.5)
    # ^ still finds a fg because we do not enforce the bounds on the fg
    # however the fg is significantly worse on the param with the wrong bounds
    # in contrast to the above the test below also runs step 6: fit on LL^n -> implications?

    # when does step 7 actually succeed?
    # when we do enforce a global optimum but the bounds are good
    boundaries_coeffs = {"c1": loc_bounds, "c2": scale_bounds}
    expression = Expression(
        expr_str, expr_name="exp1", boundaries_coeffs=boundaries_coeffs
    )

    result = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_default(2),
        predictor_names=None,
    )._find_fg()

    expected = np.array([loc, scale])
    np.testing.assert_allclose(result, expected, atol=0.1)


# @pytest.mark.xfail(reason="https://github.com/MESMER-group/mesmer/issues/581")
def test_fg_func_deriv01():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.arange(n)
    c1 = 2.0
    c2 = 0.1
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)
    weights = get_weights_uniform(targ)

    expression = Expression("norm(loc=c1*__tas__, scale=c2)", expr_name="exp1")

    fg = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_default(2),
        predictor_names=["tas"],
    )

    # preparation of derivatives
    m_smooth_targ, s_smooth_targ = np.mean(fg.smooth_targ), np.std(fg.smooth_targ)
    ind_targ_low = np.where(fg.smooth_targ < m_smooth_targ - s_smooth_targ)[0]
    ind_targ_high = np.where(fg.smooth_targ > m_smooth_targ + s_smooth_targ)[0]
    mean_high_preds = {
        pp: np.mean(fg.smooth_pred[pp][ind_targ_high], axis=0)
        for pp in fg.predictor_names
    }
    mean_low_preds = {
        pp: np.mean(fg.smooth_pred[pp][ind_targ_low], axis=0)
        for pp in fg.predictor_names
    }

    derivative_targ = np.array(
        [
            _finite_difference(
                np.mean(fg.smooth_targ[ind_targ_high]),
                np.mean(fg.smooth_targ[ind_targ_low]),
                mean_high_preds[pp],
                mean_low_preds[pp],
            )
            for pp in fg.predictor_names
        ]
    )

    loss_at_toolow = fg._fg_fun_deriv01(
        [c1 / 2], mean_high_preds, mean_low_preds, derivative_targ, m_smooth_targ
    )
    loss_at_truesolution = fg._fg_fun_deriv01(
        [c1], mean_high_preds, mean_low_preds, derivative_targ, m_smooth_targ
    )
    loss_at_toohigh = fg._fg_fun_deriv01(
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
    weights = get_weights_uniform(targ)

    expression = Expression("norm(loc=c1*__tas__, scale=c2)", expr_name="exp1")

    fg = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_default(2),
        predictor_names=["tas"],
    )

    # test local minima at true coefficients
    delta = 1.0e-4

    loss_at_toolow = fg._fg_fun_loc(c1 - delta)
    loss_at_truesolution = fg._fg_fun_loc(c1)
    loss_at_toohigh = fg._fg_fun_loc(c1 + delta)

    assert loss_at_toolow > loss_at_truesolution
    assert loss_at_toohigh > loss_at_truesolution


def test_fg_fun_loc_param_outside_bounds():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    c1 = 2.0
    c2 = 0.1
    targ = rng.normal(loc=c1 * pred, scale=c2, size=n)
    weights = get_weights_uniform(targ)

    expression = Expression(
        "norm(loc=c1*__tas__, scale=c2)",
        expr_name="exp1",
        boundaries_params={"loc": (-1, 1)},
    )

    fg = _FirstGuess(
        expression=expression,
        minimize_options=MinimizeOptions(),
        data_pred=pred,
        data_targ=targ,
        data_weights=weights,
        first_guess=fg_default(2),
        predictor_names=["tas"],
    )

    res = fg._fg_fun_loc(100)
    assert res == np.inf


def test_fg_fun_scale():
    rng = np.random.default_rng(0)
    n = 251
    loc = 0.0
    scale = 1.0

    # test normal
    targ = rng.normal(loc=loc, scale=scale, size=n)
    weights = get_weights_uniform(targ)

    expression = Expression("norm(loc=c1, scale=c2)", expr_name="exp1")
    fg = _FirstGuess(
        expression,
        MinimizeOptions(),
        data_pred=None,
        predictor_names=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=[loc, scale],
    )

    result = fg._fg_fun_scale(scale)
    np.testing.assert_allclose(result, 0, atol=0.1)

    # test GEV
    targ = sp.stats.genextreme.rvs(c=1, loc=loc, scale=scale, size=n, random_state=rng)
    weights = get_weights_uniform(targ)

    expression = Expression("genextreme(loc=c1, scale=c2, c=c3)", expr_name="exp1")
    fg = _FirstGuess(
        expression,
        MinimizeOptions(),
        data_pred=None,
        predictor_names=None,
        data_targ=targ,
        data_weights=weights,
        first_guess=[loc, scale, 1.0],
    )

    # test local minimum at true value
    delta = 1.0  # don't get closer than this since for a GEV the variance is not necessarily close to the scale
    loss_at_toolow = fg._fg_fun_scale(scale - delta)
    loss_at_truesolution = fg._fg_fun_scale(scale)
    loss_at_toohigh = fg._fg_fun_scale(scale + delta)

    assert loss_at_toolow > loss_at_truesolution
    assert loss_at_toohigh > loss_at_truesolution


def test_fg_fun_scale_param_outside_bounds():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc = 0.0
    scale = 1.0

    # test normal
    targ = rng.normal(loc=loc, scale=scale, size=n)
    weights = get_weights_uniform(targ)

    expression = Expression(
        "norm(loc=c1, scale=c2)",
        expr_name="exp1",
        boundaries_params={"scale": (0.5, 1.5)},
    )
    fg = _FirstGuess(
        expression,
        MinimizeOptions(),
        data_pred=pred,
        predictor_names=["tas"],
        data_targ=targ,
        data_weights=weights,
        first_guess=[loc, scale],
    )

    res = fg._fg_fun_scale(100)
    assert res == np.inf


# @pytest.mark.xfail(reason="https://github.com/MESMER-group/mesmer/issues/582")
def test_fg_fun_others():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc = 0
    scale = 1
    a, b = -1.2, 1.2  # nr of stds from loc at which to truncate
    targ = sp.stats.truncnorm.rvs(
        loc=loc, scale=scale, a=a, b=b, size=n, random_state=rng
    )
    weights = get_weights_uniform(targ)

    expression = Expression("truncnorm(loc=c1, scale=c2, a=c3, b=c4)", expr_name="exp1")

    fg = _FirstGuess(
        expression,
        MinimizeOptions(),
        data_pred=pred,
        predictor_names=["tas"],
        data_targ=targ,
        data_weights=weights,
        first_guess=np.array([loc, scale, a, b]),
    )

    np.testing.assert_equal(fg.data_targ, targ)
    assert list(fg.data_pred.keys()) == ["tas"]
    np.testing.assert_equal(fg.data_pred["tas"], pred)
    np.testing.assert_equal(fg.fg_coeffs, np.array([loc, scale, a, b]))
    np.testing.assert_equal(fg.expression.ind_others, np.array([2, 3]))

    # test local minimum at true value
    delta = 1
    loss_at_toolow_a = fg._fg_fun_others([a - delta, b])
    loss_at_toohigh_a = fg._fg_fun_others([a + delta, b])

    loss_at_toolow_b = fg._fg_fun_others([a, b - delta])
    loss_at_toohigh_b = fg._fg_fun_others([a, b + delta])

    loss_at_truesolution = fg._fg_fun_others([a, b])

    min_loss = np.min(
        [loss_at_toolow_a, loss_at_toohigh_a, loss_at_toolow_b, loss_at_toohigh_b]
    )

    assert min_loss >= loss_at_truesolution


def test_fg_fun_others_param_outside_bounds():
    rng = np.random.default_rng(0)
    n = 251
    pred = np.ones(n)
    loc = 0
    scale = 1
    a, b = -1.2, 1.2  # nr of stds from loc at which to truncate
    targ = sp.stats.truncnorm.rvs(
        loc=loc, scale=scale, a=a, b=b, size=n, random_state=rng
    )
    weights = get_weights_uniform(targ)

    expression = Expression(
        "truncnorm(loc=c1, scale=c2, a=c3, b=c4)",
        expr_name="exp1",
        boundaries_params={"a": (-2, 2)},
    )
    fg = _FirstGuess(
        expression,
        MinimizeOptions(),
        data_pred=pred,
        predictor_names=["tas"],
        data_targ=targ,
        data_weights=weights,
        first_guess=np.array([loc, scale, a, b]),
    )

    res = fg._fg_fun_others([100, 1])
    assert res == np.inf
