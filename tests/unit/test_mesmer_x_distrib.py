import numpy as np
import pytest

import mesmer.mesmer_x


def test_distrib_init_all_default():
    rng = np.random.default_rng(0)
    n = 250
    pred = np.linspace(0, 1, n)
    targ = rng.normal(loc=2 * pred, scale=0.1, size=n)

    expression = mesmer.mesmer_x.Expression(
        "norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1"
    )

    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    train_mx = mesmer.mesmer_x.distrib_train(expression, tests_mx, optim_mx)

    assert tests_mx.expr_fit is expression
    assert optim_mx.expr_fit is expression
    assert fg_mx.expr_fit is expression
    assert train_mx.expr_fit is expression
    assert tests_mx.threshold_min_proba == 1e-09
    assert tests_mx.boundaries_params == expression.boundaries_parameters
    assert tests_mx.boundaries_coeffs == {}  # TODO: make this a property of expr_fit
    assert fg_mx.first_guess is None
    assert fg_mx.func_first_guess is None
    assert optim_mx.n_coeffs == 2  # TODO: make this a property of expr_fit
    assert fg_mx.n_coeffs == 2  # TODO: make this a property of expr_fit
    assert optim_mx.xtol_req == 1e-06
    assert optim_mx.ftol_req == 1e-06
    assert optim_mx.maxiter == 1000 * optim_mx.n_coeffs * (
        np.log(optim_mx.n_coeffs) + 1
    )
    assert optim_mx.maxfev == 1000 * optim_mx.n_coeffs * (np.log(optim_mx.n_coeffs) + 1)
    assert optim_mx.method_fit == "Powell"
    assert optim_mx.name_ftol == "ftol"
    assert optim_mx.name_xtol == "xtol"
    assert not optim_mx.error_failedfit
    assert not optim_mx.fg_with_global_opti
    assert optim_mx.type_fun_optim == "nll"
    assert optim_mx.threshold_stopping_rule is None
    assert optim_mx.exclude_trigger is None
    assert optim_mx.ind_year_thres is None


def test_distrib_init():
    rng = np.random.default_rng(0)
    n = 250
    pred = np.linspace(0, 1, n)
    targ = rng.normal(loc=2 * pred, scale=0.1, size=n)

    expression = mesmer.mesmer_x.Expression(
        "norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1"
    )

    threshold_min_proba = 0.1
    boundaries_params = {"loc": [-10, 10], "scale": [-1, 1]}
    boundaries_coeffs = {"c1": [0, 5], "c2": [0, 1]}
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expression, threshold_min_proba, boundaries_params, boundaries_coeffs
    )

    options_optim = {
        "type_fun_optim": "fcnll",
        "threshold_stopping_rule": 0.1,
        "ind_year_thres": 10,
        "exclude_trigger": True,
    }
    options_solver = {
        "method_fit": "Nelder-Mead",
        "xtol_req": 0.1,
        "ftol_req": 0.01,
        "maxiter": 10_000,
        "maxfev": 12_000,
        "error_failedfit": True,
        "fg_with_global_opti": True,
    }
    optim_mx = mesmer.mesmer_x.distrib_optimizer(
        expression, tests_mx, options_optim, options_solver
    )

    first_guess = np.array([1, 0.1])
    func_first_guess = None
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, first_guess, func_first_guess
    )

    np.testing.assert_equal(fg_mx.first_guess, first_guess)
    assert tests_mx.expr_fit is expression
    assert optim_mx.expr_fit is expression
    assert fg_mx.expr_fit is expression
    assert tests_mx.threshold_min_proba == threshold_min_proba
    assert tests_mx.boundaries_params == {
        "loc": [-10, 10],
        "scale": [0, 1.0],
    }  # -1 -> 0 since no negative values allowed
    assert tests_mx.boundaries_coeffs == boundaries_coeffs
    assert optim_mx.xtol_req == 0.1
    assert optim_mx.ftol_req == 0.01
    assert optim_mx.maxiter == 10_000
    assert optim_mx.maxfev == 12_000
    assert optim_mx.method_fit == "Nelder-Mead"
    assert optim_mx.name_ftol == "fatol"
    assert optim_mx.name_xtol == "xatol"
    assert optim_mx.error_failedfit  # is True
    assert optim_mx.fg_with_global_opti  # is True
    assert optim_mx.type_fun_optim == "fcnll"
    assert optim_mx.threshold_stopping_rule == 0.1
    assert optim_mx.exclude_trigger  # is True
    assert optim_mx.ind_year_thres == 10
    assert optim_mx.n_coeffs == 2  # TODO: make this a property of expr_fit
    assert fg_mx.n_coeffs == 2  # TODO: make this a property of expr_fit
    assert fg_mx.func_first_guess is None


def test_distrib_init_errors():
    expression = mesmer.mesmer_x.Expression(
        "norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1"
    )
    tests_mx = mesmer.mesmer_x.distrib_tests(expression, 1.0e-9, None, None)
    optim_mx = mesmer.mesmer_x.distrib_optimizer(expression, tests_mx, None, None)
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expression, tests_mx, optim_mx, None, None
    )
    train_mx = mesmer.mesmer_x.distrib_train(expression, tests_mx, optim_mx)

    with pytest.raises(ValueError, match="nan values in predictors"):
        result = fg_mx._find_fg_np(
            np.array([1, 2, np.nan]), np.array([1, 2, 3]), np.array([1, 1, 1]) / 3
        )

    with pytest.raises(ValueError, match="infinite values in predictors"):
        result = fg_mx._find_fg_np(
            np.array([1, 2, np.inf]), np.array([1, 2, 3]), np.array([1, 1, 1]) / 3
        )

    with pytest.raises(ValueError, match="nan values in target"):
        result = fg_mx._find_fg_np(
            np.array([1, 2, 3]), np.array([1, 2, np.nan]), np.array([1, 1, 1]) / 3
        )

    with pytest.raises(ValueError, match="infinite values in target"):
        result = fg_mx._find_fg_np(
            np.array([1, 2, 3]), np.array([1, 2, np.inf]), np.array([1, 1, 1]) / 3
        )

    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        _ = mesmer.mesmer_x.distrib_tests(expression, -0.1, None, None)
    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        _ = mesmer.mesmer_x.distrib_tests(expression, 0.6, None, None)

    with pytest.raises(
        ValueError, match="The provided first guess does not have the correct shape:"
    ):
        _ = train_mx._fit_np(
            data_pred=np.array([1, 2, 3]),
            data_targ=np.array([1, 2, 3]),
            fg=np.array([1, 2, 3]),
            data_weights=np.array([1, 2, 3]),
        )

    with pytest.raises(ValueError, match="`options_solver` must be a dictionary"):
        _ = mesmer.mesmer_x.distrib_optimizer(
            expression, tests_mx, None, "this is not a dictionary"
        )

    with pytest.raises(ValueError, match="`options_optim` must be a dictionary"):
        _ = mesmer.mesmer_x.distrib_optimizer(
            expression, tests_mx, "this is not a dictionary", None
        )

    with pytest.raises(ValueError, match="method for this fit not prepared, to avoid"):
        _ = mesmer.mesmer_x.distrib_optimizer(
            expression, tests_mx, None, {"method_fit": "this is not a method"}
        )

    with pytest.raises(
        ValueError, match="`threshold_stopping_rule` and `ind_year_thres` not used"
    ):
        _ = mesmer.mesmer_x.distrib_optimizer(
            expression,
            tests_mx,
            {"type_fun_optim": "nll", "threshold_stopping_rule": 0.1},
            None,
        )

    with pytest.raises(ValueError, match="`type_fun_optim='fcnll'` needs both, .*"):
        _ = mesmer.mesmer_x.distrib_optimizer(
            expression,
            tests_mx,
            {"type_fun_optim": "fcnll", "threshold_stopping_rule": None},
            None,
        )


def test_distrib_get_weights():
    n = 3

    weights_unif = mesmer.mesmer_x.get_weights_uniform(
        targ_data=np.arange(n), target=None, dims=None
    )

    np.testing.assert_equal(weights_unif, np.ones(n) / n)

    weights_dens = mesmer.mesmer_x.get_weights_density(
        pred_data=np.arange(n),
        predictor=None,
        targ_data=np.arange(n),
        target=None,
        dims=None,
    )

    np.testing.assert_equal(weights_dens, weights_dens / np.sum(weights_dens))
