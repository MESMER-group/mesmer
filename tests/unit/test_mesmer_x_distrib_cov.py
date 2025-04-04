import numpy as np
import pytest

from mesmer.mesmer_x import Expression, distrib_cov


def test_distrib_cov_init_all_default():
    rng = np.random.default_rng(0)
    n = 250
    pred = np.linspace(0, 1, n)
    targ = rng.normal(loc=2 * pred, scale=0.1, size=n)

    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)

    np.testing.assert_equal(dist.data_targ, targ)
    np.testing.assert_equal(dist.data_pred, {"tas": pred})
    np.testing.assert_equal(dist.weights_driver, np.ones(n) / np.sum(np.ones(n)))
    assert dist.n_sample == n
    assert dist.expr_fit is expression
    assert not dist.add_test
    assert dist.data_targ_addtest is None
    assert dist.data_preds_addtest is None
    assert dist.threshold_min_proba == 1e-09
    assert dist.boundaries_params == expression.boundaries_parameters
    assert dist.boundaries_coeffs == {}  # TODO: make this a property of expr_fit
    assert dist.first_guess is None  # TODO: make this a property of expr_fit
    assert dist.func_first_guess is None  # TODO: make this a property of expr_fit
    assert dist.n_coeffs == 2  # TODO: make this a property of expr_fit
    assert dist.scores_fit == ["func_optim", "NLL", "BIC"]
    assert dist.xtol_req == 1e-06
    assert dist.ftol_req == 1e-06
    assert dist.maxiter == 1000 * dist.n_coeffs * (np.log(dist.n_coeffs) + 1)
    assert dist.maxfev == 1000 * dist.n_coeffs * (np.log(dist.n_coeffs) + 1)
    assert dist.method_fit == "Powell"
    assert dist.name_ftol == "ftol"
    assert dist.name_xtol == "xtol"
    assert not dist.error_failedfit
    assert not dist.fg_with_global_opti
    assert not dist.weighted_NLL
    assert dist.type_fun_optim == "NLL"
    assert dist.threshold_stopping_rule is None
    assert dist.exclude_trigger is None
    assert dist.ind_year_thres is None


def test_distrib_cov_init():
    rng = np.random.default_rng(0)
    n = 250
    pred = np.linspace(0, 1, n)
    targ = rng.normal(loc=2 * pred, scale=0.1, size=n)

    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")

    data_targ_addtest = rng.normal(loc=2 * pred, scale=0.1, size=n)
    data_preds_addtest = {"tas": np.linspace(0, 0.9, n)}
    threshold_min_proba = 0.1
    boundaries_params = {"loc": [-10, 10], "scale": [-1, 1]}
    boundaries_coeffs = {"c1": [0, 5], "c2": [0, 1]}
    first_guess = np.array([1, 0.1])
    func_first_guess = None
    scores_fit = ["func_optim", "NLL"]
    options_optim = {
        "type_fun_optim": "fcNLL",
        "weighted_NLL": True,
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

    dist = distrib_cov(
        targ,
        {"tas": pred},
        expression,
        data_targ_addtest=data_targ_addtest,
        data_preds_addtest=data_preds_addtest,
        threshold_min_proba=threshold_min_proba,
        boundaries_params=boundaries_params,
        boundaries_coeffs=boundaries_coeffs,
        first_guess=first_guess,
        func_first_guess=func_first_guess,
        scores_fit=scores_fit,
        options_optim=options_optim,
        options_solver=options_solver,
    )

    np.testing.assert_equal(dist.data_targ, targ)
    np.testing.assert_equal(dist.data_pred, {"tas": pred})
    np.testing.assert_equal(dist.weights_driver, dist.get_weights())
    np.testing.assert_equal(
        dist.weights_driver, dist._get_weights_nll() / np.sum(dist._get_weights_nll())
    )
    np.testing.assert_equal(dist.first_guess, first_guess)
    np.testing.assert_equal(dist.data_targ_addtest, data_targ_addtest)
    np.testing.assert_equal(dist.data_preds_addtest, data_preds_addtest)
    assert dist.n_sample == n
    assert dist.expr_fit is expression
    assert dist.add_test  # is True
    assert dist.threshold_min_proba == threshold_min_proba
    assert dist.boundaries_params == {
        "loc": [-10, 10],
        "scale": [0, 1.0],
    }  # -1 -> 0 since no negative values allowed
    assert dist.boundaries_coeffs == boundaries_coeffs
    assert dist.func_first_guess is None
    assert dist.n_coeffs == 2
    assert dist.scores_fit == scores_fit
    assert dist.xtol_req == 0.1
    assert dist.ftol_req == 0.01
    assert dist.maxiter == 10_000
    assert dist.maxfev == 12_000
    assert dist.method_fit == "Nelder-Mead"
    assert dist.name_ftol == "fatol"
    assert dist.name_xtol == "xatol"
    assert dist.error_failedfit  # is True
    assert dist.fg_with_global_opti  # is True
    assert dist.weighted_NLL  # is True
    assert dist.type_fun_optim == "fcNLL"
    assert dist.threshold_stopping_rule == 0.1
    assert dist.exclude_trigger  # is True
    assert dist.ind_year_thres == 10


def test_distrib_cov_init_errors():
    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")

    with pytest.raises(ValueError, match="nan values in target"):
        distrib_cov(np.array([1, 2, np.nan]), {"tas": np.array([1, 2, 3])}, expression)

    with pytest.raises(ValueError, match="infinite values in target"):
        distrib_cov(np.array([1, 2, np.inf]), {"tas": np.array([1, 2, 3])}, expression)

    with pytest.raises(ValueError, match="nan values in predictors"):
        distrib_cov(np.array([1, 2, 3]), {"tas": np.array([1, 2, np.nan])}, expression)

    with pytest.raises(ValueError, match="infinite values in predictors"):
        distrib_cov(np.array([1, 2, 3]), {"tas": np.array([1, 2, np.inf])}, expression)

    with pytest.raises(ValueError, match="nan values in predictors"):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, np.inf]), "tas2": np.array([1, 2, np.nan])},
            expression,
        )

    with pytest.raises(
        ValueError, match="Only one of `data_targ_addtest` & `data_preds_addtest`"
    ):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            data_targ_addtest=np.array([1, 2, 3]),
        )

    with pytest.raises(
        ValueError, match="Only one of `data_targ_addtest` & `data_preds_addtest`"
    ):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            data_preds_addtest={"tas": np.array([1, 2, 3])},
        )

    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            threshold_min_proba=-1,
        )
    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            threshold_min_proba=2,
        )

    with pytest.raises(
        ValueError, match="The provided first guess does not have the correct shape:"
    ):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            first_guess=np.array([1, 2, 3]),
        )

    with pytest.raises(ValueError, match="`options_solver` must be a dictionary"):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            options_solver="this is not a dictionary",
        )

    with pytest.raises(ValueError, match="`options_optim` must be a dictionary"):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            options_optim="this is not a dictionary",
        )

    with pytest.raises(ValueError, match="method for this fit not prepared, to avoid"):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            options_solver={"method_fit": "this is not a method"},
        )

    with pytest.raises(
        ValueError, match="`threshold_stopping_rule` and `ind_year_thres` not used"
    ):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            options_optim={"type_fun_optim": "NLL", "threshold_stopping_rule": 0.1},
        )

    with pytest.raises(ValueError, match="`type_fun_optim='fcNLL'` needs both, .*"):
        distrib_cov(
            np.array([1, 2, 3]),
            {"tas": np.array([1, 2, 3])},
            expression,
            options_optim={"type_fun_optim": "fcNLL", "threshold_stopping_rule": None},
        )


def test_distrib_cov_get_weights():
    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")
    n = 3

    dist = distrib_cov(
        np.arange(n), {}, expression, options_optim={"weighted_NLL": True}
    )
    np.testing.assert_equal(dist.weights_driver, np.ones(n) / np.sum(np.ones(n)))

    dist = distrib_cov(
        np.arange(n),
        {"tas": np.arange(n)},
        expression,
        options_optim={"weighted_NLL": True},
    )
    np.testing.assert_equal(
        dist.weights_driver, dist._get_weights_nll() / np.sum(dist._get_weights_nll())
    )

    dist = distrib_cov(
        np.arange(n),
        {"tas": np.arange(n)},
        expression,
        options_optim={"weighted_NLL": False},
    )
    np.testing.assert_equal(dist.weights_driver, np.ones(n) / np.sum(np.ones(n)))


def test_distrib_cov_get_weights_nll():
    # NOTE: case for no predictor already covered in test_distrib_cov_get_weights

    # constant density = constant weights, with each weight = 1/n_samples
    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")
    n_samples = 10
    n_bins_density = 5 + 1  # currently need to give bins + 1 to get n_bins correct

    pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    dist = distrib_cov(
        np.arange(n_samples), {"pred1": pred}, expression
    )  # can currently not provide n_bins
    weights = dist._get_weights_nll(n_bins_density=n_bins_density)

    np.testing.assert_equal(
        weights / weights.sum(), np.repeat(1 / n_samples, n_samples)
    )
