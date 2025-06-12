from math import inf

import numpy as np
import pytest
import xarray as xr

from mesmer.mesmer_x import (
    ConditionalDistribution,
    ConditionalDistributionOptions,
    Expression,
)


# fixture for default distribtion
@pytest.fixture
def default_distrib():
    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")
    return ConditionalDistribution(expression, ConditionalDistributionOptions())


def test_ConditionalDistributionOptions_errors():
    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        ConditionalDistributionOptions(
            threshold_min_proba=-0.1,
        )
    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        ConditionalDistributionOptions(
            threshold_min_proba=0.6,
        )

    with pytest.raises(ValueError, match="`options_solver` must be a dictionary"):
        ConditionalDistributionOptions(
            options_solver="this is not a dictionary",
        )

    with pytest.raises(ValueError, match="`options_optim` must be a dictionary"):
        ConditionalDistributionOptions(
            options_optim="this is not a dictionary",
        )

    with pytest.raises(ValueError, match="method for this fit not prepared, to avoid"):
        ConditionalDistributionOptions(
            options_solver={"method_fit": "this is not a method"},
        )

    with pytest.raises(
        ValueError, match="`threshold_stopping_rule` and `ind_year_thres` not used"
    ):
        ConditionalDistributionOptions(
            options_optim={"type_fun_optim": "nll", "threshold_stopping_rule": 0.1},
        )

    with pytest.raises(ValueError, match="`type_fun_optim='fcnll'` needs both, .*"):
        ConditionalDistributionOptions(
            options_optim={"type_fun_optim": "fcnll", "threshold_stopping_rule": None},
        )


def test_ConditionalDistribution_init_all_default(default_distrib):
    assert default_distrib.expression.expression == "norm(loc=c1 * __tas__, scale=c2)"
    assert default_distrib.expression.boundaries_params == {"scale": [0, inf]}
    assert default_distrib.expression.boundaries_coeffs == {}
    assert default_distrib.expression.n_coeffs == 2

    assert default_distrib.options.threshold_min_proba == 1e-09
    assert default_distrib.options.xtol_req == 1e-06
    assert default_distrib.options.ftol_req == 1e-06
    assert default_distrib.options.maxiter is None
    assert default_distrib.options.maxfev is None
    assert default_distrib.options.method_fit == "Powell"
    assert default_distrib.options.name_ftol == "ftol"
    assert default_distrib.options.name_xtol == "xtol"
    assert not default_distrib.options.error_failedfit
    assert not default_distrib.options.fg_with_global_opti
    assert default_distrib.options.type_fun_optim == "nll"
    assert default_distrib.options.threshold_stopping_rule is None
    assert default_distrib.options.exclude_trigger is None
    assert default_distrib.options.ind_year_thres is None


def test_ConditionalDistribution_custom_init():
    boundaries_params = {"loc": [-10, 10], "scale": [0, 1]}
    boundaries_coeffs = {"c1": [0, 5], "c2": [0, 1]}
    expression = Expression(
        "norm(loc=c1 * __tas__, scale=c2)",
        expr_name="exp1",
        boundaries_params=boundaries_params,
        boundaries_coeffs=boundaries_coeffs,
    )

    threshold_min_proba = 0.1
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

    options = ConditionalDistributionOptions(
        threshold_min_proba=threshold_min_proba,
        options_optim=options_optim,
        options_solver=options_solver,
    )

    distrib = ConditionalDistribution(expression, options)

    assert distrib.expression.boundaries_params == {
        "loc": [-10, 10],
        "scale": [0, 1.0],
    }
    assert distrib.expression.n_coeffs == 2
    assert distrib.expression.boundaries_coeffs == boundaries_coeffs

    assert distrib.options.threshold_min_proba == threshold_min_proba
    assert distrib.options.xtol_req == 0.1
    assert distrib.options.ftol_req == 0.01
    assert distrib.options.maxiter == 10_000
    assert distrib.options.maxfev == 12_000
    assert distrib.options.method_fit == "Nelder-Mead"
    assert distrib.options.name_ftol == "fatol"
    assert distrib.options.name_xtol == "xatol"
    assert distrib.options.error_failedfit  # is True
    assert distrib.options.fg_with_global_opti  # is True
    assert distrib.options.type_fun_optim == "fcnll"
    assert distrib.options.threshold_stopping_rule == 0.1
    assert distrib.options.exclude_trigger  # is True
    assert distrib.options.ind_year_thres == 10


def test_ConditionalDistribution_fit(default_distrib):
    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(1, n, n)
    c1 = 2.0
    c2 = 0.1

    targ = default_distrib.expression.distrib.rvs(
        loc=c1 * pred, scale=c2, size=n, random_state=rng
    )

    pred = {"tas": xr.DataArray(pred, dims=["time"])}
    targ = xr.DataArray(targ, dims=["time"], name="tas")
    fg = xr.Dataset(
        {"c1": c1, "c2": c2},
    )
    weights = xr.ones_like(targ)
    weights.name = "weight"

    default_distrib.fit(
        predictors=pred, target=targ, first_guess=fg, weights=weights, sample_dim="time"
    )

    np.testing.assert_allclose(default_distrib.coefficients.c1, c1, atol=1.0e-4)
    np.testing.assert_allclose(default_distrib.coefficients.c2, c2, atol=0.0015)


def test_ConditionalDistribution_smooth_fg_error(default_distrib):

    with pytest.raises(ValueError, match="option_smooth_coeffs has been renamed"):
        default_distrib.fit(
            predictors="dummy",
            target="dummy",
            first_guess="dummy",
            weights="dummy",
            sample_dim="dummy",
            option_smooth_coeffs=True,
        )
