from coverage import data
from networkx.algorithms import threshold
import numpy as np
import pytest
import scipy as sp
from toolz import first
import xarray as xr

import mesmer
from mesmer.mesmer_x import distrib_cov, Expression

def test_distrib_cov_init_all_default():
    rng = np.random.default_rng(0)
    n = 250
    pred = np.linspace(0, 1, n)
    targ = rng.normal(loc=2*pred, scale=0.1, size=n)

    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")

    dist = distrib_cov(targ, {"tas": pred}, expression)

    np.testing.assert_equal(dist.data_targ, targ)
    np.testing.assert_equal(dist.data_pred, {"tas": pred})
    np.testing.assert_equal(dist.weights_driver, np.ones(n) / n)
    assert dist.n_sample == n
    assert dist.expr_fit == expression
    assert dist.add_test == False
    assert dist.data_targ_addtest is None
    assert dist.data_preds_addtest is None
    assert dist.threshold_min_proba == 1e-09
    assert dist.boundaries_params == expression.boundaries_parameters
    assert dist.boundaries_coeffs == {}
    assert dist.first_guess == None
    assert dist.func_first_guess == None
    assert dist.n_coeffs == 2
    assert dist.scores_fit == ["func_optim", "NLL", "BIC"]
    assert dist.xtol_req == 1e-06
    assert dist.ftol_req == 1e-06
    assert dist.maxiter == 1000 * dist.n_coeffs * np.log(dist.n_coeffs)
    assert dist.maxfev == 1000 * dist.n_coeffs * np.log(dist.n_coeffs)
    assert dist.method_fit == "Powell"
    assert dist.name_ftol == "ftol"
    assert dist.name_xtol == "xtol"
    assert dist.error_failedfit == False
    assert dist.fg_with_global_opti == False
    assert dist.weighted_NLL == False
    assert dist.type_fun_optim == "NLL"
    assert dist.threshold_stopping_rule == None
    assert dist.exclude_trigger == None
    assert dist.ind_year_thres == None

def test_distrib_cov_init():
    rng = np.random.default_rng(0)
    n = 250
    pred = np.linspace(0, 1, n)
    targ = rng.normal(loc=2*pred, scale=0.1, size=n)

    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")

    data_targ_addtest = rng.normal(loc=2*pred, scale=0.1, size=n)
    data_preds_addtest = {"tas": np.linspace(0, 0.9, n)}
    threshold_min_proba = 0.1
    boundaries_params = {"loc": [-10, 10], "scale": [0, 1]}
    boundaries_coeffs = {"c1": [0, 5], "c2": [0, 1]}
    first_guess = np.array([1, 0.1])
    func_first_guess = None
    scores_fit = ["func_optim", "NLL"]
    options_optim = {"type_fun_optim": "fcNLL",
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

    dist = distrib_cov(targ, {"tas": pred}, expression,
                       data_targ_addtest=data_targ_addtest,
                       data_preds_addtest=data_preds_addtest,
                       threshold_min_proba=threshold_min_proba,
                       boundaries_params=boundaries_params,
                       boundaries_coeffs=boundaries_coeffs,
                       first_guess=first_guess,
                       func_first_guess=func_first_guess,
                       scores_fit=scores_fit,
                       options_optim=options_optim,
                       options_solver=options_solver)

    np.testing.assert_equal(dist.data_targ, targ)
    np.testing.assert_equal(dist.data_pred, {"tas": pred})
    np.testing.assert_equal(dist.weights_driver, dist.get_weights())
    # np.testing.assert_equal(dist.weights_driver, dist._get_weights_nll()) # WHY NOT???
    np.testing.assert_equal(dist.first_guess, first_guess)
    assert dist.n_sample == n
    assert dist.expr_fit == expression
    assert dist.add_test == True
    assert dist.data_targ_addtest is data_targ_addtest
    assert dist.data_preds_addtest is data_preds_addtest
    assert dist.threshold_min_proba == threshold_min_proba
    assert dist.boundaries_params == boundaries_params
    assert dist.boundaries_coeffs == boundaries_coeffs
    assert dist.func_first_guess == None
    assert dist.n_coeffs == 2
    assert dist.scores_fit == scores_fit
    assert dist.xtol_req == 0.1
    assert dist.ftol_req == 0.01
    assert dist.maxiter == 10_000
    assert dist.maxfev == 12_000
    assert dist.method_fit == "Nelder-Mead"
    assert dist.name_ftol == "fatol"
    assert dist.name_xtol == "xatol"
    assert dist.error_failedfit == True
    assert dist.fg_with_global_opti == True
    assert dist.weighted_NLL == True
    assert dist.type_fun_optim == "fcNLL"
    assert dist.threshold_stopping_rule == 0.1
    assert dist.exclude_trigger == True
    assert dist.ind_year_thres == 10

def test_distrib_cov_init_errors():
    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")

    with pytest.raises(ValueError, match = "NaN or infinite values in target of fit"):
        distrib_cov(np.array([1, 2, np.nan]), {"tas": np.array([1, 2, 3])}, expression)

    with pytest.raises(ValueError, match = "NaN or infinite values in target of fit"):
        distrib_cov(np.array([1, 2, np.inf]), {"tas": np.array([1, 2, 3])}, expression)
    
    with pytest.raises(ValueError, match = "NaN or infinite values in predictors of fit"):
        distrib_cov(np.array([1, 2, 3]), {"tas": np.array([1, 2, np.nan])}, expression)

    with pytest.raises(ValueError, match = "NaN or infinite values in predictors of fit"):
        distrib_cov(np.array([1, 2, 3]), {"tas": np.array([1, 2, np.inf])}, expression)

    with pytest.raises(ValueError, match = "NaN or infinite values in predictors of fit"):
        distrib_cov(np.array([1, 2, 3]), {"tas": np.array([1, 2, np.inf]),
                                          "tas2": np.array([1,2, np.nan])}, expression)