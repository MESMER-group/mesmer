from math import inf

import numpy as np
import pytest
import xarray as xr

import mesmer.distrib._conditional_distribution
from mesmer._core.utils import _check_dataarray_form, _check_dataset_form
from mesmer.distrib import (
    ConditionalDistribution,
    Expression,
    MinimizeOptions,
)
from mesmer.distrib._conditional_distribution import OptimizerNLL
from mesmer.testing import trend_data_1D, trend_data_2D


# fixture for default distribtion
@pytest.fixture
def default_distrib():
    expression = Expression("norm(loc=c1 * __tas__, scale=c2)", expr_name="exp1")
    return ConditionalDistribution(expression)


def test_minimize_options():

    default_minimize_options = MinimizeOptions()
    expected = "MinimizeOptions: 'Nelder-Mead' solver and default tolerance"

    assert default_minimize_options.method == "Nelder-Mead"
    assert default_minimize_options.tol is None
    assert default_minimize_options.options is None
    assert default_minimize_options.__repr__() == expected

    minimize_options = MinimizeOptions(
        method="Powell", tol=1e-4, options={"maxiter": 1250}
    )
    expected = "MinimizeOptions: 'Powell' solver, tol=0.0001, and additional options"

    assert minimize_options.method == "Powell"
    assert minimize_options.tol == 1e-4
    assert minimize_options.options == {"maxiter": 1250}
    assert minimize_options.__repr__() == expected


def test_ConditionalDistribution_errors():

    with pytest.raises(TypeError, match="'expression' must be an `Expression`"):
        ConditionalDistribution(None)

    expression = Expression("norm(loc=0, scale=1)", expr_name="exp1")
    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        ConditionalDistribution(expression, threshold_min_proba=-0.1)
    with pytest.raises(ValueError, match="`threshold_min_proba` must be in"):
        ConditionalDistribution(expression, threshold_min_proba=0.6)

    minimize_options = MinimizeOptions("Powell")

    with pytest.raises(
        ValueError, match="First and second minimizer have the same method"
    ):
        ConditionalDistribution(
            expression,
            minimize_options=minimize_options,
            second_minimizer=minimize_options,
        )


def test_ConditionalDistribution_init_all_default(default_distrib):
    assert default_distrib.expression.expression == "norm(loc=c1 * __tas__, scale=c2)"
    assert default_distrib.expression.boundaries_params == {"scale": [0, inf]}
    assert default_distrib.expression.boundaries_coeffs == {}
    assert default_distrib.expression.n_coeffs == 2

    assert default_distrib.threshold_min_proba == 1e-09

    assert default_distrib.minimize_options.method == "Nelder-Mead"
    assert default_distrib.minimize_options.tol is None
    assert default_distrib.minimize_options.options is None

    assert isinstance(default_distrib._optimizer, OptimizerNLL)


def test_ConditionalDistribution_custom_init():
    boundaries_params = {"loc": [-10, 10], "scale": [0, 1]}
    boundaries_coeffs = {"c1": [0, 5], "c2": [0, 1]}
    Expression(
        "norm(loc=c1 * __tas__, scale=c2)",
        expr_name="exp1",
        boundaries_params=boundaries_params,
        boundaries_coeffs=boundaries_coeffs,
    )

    MinimizeOptions("Powell", tol=1e-10, options={"maxiter": 10_000})


def test_ConditionalDistribution_from_dataset_errors():

    ds = xr.Dataset()

    with pytest.raises(ValueError, match="The 'expression' attribute is missing"):
        ConditionalDistribution.from_dataset(ds)

    ds = xr.Dataset(attrs={"expression": "norm(loc=c1 * __tas__, scale=c2)"})

    with pytest.raises(ValueError, match="The 'expression_name' attribute is missing"):
        ConditionalDistribution.from_dataset(ds)

    attrs = {
        "expression": "norm(loc=c1 * __tas__, scale=c2)",
        "expression_name": "expr",
    }

    ds = xr.Dataset(attrs=attrs)

    with pytest.raises(
        ValueError, match="'coefficients' is missing the required data_vars: 'c1', 'c2'"
    ):
        ConditionalDistribution.from_dataset(ds)


def test_ConditionalDistribution_from_dataset():

    attrs = {
        "expression": "norm(loc=c1 * __tas__, scale=c2)",
        "expression_name": "expr",
    }

    data_vars = {"c1": ("gridcell", [0, 1]), "c2": ("gridcell", [2, 3])}
    ds = xr.Dataset(data_vars=data_vars, attrs=attrs)

    cd = ConditionalDistribution.from_dataset(ds)

    assert cd.expression.expression == attrs["expression"]
    assert cd.expression.expression_name == attrs["expression_name"]

    xr.testing.assert_equal(cd.coefficients, ds)


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
        predictors=pred, target=targ, weights=weights, first_guess=fg, sample_dim="time"
    )

    np.testing.assert_allclose(default_distrib.coefficients.c1, c1, atol=1.0e-4)
    np.testing.assert_allclose(default_distrib.coefficients.c2, c2, atol=0.0015)


def test_ConditionalDistribution_fit_wrong_shape_fg(default_distrib):
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
        {"c1": c1, "c2": c2, "wrong_coeff": 1},
    )
    weights = xr.ones_like(targ)
    weights.name = "weight"

    with pytest.raises(
        ValueError,
        match="The provided first guess does not have the correct number of coeffs",
    ):
        default_distrib.fit(
            predictors=pred,
            target=targ,
            weights=weights,
            first_guess=fg,
            sample_dim="time",
        )


def test_ConditionalDistribution_fit_predictors_wrong_type(default_distrib):

    fg = xr.Dataset({"c1": 2.0, "c2": 0.1})

    with pytest.raises(
        TypeError,
        match="predictors is supposed to be a dict of xr.DataArray or a xr.Dataset",
    ):
        default_distrib.fit(
            predictors=None,
            target=None,
            weights=None,
            first_guess=fg,
            sample_dim="time",
        )


def test_ConditionalDistribution_fit_failed():
    # impose impossible bounds
    expr = Expression(
        "norm(loc=c1 * __tas__, scale=c2)",
        expr_name="exp1",
        boundaries_params={"loc": [0, 0], "scale": [0, 0]},
        boundaries_coeffs={},
    )
    # set error_failedfit to True so error is raised
    distrib = ConditionalDistribution(expr)

    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(1, n, n)
    c1 = 2.0
    c2 = 0.1

    targ = distrib.expression.distrib.rvs(
        loc=c1 * pred, scale=c2, size=n, random_state=rng
    )

    pred = {"tas": xr.DataArray(pred, dims=["time"])}
    targ = xr.DataArray(targ, dims=["time"], name="tas")
    fg = xr.Dataset(
        {"c1": c1, "c2": c2},
    )
    weights = xr.ones_like(targ)
    weights.name = "weight"

    # raises per default
    with pytest.raises(ValueError, match="Failed fit"):
        distrib.fit(
            predictors=pred,
            target=targ,
            weights=weights,
            first_guess=fg,
            sample_dim="time",
        )

    distrib.fit(
        predictors=pred,
        target=targ,
        weights=weights,
        first_guess=fg,
        sample_dim="time",
        on_failed_fit="ignore",
    )

    result = distrib.coefficients
    # TODO: set the coeffs to nan?
    expected = xr.Dataset(data_vars={"c1": 2.0, "c2": 0.1})

    xr.testing.assert_equal(result, expected)


def test_ConditionalDistribution_smooth_fg(default_distrib):
    rng = np.random.default_rng(0)
    n = 251
    pred = np.linspace(1, n, n)
    c1 = 2.0
    c2 = 0.1
    n_lon = 3
    n_lat = 2

    targ = trend_data_2D(n_timesteps=n, n_lat=n_lat, n_lon=n_lon)
    targ_dat = default_distrib.expression.distrib.rvs(
        loc=c1 * pred, scale=c2, size=(n_lon * n_lat, n), random_state=rng
    )
    targ = targ.copy(data=targ_dat)

    pred = {"tas": trend_data_1D(n_timesteps=n)}

    first_guess = xr.Dataset()
    for coef in default_distrib.expression.coefficients_list:
        first_guess[coef] = xr.DataArray(
            np.zeros(n_lon * n_lat),
            dims="cells",
            coords={"cells": np.arange(n_lon * n_lat)},
        )

    weights = xr.ones_like(targ)
    weights.name = "weight"

    default_distrib.fit(
        predictors=pred,
        target=targ,
        weights=weights,
        first_guess=first_guess,
        sample_dim="time",
        smooth_coeffs=True,
        on_failed_fit="ignore",
    )

    result = default_distrib.coefficients

    _check_dataset_form(result, "coefficients", required_vars=["c1", "c2"])


def test_smooth_first_guess():
    n = 251
    n_lon = 3
    n_lat = 2

    targ = trend_data_2D(n_timesteps=n, n_lat=n_lat, n_lon=n_lon)
    coords = targ.cells.coords

    first_guess = xr.Dataset()
    coeff1 = np.array([1.0, 2.0, 30.0, 4.0, 5.0, 6.0])
    coeff2 = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    first_guess["c1"] = xr.DataArray(
        coeff1, dims="cells", coords={"cells": np.arange(n_lon * n_lat)}
    )
    first_guess["c2"] = xr.DataArray(
        coeff2, dims="cells", coords={"cells": np.arange(n_lon * n_lat)}
    )

    first_guess_stacked = first_guess.to_dataarray(dim="coefficient")

    smoothed_guess = mesmer.distrib._conditional_distribution._smooth_first_guess(
        first_guess_stacked, "cells", coords, 500
    )

    coeff1_exp = np.array([4.0, 4.0, 5.0, 4.0, 5.0, 5.0])
    coeff2_exp = np.array([0.1, 0.1, 0.3, 0.1, 0.1, 0.3])

    np.testing.assert_equal(smoothed_guess.sel(coefficient="c1").data, coeff1_exp)
    np.testing.assert_equal(smoothed_guess.sel(coefficient="c2").data, coeff2_exp)

    _check_dataarray_form(
        smoothed_guess,
        "smoothed_guess",
        required_dims=["cells", "coefficient"],
        required_coords={
            "cells": np.arange(n_lon * n_lat),
            "coefficient": ["c1", "c2"],
        },
        shape=(2, n_lon * n_lat),
    )

    # the same but with non-dimension coords

    lat = lon = np.arange(6)
    first_guess_stacked = first_guess_stacked.drop_vars("cells")
    first_guess_stacked = first_guess_stacked.assign_coords(lat=("cells", lat))
    first_guess_stacked = first_guess_stacked.assign_coords(lon=("cells", lon))

    smoothed_guess = mesmer.distrib._conditional_distribution._smooth_first_guess(
        first_guess_stacked, "cells", coords, 500
    )

    _check_dataarray_form(
        smoothed_guess,
        "smoothed_guess",
        required_dims=["cells", "coefficient"],
        required_coords={"lat": lat, "lon": lon, "coefficient": ["c1", "c2"]},
        shape=(2, n_lon * n_lat),
    )


def test_smooth_first_guess_nans():
    n = 251
    n_lon = 3
    n_lat = 2

    targ = trend_data_2D(n_timesteps=n, n_lat=n_lat, n_lon=n_lon)
    coords = targ.cells.coords

    first_guess = xr.Dataset()
    coeff1 = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    first_guess["c1"] = xr.DataArray(
        coeff1, dims="cells", coords={"cells": np.arange(n_lon * n_lat)}
    )

    first_guess_stacked = first_guess.to_dataarray(dim="coefficient")

    smoothed_guess = mesmer.distrib._conditional_distribution._smooth_first_guess(
        first_guess_stacked, "cells", coords, 100
    )

    coeff1_exp = np.array([1.0, 2.0, 5.0, 4.0, 5.0, 6.0])

    np.testing.assert_equal(smoothed_guess.sel(coefficient="c1").data, coeff1_exp)
    _check_dataarray_form(
        smoothed_guess,
        "smoothed_guess",
        required_dims=["cells", "coefficient"],
        required_coords={"cells": np.arange(n_lon * n_lat), "coefficient": ["c1"]},
        shape=(
            1,
            n_lon * n_lat,
        ),
    )


def test_ConditionalDistribution_find_first_guess(default_distrib):
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
    weights = xr.ones_like(targ)
    weights.name = "weight"

    first_guess = default_distrib.find_first_guess(
        predictors=pred,
        target=targ,
        weights=weights,
        first_guess=None,
        sample_dim="time",
    )

    np.testing.assert_allclose(first_guess.c1, c1, atol=1.0e-4)
    np.testing.assert_allclose(first_guess.c2, c2, atol=0.0015)


def test_ConditionalDistribution_find_first_guess_providedcoeffs(default_distrib):
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
    weights = xr.ones_like(targ)
    weights.name = "weight"

    first_guess_coeffs = xr.Dataset(
        {"c1": c1, "c2": c2},
    )

    first_guess = default_distrib.find_first_guess(
        predictors=pred,
        target=targ,
        weights=weights,
        first_guess=first_guess_coeffs,
        sample_dim="time",
    )

    np.testing.assert_allclose(first_guess.c1, c1, atol=1.0e-4)
    np.testing.assert_allclose(first_guess.c2, c2, atol=0.0015)


def test_ConditionalDistribution_compute_quality_scores(default_distrib):
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
    coeffs = xr.Dataset(
        {"c1": c1, "c2": c2},
    )
    weights = xr.ones_like(targ)
    weights.name = "weight"

    default_distrib._coefficients = coeffs

    scores = default_distrib.compute_quality_scores(
        predictors=pred,
        target=targ,
        weights=weights,
        sample_dim="time",
        scores=["func_optim", "nll", "bic", "crps"],
    )

    data_vars = {
        "func_optim": -217.9307,
        "nll": -217.9307,
        "bic": -424.8105,
        "crps": 0.9363,
    }
    expected = xr.Dataset(data_vars=data_vars)

    xr.testing.assert_allclose(scores, expected, rtol=1e-4)

    # switching order should still work
    scores = ["bic", "nll"]  # crps is slow
    result = default_distrib.compute_quality_scores(
        predictors=pred,
        target=targ,
        weights=weights,
        sample_dim="time",
        scores=scores,
    )
    expected = expected.drop_vars(["crps", "func_optim"])
    xr.testing.assert_allclose(result, expected, rtol=1e-4)


def test_ConditionalDistribution_coeffs(default_distrib):
    with pytest.raises(ValueError, match="'coefficients' not set"):
        default_distrib.coefficients

    coefficients = xr.Dataset(
        {"c1": 2.0, "c2": 0.1},
    )
    default_distrib.coefficients = coefficients

    xr.testing.assert_equal(default_distrib.coefficients, coefficients)
