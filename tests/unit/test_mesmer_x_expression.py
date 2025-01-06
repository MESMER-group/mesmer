import numpy as np
import pytest
import scipy as sp
import xarray as xr

import mesmer
from mesmer.mesmer_x import Expression

inf = float("inf")


def test_expression_wrong_distr():

    with pytest.raises(AttributeError, match="Could not find distribution 'wrong'"):
        Expression("wrong()", expr_name="name")


def test_expression_wrong_param():

    with pytest.raises(ValueError, match="The parameter 'wrong' is not part"):
        Expression("norm(wrong=5)", expr_name="name")


def test_expression_missing_param():

    with pytest.raises(ValueError, match="No information provided for `loc`"):
        Expression("norm(scale=5)", expr_name="name")


def test_expression_wrong_np_function():

    with pytest.raises(
        ValueError, match="Proposed a numpy function that does not exist: 'np.wrong'"
    ):
        Expression("norm(scale=5, loc=np.wrong())", expr_name="name")


def test_expression_wrong_math_function():

    with pytest.raises(
        ValueError, match="Proposed a math function that does not exist: 'math.wrong'"
    ):
        Expression("norm(scale=5, loc=math.wrong())", expr_name="name")


def test_expression_wrong_function():

    with pytest.raises(
        ValueError, match=r"Unknown function 'mean' in expression 'mean\(\)' for 'loc'"
    ):
        Expression("norm(scale=5, loc=mean())", expr_name="name")


def test_expression_set_params():

    expression_str = "norm(loc=0, scale=c1)"
    expr = Expression(expression_str, expr_name="name")

    assert expr.expression == expression_str
    assert expr.expression_name == "name"

    assert expr.distrib == sp.stats.norm
    assert not expr.is_distrib_discrete

    assert expr.parameters_list == ["loc", "scale"]

    bounds = {"loc": [-inf, inf], "scale": [0, inf]}
    assert expr.boundaries_parameters == bounds

    param_expr = {"loc": "0", "scale": "c1"}
    assert expr.parameters_expressions == param_expr

    coeffs = ["c1"]
    assert expr.coefficients_list == coeffs

    coeffs_per_param = {"loc": [], "scale": ["c1"]}
    assert expr.coefficients_dict == coeffs_per_param


def test_expression_genextreme():

    expression_str = (
        "genextreme(loc=c1 + c2 * __pred1__, scale=c3 + c4 * __pred2__**2, c=c5)"
    )

    expr = Expression(expression_str, expr_name="name")

    assert expr.expression == expression_str
    assert expr.expression_name == "name"

    assert expr.distrib == sp.stats.genextreme
    assert not expr.is_distrib_discrete

    assert expr.parameters_list == ["c", "loc", "scale"]

    bounds = {"c": [-inf, inf], "loc": [-inf, inf], "scale": [0, inf]}
    assert expr.boundaries_parameters == bounds

    param_expr = {"loc": "c1+c2*pred1", "scale": "c3+c4*pred2**2", "c": "c5"}
    assert expr.parameters_expressions == param_expr

    coeffs = ["c1", "c2", "c3", "c4", "c5"]
    assert expr.coefficients_list == coeffs

    coeffs_per_param = {"loc": ["c1", "c2"], "scale": ["c3", "c4"], "c": ["c5"]}
    assert expr.coefficients_dict == coeffs_per_param


def test_expression_norm():

    expression_str = "norm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __GMT_t__ + c4 * __GMT_tm1__ - c5) ), scale=c6)"

    expr = Expression(expression_str, expr_name="name")

    assert expr.expression == expression_str
    assert expr.expression_name == "name"

    assert expr.distrib == sp.stats.norm
    assert not expr.is_distrib_discrete

    assert expr.parameters_list == ["loc", "scale"]

    bounds = {"loc": [-inf, inf], "scale": [0, inf]}
    assert expr.boundaries_parameters == bounds

    param_expr = {"loc": "c1+(c2-c1)/(1+np.exp(c3*GMT_t+c4*GMT_tm1-c5))", "scale": "c6"}
    assert expr.parameters_expressions == param_expr

    coeffs = ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert expr.coefficients_list == coeffs

    coeffs_per_param = {"loc": ["c1", "c2", "c3", "c4", "c5"], "scale": ["c6"]}
    assert expr.coefficients_dict == coeffs_per_param


def test_expression_binom():
    # a discrete distribution

    expression_str = "binom(loc=c1, n=5, p=7)"

    expr = Expression(expression_str, expr_name="name")

    assert expr.expression == expression_str
    assert expr.expression_name == "name"

    assert expr.distrib == sp.stats.binom
    assert expr.is_distrib_discrete

    assert expr.parameters_list == ["n", "p", "loc"]

    bounds = {"n": [-inf, inf], "loc": [-inf, inf], "p": [-inf, inf]}
    assert expr.boundaries_parameters == bounds

    param_expr = {"loc": "c1", "n": "5", "p": "7"}
    assert expr.parameters_expressions == param_expr

    coeffs = ["c1"]
    assert expr.coefficients_list == coeffs

    coeffs_per_param = {"loc": ["c1"], "n": [], "p": []}
    assert expr.coefficients_dict == coeffs_per_param


@pytest.mark.xfail(reason="https://github.com/MESMER-group/mesmer/issues/525")
def test_expression_exponpow():

    expression_str = "exponpow(loc=c1, scale=c2+np.min([np.max(np.mean([__GMT_tm1__,__GMT_tp1__],axis=0)), math.gamma(__XYZ__)]), b=c3)"

    expr = Expression(expression_str, expr_name="name")

    assert expr.expression == expression_str
    assert expr.expression_name == "name"

    assert expr.distrib == sp.stats.exponpow
    assert not expr.is_distrib_discrete

    assert expr.parameters_list == ["loc", "scale"]

    bounds = {"b": [inf, inf], "loc": [-inf, inf], "scale": [0, inf]}
    assert expr.boundaries_parameters == bounds

    param_expr = {
        "loc": "c1",
        "scale": "c2+np.min([np.max(np.mean([GMT_tm1,GMT_tp1],axis=0))]",
        "b": "c3",
    }
    assert expr.parameters_expressions == param_expr

    coeffs = ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert expr.coefficients_list == coeffs

    coeffs_per_param = {"loc": ["c1"], "scale": ["c2"], "b": ["c3"]}
    assert expr.coefficients_dict == coeffs_per_param


@pytest.mark.xfail(
    reason="https://github.com/MESMER-group/mesmer/issues/525#issuecomment-2385065252"
)
def test_expression_coefficients_two_digits():

    expr = Expression("norm(loc=c1, scale=c2 + c10 * __T__)", "name")
    assert expr.coefficients_list == ["c1", "c2", "c10"]


@pytest.mark.xfail(
    reason="https://github.com/MESMER-group/mesmer/issues/525#issuecomment-2385065252"
)
def test_expression_covariate_c_digit():

    expr = Expression("norm(loc=c1, scale=c2 * __Tc3__)", "name")
    assert expr.coefficients_list == ["c1", "c2"]


@pytest.mark.xfail(
    reason="https://github.com/MESMER-group/mesmer/issues/525#issuecomment-2385065252"
)
def test_expression_covariate_wrong_underscores():

    # not sure wath the correct behavior should be
    # - raise?
    # - get "T__C" as covariate?

    with pytest.raises(ValueError, match=""):
        Expression("norm(loc=c1, scale=c2 * __T__C__)", "name")


def test_expression_covariate_substring():

    # test that GMT and GMT1 (which contains the substring GMT) is correctly parsed

    expression_str = "norm(loc=c1 + c2 * __GMT__ + c3 * __GMT1__, scale=c4)"

    expr = Expression(expression_str, expr_name="name")

    assert expr.expression == expression_str

    param_expr = {"loc": "c1+c2*GMT+c3*GMT1", "scale": "c4"}
    assert expr.parameters_expressions == param_expr

    coeffs = ["c1", "c2", "c3", "c4"]
    assert expr.coefficients_list == coeffs

    coeffs_per_param = {"loc": ["c1", "c2", "c3"], "scale": ["c4"]}
    assert expr.coefficients_dict == coeffs_per_param


def test_evaluate_missing_coefficient_dict():

    expr = Expression("norm(loc=c1, scale=c2)", expr_name="name")

    with pytest.raises(
        ValueError, match="Missing information for the coefficient: 'c1'"
    ):
        expr.evaluate({}, {})

    with pytest.raises(
        ValueError, match="Missing information for the coefficient: 'c2'"
    ):
        expr.evaluate({"c1": 1}, {})


def test_evaluate_missing_coefficient_dataset():

    expr = Expression("norm(loc=c1, scale=c2)", expr_name="name")

    with pytest.raises(
        ValueError, match="Missing information for the coefficient: 'c1'"
    ):
        expr.evaluate(xr.Dataset(), {})

    with pytest.raises(
        ValueError, match="Missing information for the coefficient: 'c2'"
    ):
        expr.evaluate(xr.Dataset(data_vars={"c1": 1}), {})


def test_evaluate_missing_coefficient_list():

    expr = Expression("norm(loc=c1, scale=c2)", expr_name="name")

    with pytest.raises(
        ValueError, match="Inconsistent information for the coefficients_values"
    ):
        expr.evaluate([], {})

    with pytest.raises(
        ValueError, match="Inconsistent information for the coefficients_values"
    ):
        expr.evaluate([1], {})


def test_evaluate_missing_covariates_dict():

    expr = Expression("norm(loc=c1 * __T__, scale=c2 * __F__)", expr_name="name")

    with pytest.raises(ValueError, match="Missing information for the input: 'T'"):
        expr.evaluate([1, 1], {})

    with pytest.raises(ValueError, match="Missing information for the input: 'F'"):
        expr.evaluate([1, 1], {"T": 1})


def test_evaluate_missing_covariates_ds():

    expr = Expression("norm(loc=c1 * __T__, scale=c2 * __F__)", expr_name="name")

    with pytest.raises(ValueError, match="Missing information for the input: 'T'"):
        expr.evaluate([1, 1], xr.Dataset())

    with pytest.raises(ValueError, match="Missing information for the input: 'F'"):
        expr.evaluate([1, 1], xr.Dataset(data_vars={"T": 1}))


def test_evaluate_covariates_wrong_shape():

    expr = Expression("norm(loc=c1 * __T__, scale=c2 * __F__)", expr_name="name")

    T = np.array([1])
    F = np.array([1, 1])
    data_vars = {"T": T, "F": F}

    with pytest.raises(ValueError, match="shapes of inputs must be equal"):
        expr.evaluate([1, 1], data_vars)

    with pytest.raises(ValueError, match="shapes of inputs must be equal"):
        expr.evaluate([1, 1], xr.Dataset(data_vars=data_vars))


def test_evaluate_params_norm():

    expr = Expression("norm(loc=c1 * __T__, scale=c2)", expr_name="name")
    params = expr.evaluate_params([1, 2], {"T": np.array([1, 2])})

    assert isinstance(params, dict)

    expected = {"loc": np.array([1, 2]), "scale": np.array([2.0, 2.0])}

    # assert frozen params are equal
    mesmer.testing.assert_dict_allclose(params, expected)

    # a second set of values
    params = expr.evaluate_params([2, 1], {"T": np.array([2, 5])})

    expected = {"loc": np.array([4, 10]), "scale": np.array([1.0, 1.0])}

    # assert frozen params are equal
    mesmer.testing.assert_dict_allclose(params, expected)


@pytest.mark.xfail(
    reason="https://github.com/MESMER-group/mesmer/issues/525#issuecomment-2557261793"
)
def test_evaluate_params_norm_set_params_with_float():

    expr = Expression("norm(loc= c1 * __T__, scale=0.1)", expr_name="name")
    params = expr.evaluate_params([1], {"T": np.array([1, 2])})

    assert isinstance(params, dict)

    expected = {"loc": np.array([1, 2]), "scale": np.array([0.1, 0.1])}

    # assert frozen params are equal
    mesmer.testing.assert_dict_allclose(params, expected)

    # a second set of values
    params = expr.evaluate_params([2], {"T": np.array([2, 5])})

    expected = {"loc": np.array([4, 10]), "scale": np.array([0.1, 0.1])}

    # assert frozen params are equal
    mesmer.testing.assert_dict_allclose(params, expected)


def test_evaluate_params_norm_dataset():
    # NOTE: not sure if passing DataArray to scipy distribution is a good idea

    expr = Expression("norm(loc=c1 * __T__, scale=c2)", expr_name="name")

    coefficients_values = xr.Dataset(data_vars={"c1": 1, "c2": 2})
    inputs_values = xr.Dataset(data_vars={"T": ("x", np.array([1, 2]))})

    params = expr.evaluate_params(coefficients_values, inputs_values)

    loc = xr.DataArray([1, 2], dims="x")
    scale = xr.DataArray([2, 2], dims="x")

    expected = {"loc": loc, "scale": scale}

    # assert frozen params are equal
    mesmer.testing.assert_dict_allclose(params, expected)


def test_evaluate_norm():

    expr = Expression("norm(loc=c1 * __T__, scale=c2)", expr_name="name")
    dist = expr.evaluate([1, 2], {"T": np.array([1, 2])})

    assert isinstance(dist.dist, type(sp.stats.norm))

    expected = {"loc": np.array([1, 2]), "scale": np.array([2.0, 2.0])}

    # assert frozen params are equal
    mesmer.testing.assert_dict_allclose(dist.kwds, expected)

    # a second set of values
    dist = expr.evaluate([2, 1], {"T": np.array([2, 5])})

    expected = {"loc": np.array([4, 10]), "scale": np.array([1.0, 1.0])}
    mesmer.testing.assert_dict_allclose(dist.kwds, expected)


def test_evaluate_norm_dataset():
    # NOTE: not sure if passing DataArray to scipy distribution is a good idea

    expr = Expression("norm(loc=c1 * __T__, scale=c2)", expr_name="name")

    coefficients_values = xr.Dataset(data_vars={"c1": 1, "c2": 2})
    inputs_values = xr.Dataset(data_vars={"T": ("x", np.array([1, 2]))})

    dist = expr.evaluate(coefficients_values, inputs_values)

    assert isinstance(dist.dist, type(sp.stats.norm))

    loc = xr.DataArray([1, 2], dims="x")
    scale = xr.DataArray([2, 2], dims="x")

    expected = {"loc": loc, "scale": scale}

    # assert frozen params are equal
    mesmer.testing.assert_dict_allclose(dist.kwds, expected)
