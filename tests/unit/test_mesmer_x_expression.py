import pytest
import scipy as sp

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
        ValueError, match="Proposed a numpy function that does not exist: np.wrong"
    ):
        Expression("norm(scale=5, loc=np.wrong())", expr_name="name")


def test_expression_wrong_math_function():

    with pytest.raises(
        ValueError, match="Proposed a math function that does not exist: math.wrong"
    ):
        Expression("norm(scale=5, loc=math.wrong())", expr_name="name")


def test_expression_wrong_numpy_or_math_function():

    match = (
        "The term 'wrong' appears in the expression 'wrong(c1)' for 'loc', but couldn't"
        " find an equivalent in numpy or math."
    )

    with pytest.raises(ValueError, match=match):
        Expression("norm(scale=5, loc=wrong(c1))", expr_name="name")


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
    assert not expr.is_distrib_discrete

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
