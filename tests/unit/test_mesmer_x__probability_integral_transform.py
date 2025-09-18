import numpy as np
import pytest
import scipy as sp
import xarray as xr

import mesmer


def test_probabilityintegraltransform_attrs():

    expr0 = mesmer.distrib.Expression("norm(loc=c1, scale=1)", "std_normal")
    data0 = xr.Dataset({"c1": 3.14})
    cond_dist0 = mesmer.distrib.ConditionalDistribution(expr0)
    cond_dist0.coefficients = data0

    expr1 = mesmer.distrib.Expression("norm(loc=0, scale=c1)", "std_normal")
    data1 = xr.Dataset({"c1": 2.71})
    cond_dist1 = mesmer.distrib.ConditionalDistribution(expr1)
    cond_dist1.coefficients = data1

    pit = mesmer.distrib.ProbabilityIntegralTransform(cond_dist0, cond_dist1)

    assert pit.expression_orig is expr0
    assert pit.expression_targ is expr1

    xr.testing.assert_equal(pit.coefficients_orig, data0)
    xr.testing.assert_equal(pit.coefficients_targ, data1)


@pytest.mark.parametrize(
    "expression",
    (
        "norm(loc=0, scale=5)",
        "genextreme(loc=0, scale=1, c=-1)",
        "genextreme(loc=0, scale=1, c=1)",
        "t(df=7, loc=0, scale=1)",
    ),
)
def test_probabilityintegraltransform_trivial_transform(expression):

    expr = mesmer.distrib.Expression(expression, "expression")

    seed = 1224851248

    rng = np.random.default_rng(seed)

    # use the distribution to draw samples - avoids values out of the support
    data = expr.distrib.rvs(**expr.evaluate_params({}, {}), size=100, random_state=rng)

    da = xr.DataArray(data, dims="x")
    ds = expected = xr.Dataset(data_vars={"foo": da})
    cond_dist = mesmer.distrib.ConditionalDistribution(expr)

    pit = mesmer.distrib.ProbabilityIntegralTransform(cond_dist, cond_dist)
    result = pit.transform(ds, target_name="foo")

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "e1",
    (
        "norm(loc=0, scale=5)",
        "genextreme(loc=0, scale=1, c=-1)",
        "t(df=7, loc=0, scale=1)",
    ),
)
@pytest.mark.parametrize(
    "e2",
    (
        "norm(loc=0, scale=5)",
        "genextreme(loc=0, scale=1, c=-1)",
        "t(df=7, loc=0, scale=1)",
    ),
)
def test_probabilityintegraltransform_back_transform(e1, e2):

    expr1 = mesmer.distrib.Expression(e1, "e1")
    expr2 = mesmer.distrib.Expression(e2, "e1")

    seed = 1224851248

    rng = np.random.default_rng(seed)

    # use the distribution to draw samples - avoids values out of the support
    data = expr1.distrib.rvs(
        **expr1.evaluate_params({}, {}), size=100, random_state=rng
    )
    da = xr.DataArray(data, dims="x")
    ds = expected = xr.Dataset(data_vars={"foo": da})

    cond_dist1 = mesmer.distrib.ConditionalDistribution(expr1)
    cond_dist2 = mesmer.distrib.ConditionalDistribution(expr2)

    pit = mesmer.distrib.ProbabilityIntegralTransform(cond_dist1, cond_dist2)
    transformed = pit.transform(ds, target_name="foo")

    pit_back = mesmer.distrib.ProbabilityIntegralTransform(cond_dist2, cond_dist1)
    result = pit_back.transform(transformed, target_name="foo")

    xr.testing.assert_allclose(result, expected)


def test_probabilityintegraltransform_threshold_proba_error():

    e = mesmer.distrib.Expression("norm(loc=0, scale=1)", "std_normal")
    d_norm = mesmer.distrib.ConditionalDistribution(e)

    pit = mesmer.distrib.ProbabilityIntegralTransform(d_norm, d_norm)

    with pytest.raises(ValueError, match=r"`threshold_proba` must be in \[0, 0.5\]"):
        pit.transform(None, "foo", threshold_proba=-1)

    with pytest.raises(ValueError, match=r"`threshold_proba` must be in \[0, 0.5\]"):
        pit.transform(None, "foo", threshold_proba=0.51)


def test_probabilityintegraltransform_threshold_proba_default():

    e = mesmer.distrib.Expression("norm(loc=0, scale=1)", "std_normal")
    d_norm = mesmer.distrib.ConditionalDistribution(e)

    # normal distribution truncated at +- 1
    e = mesmer.distrib.Expression("truncnorm(a=-1, b=1, loc=0, scale=1)", "truncnorm")
    d_truncnorm = mesmer.distrib.ConditionalDistribution(e)

    pit = mesmer.distrib.ProbabilityIntegralTransform(d_truncnorm, d_norm)

    # values below lower support
    data = xr.Dataset(data_vars={"foo": xr.DataArray([-3, -2, -1])})
    expected = sp.stats.norm.ppf(1e-9, loc=0, scale=1)
    result = pit.transform(data, "foo")
    np.testing.assert_allclose(expected, result.foo.data)

    # values above upper support
    data = xr.Dataset(data_vars={"foo": xr.DataArray([1, 2, 3])})
    expected = sp.stats.norm.ppf(1 - 1e-9, loc=0, scale=1)
    result = pit.transform(data, "foo")
    np.testing.assert_allclose(expected, result.foo.data)


@pytest.mark.parametrize("threshold_proba", (1e-10, 0.1))
def test_probabilityintegraltransform_threshold_proba(threshold_proba):

    e = mesmer.distrib.Expression("norm(loc=0, scale=1)", "std_normal")
    d_norm = mesmer.distrib.ConditionalDistribution(e)

    e = mesmer.distrib.Expression("truncnorm(a=-1, b=1, loc=0, scale=1)", "truncnorm")
    d_truncnorm = mesmer.distrib.ConditionalDistribution(e)

    pit = mesmer.distrib.ProbabilityIntegralTransform(d_truncnorm, d_norm)

    # values below lower support
    data = xr.Dataset(data_vars={"foo": xr.DataArray([-3, -2, -1])})
    expected = sp.stats.norm.ppf(threshold_proba, loc=0, scale=1)
    result = pit.transform(data, "foo", threshold_proba=threshold_proba)
    np.testing.assert_allclose(expected, result.foo.data)

    # values above upper support
    data = xr.Dataset(data_vars={"foo": xr.DataArray([1, 2, 3])})
    expected = sp.stats.norm.ppf(1 - threshold_proba, loc=0, scale=1)
    result = pit.transform(data, "foo", threshold_proba=threshold_proba)
    np.testing.assert_allclose(expected, result.foo.data)


@pytest.mark.parametrize("loc", (-3.14, 2.71))
@pytest.mark.parametrize("scale", (3.14, 2.71))
def test_probabilityintegraltransform_transform_coeffs(loc, scale):

    expr_std_norm = mesmer.distrib.Expression("norm(loc=0, scale=1)", "std_normal")
    expr_norm = mesmer.distrib.Expression("norm(loc=c1, scale=c2)", "std_normal")

    seed = 12248512485

    rng = np.random.default_rng(seed)

    coeffs = xr.Dataset(data_vars={"c1": loc, "c2": scale})

    # use the distribution to draw samples - avoids values out of the support
    data = expr_std_norm.distrib.rvs(
        **expr_std_norm.evaluate_params(coeffs, {}), size=100, random_state=rng
    )
    da = xr.DataArray(data, dims="x")
    ds = xr.Dataset(data_vars={"foo": da})

    cond_norm = mesmer.distrib.ConditionalDistribution(expr_norm)
    cond_norm.coefficients = coeffs
    cond_std_norm = mesmer.distrib.ConditionalDistribution(expr_std_norm)

    # (1) normal -> standard normal

    # converted to a standard normal by shifting and scaling
    expected = (ds - loc) / scale

    pit = mesmer.distrib.ProbabilityIntegralTransform(cond_norm, cond_std_norm)
    result = pit.transform(ds, target_name="foo")

    xr.testing.assert_allclose(result, expected)

    # (2) standard normal -> normal

    # converted from a standard normal by scaling and scaling
    expected = (ds * scale) + loc

    pit = mesmer.distrib.ProbabilityIntegralTransform(cond_std_norm, cond_norm)
    result = pit.transform(ds, target_name="foo")

    xr.testing.assert_allclose(result, expected)
