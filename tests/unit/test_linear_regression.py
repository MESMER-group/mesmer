from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

import mesmer
import mesmer.stats._linear_regression
from mesmer.testing import trend_data_1D, trend_data_2D


def convert_to(dct: dict, data_type: str) -> dict | xr.Dataset:
    if data_type == "dict":
        return dct
    elif data_type == "xr_dataset":
        return xr.Dataset(dct)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def trend_data_1D_or_2D(as_2D, slope, scale, intercept):
    if as_2D:
        return trend_data_2D(slope=slope, scale=scale, intercept=intercept)

    return trend_data_1D(slope=slope, scale=scale, intercept=intercept)


def LinearRegression_fit_wrapper(*args, **kwargs):
    # wrapper for LinearRegression().fit() because it has no return value - should it?
    # -> no: a class method should either change state or have a return value, it's a
    # bit awkward for testing but better overall

    lr = mesmer.stats.LinearRegression()

    lr.fit(*args, **kwargs)
    return lr.params


LR_METHOD_OR_FUNCTION = [
    mesmer.stats._linear_regression._fit_linear_regression_xr,
    LinearRegression_fit_wrapper,
]

# TEST LinearRegression class


def test_lr_params():

    lr = mesmer.stats.LinearRegression()

    with pytest.raises(ValueError, match="'params' not set"):
        lr.params

    with pytest.raises(TypeError, match="Expected params to be an xr.Dataset"):
        lr.params = None

    with pytest.raises(ValueError, match="missing the required data_vars"):
        lr.params = xr.Dataset()

    with pytest.raises(ValueError, match="missing the required data_vars"):
        lr.params = xr.Dataset(data_vars={"weights": ("x", [5])})

    with pytest.raises(ValueError, match="Expected additional variables"):
        lr.params = xr.Dataset(
            data_vars={"intercept": ("x", [5]), "fit_intercept": True}
        )

    ds = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "weights": ("x", [5]),
        }
    )
    with pytest.raises(ValueError, match="Expected additional variables"):
        lr.params = ds

    ds = xr.Dataset(
        data_vars={"intercept": ("x", [5]), "fit_intercept": True, "tas": ("x", [5])}
    )
    lr.params = ds

    xr.testing.assert_equal(ds, lr.params)

    ds = xr.Dataset(data_vars={"intercept": 5, "fit_intercept": True, "tas": 5})
    lr.params = ds
    xr.testing.assert_equal(ds, lr.params)


def test_lr_from_params():

    with pytest.raises(TypeError, match="Expected params to be an xr.Dataset"):
        mesmer.stats.LinearRegression.from_params(None)

    with pytest.raises(ValueError, match="missing the required data_vars"):
        mesmer.stats.LinearRegression.from_params(xr.Dataset())

    ds = xr.Dataset(data_vars={"intercept": 5, "fit_intercept": True, "tas": 5})
    lr = mesmer.stats.LinearRegression.from_params(ds)
    xr.testing.assert_equal(ds, lr.params)


@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_lr_predict(as_2D, data_type):
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={"intercept": ("x", [5]), "fit_intercept": True, "tas": ("x", [3])}
    )
    lr.params = params if as_2D else params.squeeze()

    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")
    pred = convert_to({"tas": tas}, data_type)

    result = lr.predict(pred)
    expected = xr.DataArray(
        [[5, 8, 11]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_lr_predict_two_predictors(as_2D, data_type):
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
            "tas2": ("x", [1]),
        }
    )
    lr.params = params if as_2D else params.squeeze()

    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")
    pred = convert_to({"tas": tas, "tas2": tas.rename("tas2")}, data_type)

    result = lr.predict(pred)
    expected = xr.DataArray(
        [[5, 9, 13]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_lr_predict_two_predictors_diffnames(as_2D, data_type):
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
            "tas2": ("x", [1]),
        }
    )
    lr.params = params if as_2D else params.squeeze()

    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")
    pred = convert_to({"tas": tas, "tas2": tas}, data_type)

    result = lr.predict(pred)
    expected = xr.DataArray(
        [[5, 9, 13]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_lr_predict_missing_superfluous(data_type):
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
            "tas2": ("x", [1]),
        }
    )
    lr.params = params
    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")

    with pytest.raises(ValueError, match="Missing predictors: 'tas', 'tas2'"):
        lr.predict(convert_to({}, data_type))

    with pytest.raises(ValueError, match="Missing predictors: 'tas'"):
        lr.predict(convert_to({"tas2": None}, data_type))

    with pytest.raises(ValueError, match="Superfluous predictors: 'something else'"):
        lr.predict(
            convert_to({"tas": tas, "tas2": tas, "something else": None}, data_type)
        )

    with pytest.raises(ValueError, match="Superfluous predictors: 'something else'"):
        lr.predict(
            convert_to({"tas": tas, "tas2": tas, "something else": None}, data_type),
            exclude="tas2",
        )

    # should not error
    lr.predict(convert_to({"tas": tas, "tas2": tas}, data_type), exclude="tas2")

    with pytest.raises(ValueError, match="Superfluous predictors: 'bar', 'foo'"):
        lr.predict(
            convert_to({"tas": tas, "tas2": tas, "foo": None, "bar": None}, data_type)
        )

    with pytest.raises(ValueError, match="Missing predictors: 'tas'"):
        lr.predict(convert_to({}, data_type), only="tas")

    with pytest.raises(ValueError, match="Superfluous predictors: 'extra'"):
        lr.predict(convert_to({"tas": tas, "extra": tas}, data_type), only="tas")

    # this should not error
    lr.predict(convert_to({"tas": tas, "tas2": tas}, data_type), only="tas")


def test_lr_predict_exclude_and_only_errors():
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
        }
    )
    lr.params = params

    with pytest.raises(TypeError, match="Cannot set both `exclude` and `only`"):
        lr.predict({}, exclude="tas", only="tas")


@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_lr_predict_exclude(as_2D, data_type):
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
            "tas2": ("x", [1]),
        }
    )
    lr.params = params if as_2D else params.squeeze()

    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")

    result = lr.predict(convert_to({"tas": tas}, data_type), exclude="tas2")
    expected = xr.DataArray(
        [[5, 8, 11]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)

    result = lr.predict(convert_to({"tas": tas}, data_type), exclude={"tas2"})
    expected = xr.DataArray(
        [[5, 8, 11]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)

    result = lr.predict(convert_to({}, data_type), exclude={"tas", "tas2"})
    expected = xr.DataArray([5], dims="x", name="prediction").to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_lr_predict_exclude_intercept(as_2D, data_type):
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
        }
    )
    lr.params = params if as_2D else params.squeeze()

    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")

    result = lr.predict(convert_to({"tas": tas}, data_type), exclude="intercept")
    expected = xr.DataArray(
        [[0, 3, 6]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)

    result = lr.predict(convert_to({}, data_type), exclude={"tas", "intercept"})
    expected = xr.DataArray([0], dims="x", name="prediction").to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_lr_predict_only(as_2D, data_type):
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
            "tas2": ("x", [1]),
        }
    )
    lr.params = params if as_2D else params.squeeze()

    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")

    result = lr.predict(convert_to({"tas": tas}, data_type), only="tas")
    expected = xr.DataArray(
        [[0, 3, 6]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()

    result = lr.predict(convert_to({"tas": tas}, data_type), only={"tas", "intercept"})
    expected = xr.DataArray(
        [[5, 8, 11]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)

    result = lr.predict(convert_to({"tas2": tas}, data_type), only={"tas2"})
    expected = xr.DataArray(
        [[0, 1, 2]], dims=("x", "time"), name="prediction"
    ).to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)

    result = lr.predict(convert_to({}, data_type), only={"intercept"})
    expected = xr.DataArray([5], dims="x", name="prediction").to_dataset()
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(result, expected)


def test_lr_predict_datatree():
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": ("x", [5]),
            "fit_intercept": True,
            "tas": ("x", [3]),
            "tas2": ("x", [1]),
        }
    )
    lr.params = params

    scen1 = xr.Dataset({"tas": trend_data_1D(), "tas2": trend_data_1D(seed=1)})
    scen2 = scen1 * 2

    pred = xr.DataTree.from_dict({"scen1": scen1, "scen2": scen2})

    result = lr.predict(pred)

    expected1 = lr.predict(scen1)
    expected2 = lr.predict(scen2)

    expected = xr.DataTree.from_dict({"scen1": expected1, "scen2": expected2})
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("as_datatree", (False, True))
def test_lr_residuals_one_var(as_datatree):
    pass

    lr = mesmer.stats.LinearRegression()

    # empty ds errors
    target = xr.Dataset()

    match = "Expected 'target' to have exactly one `data_variable`, found 0"
    with pytest.raises(ValueError, match=match):
        lr.residuals({}, target)

    # ds with more than one var errors
    target = xr.Dataset(data_vars={"a": 1, "b": 2})

    match = (
        r"Expected 'target' to have exactly one `data_variable`, found 2 \('a', 'b'\)"
    )
    with pytest.raises(ValueError, match=match):
        lr.residuals({}, target)


def test_lr_residuals_datatree():
    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={
            "intercept": 5,
            "fit_intercept": True,
            "tas": 3,
            "tas2": 1,
        }
    )

    lr.params = params

    scen1 = xr.Dataset({"tas": trend_data_1D(), "tas2": trend_data_1D(seed=1)})
    scen2 = scen1 * 2

    pred = xr.DataTree.from_dict({"scen1": scen1, "scen2": scen2})

    predicton1 = lr.predict(scen1)
    predicton2 = lr.predict(scen2)

    # slope=0 -> only the noise part
    noise1 = trend_data_1D(slope=0, seed=14)
    noise2 = trend_data_1D(slope=0, seed=41)

    target = xr.DataTree.from_dict(
        {"scen1": predicton1 + noise1, "scen2": predicton2 + noise2}
    )

    result = lr.residuals(pred, target)

    expected = xr.DataTree.from_dict(
        {
            "scen1": xr.Dataset({"residuals": noise1}),
            "scen2": xr.Dataset({"residuals": noise2}),
        }
    )

    mesmer.datatree.map_over_datasets(xr.testing.assert_allclose, result, expected)

    # =====================


@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_LR_residuals(as_2D, data_type):

    lr = mesmer.stats.LinearRegression()

    params = xr.Dataset(
        data_vars={"intercept": ("x", [5]), "fit_intercept": True, "tas": ("x", [0])}
    )
    lr.params = params if as_2D else params.squeeze()

    tas = xr.DataArray([0, 1, 2], dims="time", name="tas")
    target = xr.DataArray([[5, 8, 0]], dims=("x", "time"))
    target = target if as_2D else target.squeeze()

    result = lr.residuals(convert_to({"tas": tas}, data_type), target)
    expected = xr.DataArray([[0, 3, -5]], dims=("x", "time"), name="residuals")
    expected = expected if as_2D else expected.squeeze()

    xr.testing.assert_equal(expected, result)


# TEST XARRAY WRAPPER & LinearRegression().fit
@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_errors(lr_method_or_function, data_type):

    pred0 = trend_data_1D()
    pred1 = trend_data_1D()

    tgt = trend_data_2D()

    weights = trend_data_1D(intercept=1, slope=0, scale=0)

    # test predictors have to be dict, dataset or DataTree
    with pytest.raises(
        TypeError,
        match="predictors should be a dict or xr.Dataset, got <class 'list'>.",
    ):
        lr_method_or_function([pred0, pred1], tgt, dim="time")

    def test_unequal_coords(pred0, pred1, tgt, weights):
        with pytest.raises(ValueError, match="cannot align objects"):
            lr_method_or_function(
                convert_to({"pred0": pred0, "pred1": pred1}, data_type),
                tgt,
                dim="time",
                weights=weights,
            )

    if not data_type == "xr_dataset":
        # for xr_dataset this leads to nans in the predictors -> user responsibility
        test_unequal_coords(pred0.isel(time=slice(0, 5)), pred1, tgt, weights)
        test_unequal_coords(pred0, pred1.isel(time=slice(0, 5)), tgt, weights)
    test_unequal_coords(pred0, pred1, tgt.isel(time=slice(0, 5)), weights)
    test_unequal_coords(pred0, pred1, tgt, weights.isel(time=slice(0, 5)))

    def test_wrong_type(pred0, pred1, tgt, weights, name, preds_wrong=False):

        msg = f"Expected {name} to be an xr.DataArray"
        errortype = TypeError

        if preds_wrong:
            if data_type == "dict":
                msg = f"Expected {name} to be an xr.DataArray"
                errortype = TypeError
            else:
                msg = f"{name} should be 1D, but is 0D"
                errortype = ValueError

        with pytest.raises(errortype, match=msg):
            lr_method_or_function(
                convert_to({"pred0": pred0, "pred1": pred1}, data_type),
                tgt,
                dim="time",
                weights=weights,
            )

    test_wrong_type(
        None, pred1, tgt, weights, name="predictor: pred0", preds_wrong=True
    )
    test_wrong_type(
        pred0, None, tgt, weights, name="predictor: pred1", preds_wrong=True
    )
    test_wrong_type(pred0, pred1, None, weights, name="target")
    test_wrong_type(pred0, pred1, tgt, xr.Dataset(), name="weights")

    def test_wrong_shape(pred0, pred1, tgt, weights, name, ndim):
        with pytest.raises(ValueError, match=f"{name} should be {ndim}D"):
            lr_method_or_function(
                convert_to({"pred0": pred0, "pred1": pred1}, data_type),
                tgt,
                dim="time",
                weights=weights,
            )

    test_wrong_shape(
        pred0.expand_dims("new"), pred1, tgt, weights, name="predictor: pred0", ndim=1
    )
    test_wrong_shape(
        pred0, pred1.expand_dims("new"), tgt, weights, name="predictor: pred1", ndim=1
    )
    test_wrong_shape(
        pred0, pred1, tgt, weights.expand_dims("new"), name="weights", ndim=1
    )

    # target ndim test has a different error message
    with pytest.raises(ValueError, match="target should be 1D or 2D"):
        lr_method_or_function(
            convert_to({"pred0": pred0, "pred1": pred1}, data_type),
            tgt.expand_dims("new"),
            dim="time",
            weights=weights,
        )

    def test_missing_dim(pred0, pred1, tgt, weights, name):
        with pytest.raises(ValueError, match=f"{name} is missing the required dims"):
            lr_method_or_function(
                convert_to({"pred0": pred0, "pred1": pred1}, data_type),
                tgt,
                dim="time",
                weights=weights,
            )

    test_missing_dim(
        pred0.rename(time="t"), pred1, tgt, weights, name="predictor: pred0"
    )
    test_missing_dim(
        pred0, pred1.rename(time="t"), tgt, weights, name="predictor: pred1"
    )
    test_missing_dim(pred0, pred1, tgt.rename(time="t"), weights, name="target")
    test_missing_dim(pred0, pred1, tgt, weights.rename(time="t"), name="weights")

    with pytest.raises(
        ValueError,
        match="A predictor with the name 'weights' or 'intercept' is not allowed",
    ):
        lr_method_or_function(
            convert_to({"weights": pred0, "intercept": pred1}, data_type),
            tgt,
            dim="time",
        )
    with pytest.raises(ValueError, match="dim cannot currently be 'predictor'."):
        lr_method_or_function(
            convert_to({"pred0": pred0}, data_type), tgt, dim="predictor"
        )


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("intercept", (0, 3.14))
@pytest.mark.parametrize("slope", (0, 3.14))
@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_one_predictor(
    lr_method_or_function, intercept, slope, as_2D, data_type
):

    pred0 = trend_data_1D(slope=1, scale=0)

    tgt = trend_data_1D_or_2D(as_2D=as_2D, slope=slope, scale=0, intercept=intercept)

    result = lr_method_or_function(convert_to({"pred0": pred0}, data_type), tgt, "time")

    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_pred0 = xr.full_like(template, slope)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "fit_intercept": True,
        }
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict"])
def test_linear_regression_predictor_named_like_dim(
    lr_method_or_function, as_2D, data_type
):
    # cannot be DataTree, because data_var cannot have the same name as coord
    # cannot be Dataset because we need pred as a data_variable not coord
    slope, intercept = 0.3, 0.2
    tgt = trend_data_1D_or_2D(as_2D=as_2D, slope=slope, scale=0, intercept=intercept)

    result = lr_method_or_function(
        convert_to({"time": tgt.time}, data_type), tgt, "time"
    )
    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_time = xr.full_like(template, slope)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "time": expected_time,
            "fit_intercept": True,
        }
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_predictor_has_non_dim_coors(
    lr_method_or_function, as_2D, data_type
):
    slope, intercept = 0.3, 0.2
    tgt = trend_data_1D_or_2D(as_2D=as_2D, slope=slope, scale=0, intercept=intercept)
    tgt = tgt.assign_coords(year=("time", tgt.time.values + 1850))

    result = lr_method_or_function(
        convert_to({"pred0": tgt.time.rename("pred0")}, data_type), tgt, "time"
    )
    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_pred0 = xr.full_like(template, slope)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "fit_intercept": True,
        }
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_fit_intercept(lr_method_or_function, as_2D, data_type):

    pred0 = trend_data_1D(slope=1, scale=0)
    tgt = trend_data_1D_or_2D(as_2D=as_2D, slope=1, scale=0, intercept=1)

    result = lr_method_or_function(
        convert_to({"pred0": pred0}, data_type), tgt, "time", fit_intercept=False
    )

    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, 0)
    expected_pred0 = xr.full_like(template, 1.05084746)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "fit_intercept": False,
        }
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_no_coords(lr_method_or_function, as_2D, data_type):
    slope, intercept = 3.14, 3.14

    pred0 = trend_data_1D(slope=1, scale=0)
    tgt = trend_data_1D_or_2D(as_2D=as_2D, slope=slope, scale=0, intercept=intercept)

    # remove the coords
    pred0 = pred0.drop_vars(pred0.coords.keys())
    tgt = tgt.drop_vars(tgt.coords.keys())

    result = lr_method_or_function(convert_to({"pred0": pred0}, data_type), tgt, "time")

    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_pred0 = xr.full_like(template, slope)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "fit_intercept": True,
        }
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("intercept", (0, 3.14))
@pytest.mark.parametrize("slope", (0, 3.14))
@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_two_predictors(
    lr_method_or_function, intercept, slope, as_2D, data_type
):

    pred0 = trend_data_1D(slope=1, scale=0)
    pred1 = trend_data_1D(slope=1, scale=0)
    tgt = trend_data_1D_or_2D(as_2D=as_2D, slope=slope, scale=0, intercept=intercept)

    result = lr_method_or_function(
        convert_to({"pred0": pred0, "pred1": pred1}, data_type), tgt, "time"
    )

    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_pred0 = xr.full_like(template, slope / 2)
    expected_pred1 = xr.full_like(template, slope / 2)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "pred1": expected_pred1,
            "fit_intercept": True,
        }
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("intercept", (0, 3.14))
@pytest.mark.parametrize("slope", (0, 3.14))
@pytest.mark.parametrize("as_2D", [True, False])
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_two_predictors_diffnames(
    lr_method_or_function, intercept, slope, as_2D, data_type
):

    pred0 = trend_data_1D(slope=1, scale=0).rename("bar")
    pred1 = trend_data_1D(slope=1, scale=0).rename("foo")
    tgt = trend_data_1D_or_2D(as_2D=as_2D, slope=slope, scale=0, intercept=intercept)

    result = lr_method_or_function(
        convert_to({"pred0": pred0, "pred1": pred1}, data_type), tgt, "time"
    )

    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_pred0 = xr.full_like(template, slope / 2)
    expected_pred1 = xr.full_like(template, slope / 2)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "pred1": expected_pred1,
            "fit_intercept": True,
        }
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_two_predictors_extra_dim(lr_method_or_function, data_type):
    # add a 0D dimension/ coordinate and ensure it still works
    # NOTE: requires 3 predictors to trigger the error (might be an xarray issue)

    intercept = 1.25
    slope = 3.14

    pred0 = trend_data_1D(slope=1, scale=0)
    # add height coordinate
    pred0 = pred0.assign_coords(height=2)
    pred1 = trend_data_1D(slope=1, scale=0)

    tgt = trend_data_2D(slope=slope, scale=0, intercept=intercept)

    result = lr_method_or_function(
        convert_to({"pred0": pred0, "pred1": pred1, "pred2": pred0}, data_type),
        tgt,
        "time",
    )

    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_pred0 = xr.full_like(template, slope / 3)
    expected_pred1 = xr.full_like(template, slope / 3)
    expected_pred2 = xr.full_like(template, slope / 3)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "pred1": expected_pred1,
            "pred2": expected_pred2,
            "fit_intercept": True,
        }
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lr_method_or_function", LR_METHOD_OR_FUNCTION)
@pytest.mark.parametrize("intercept", (0, 3.14))
@pytest.mark.parametrize("data_type", ["dict", "xr_dataset"])
def test_linear_regression_weights(lr_method_or_function, intercept, data_type):

    pred0 = trend_data_1D(slope=1, scale=0)
    tgt = trend_data_2D(slope=1, scale=0, intercept=intercept)

    weights = trend_data_1D(intercept=0, slope=0, scale=0)
    weights[0] = 1

    result = lr_method_or_function(
        convert_to({"pred0": pred0}, data_type), tgt, "time", weights=weights
    )

    template = tgt.isel(time=0, drop=True)

    expected_intercept = xr.full_like(template, intercept)
    expected_pred0 = xr.zeros_like(template)

    expected = xr.Dataset(
        {
            "intercept": expected_intercept,
            "pred0": expected_pred0,
            "weights": weights,
            "fit_intercept": True,
        }
    )

    xr.testing.assert_allclose(result, expected)


# TEST NUMPY FUNCTION


@pytest.mark.parametrize(
    "predictors,target",
    (
        ([[1], [2], [3]], [1, 2]),
        ([[1, 2, 3], [2, 4, 0]], [1, 2, 2]),
    ),
)
def test_bad_shape(predictors, target):
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        mesmer.stats._linear_regression._fit_linear_regression_np(predictors, target)


@pytest.mark.parametrize(
    "predictors,target,weight",
    (
        ([[1], [2], [3]], [1, 2, 2], [1, 10]),
        ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1, 1]),
    ),
)
def test_bad_shape_weights(predictors, target, weight):
    with pytest.raises(ValueError, match="sample_weight.shape.*expected"):
        mesmer.stats._linear_regression._fit_linear_regression_np(
            predictors, target, weight
        )


def test_basic_regression():
    res = mesmer.stats._linear_regression._fit_linear_regression_np(
        [[0], [1], [2]], [0, 2, 4]
    )

    npt.assert_allclose(res, [[0, 2]], atol=1e-10)


def test_basic_regression_two_targets():
    res = mesmer.stats._linear_regression._fit_linear_regression_np(
        [[0], [1], [2]], [[0, 1], [2, 3], [4, 5]]
    )

    npt.assert_allclose(res, [[0, 2], [1, 2]], atol=1e-10)


def test_basic_regression_three_targets():
    res = mesmer.stats._linear_regression._fit_linear_regression_np(
        [[0], [1], [2]], [[0, 1, 2], [2, 3, 7], [4, 5, 12]]
    )

    # each target gets its own row in the results
    npt.assert_allclose(res, [[0, 2], [1, 2], [2, 5]], atol=1e-10)


def test_basic_regression_with_weights():
    res = mesmer.stats._linear_regression._fit_linear_regression_np(
        [[0], [1], [2], [3]], [0, 2, 4, 5], [10, 10, 10, 0.1]
    )

    npt.assert_allclose(res, [[0.0065, 1.99]], atol=1e-3)


def test_basic_regression_multidimensional():
    res = mesmer.stats._linear_regression._fit_linear_regression_np(
        [[0, 1], [1, 3], [2, 4]], [2, 7, 8]
    )

    # intercept before coefficients, in same order as columns of
    # predictors
    npt.assert_allclose(res, [[-2, -3, 4]])


def test_basic_regression_multidimensional_multitarget():
    res = mesmer.stats._linear_regression._fit_linear_regression_np(
        [[0, 1], [1, 3], [2, 4]], [[2, 0], [7, 0], [8, 5]]
    )

    # intercept before coefficients, in same order as columns of
    # predictors, rows in same order as columns of target
    npt.assert_allclose(res, [[-2, -3, 4], [5, 10, -5]])


def test_regression_with_weights_multidimensional_multitarget():
    res = mesmer.stats._linear_regression._fit_linear_regression_np(
        [[0, 1], [1, 3], [2, 4], [3, 5]],
        [[2, 0], [7, 0], [8, 5], [11, 11]],
        # extra point with low weight alters results in a minor way
        weights=[10, 10, 10, 1e-3],
    )

    # intercept before coefficients, in same order as columns of
    # predictors, rows in same order as columns of target
    npt.assert_allclose(res, [[-2, -3, 4], [5, 10, -5]], atol=1e-2)


def test_regression_order():
    x = np.array([[0, 1], [1, 3], [2, 4]])
    y = np.array([2, 7, 10])

    res_original = mesmer.stats._linear_regression._fit_linear_regression_np(x, y)

    res_reversed = mesmer.stats._linear_regression._fit_linear_regression_np(
        np.flip(x, axis=1), y
    )

    npt.assert_allclose(res_original[0][0], res_reversed[0][0], atol=1e-10)
    npt.assert_allclose(res_original[0][1:], res_reversed[0][-1:0:-1])


def test_regression_order_with_weights():
    x = np.array([[0, 1], [1, 3], [2, 4], [1, 1]])
    y = np.array([2, 7, 8, 0])
    weights = [10, 10, 10, 0.1]

    res_original = mesmer.stats._linear_regression._fit_linear_regression_np(
        x, y, weights=weights
    )
    res_reversed = mesmer.stats._linear_regression._fit_linear_regression_np(
        np.flip(x, axis=1), y, weights=weights
    )

    npt.assert_allclose(res_original[0][0], -1.89, atol=1e-2)
    npt.assert_allclose(res_original[0][0], res_reversed[0][0])
    npt.assert_allclose(res_original[0][1:], res_reversed[0][-1:0:-1])


@pytest.mark.parametrize(
    "x_shape,y_shape,exp_output_shape",
    (
        # one predictor
        ((3, 1), (3,), (1, 2)),
        ((3, 1), (3, 1), (1, 2)),
        ((3, 1), (3, 2), (2, 2)),
        # two predictors
        ((3, 2), (3,), (1, 3)),
        ((3, 2), (3, 1), (1, 3)),
        ((3, 2), (3, 2), (2, 3)),
    ),
)
def test_linear_regression_np_output_shape(x_shape, y_shape, exp_output_shape):

    x = np.random.randn(*x_shape)
    y = np.random.randn(*y_shape)

    res = mesmer.stats._linear_regression._fit_linear_regression_np(x, y)

    assert res.shape == exp_output_shape


@pytest.mark.parametrize(
    "predictors,target,weight",
    (
        ([[1], [2], [3]], [1, 2, 2], None),
        ([[1, 2, 3], [2, 4, 0]], [1, 2], [3, 1]),
    ),
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_np(predictors, target, weight, fit_intercept):
    # Unit test i.e. mocks as much as possible so that there are no
    # dependencies on external libraries etc.

    # This testing is really nasty because the function is (deliberately)
    # written without proper dependency injection. See e.g.
    # https://stackoverflow.com/a/46865495 which recommends against this
    # approach. At the moment, I can't see how to write a suitably simple
    # function for regressions that uses proper dependency injection and
    # doesn't make the interface more complicated.
    mock_regressor = mock.Mock()
    mock_regressor.intercept_ = 12 if fit_intercept else 0
    mock_regressor.coef_ = [123, -38]

    with mock.patch(
        "sklearn.linear_model.LinearRegression"
    ) as mocked_linear_regression:
        mocked_linear_regression.return_value = mock_regressor

        if weight is None:
            # check that the default behaviour is to pass None to `fit`
            # internally
            expected_weights = None
            res = mesmer.stats._linear_regression._fit_linear_regression_np(
                predictors, target, fit_intercept=fit_intercept
            )
        else:
            # check that the intended weights are indeed passed to `fit`
            # internally
            expected_weights = weight
            res = mesmer.stats._linear_regression._fit_linear_regression_np(
                predictors, target, weight, fit_intercept=fit_intercept
            )

        mocked_linear_regression.assert_called_once()
        mocked_linear_regression.assert_called_with(fit_intercept=fit_intercept)
        mock_regressor.fit.assert_called_once()
        mock_regressor.fit.assert_called_with(
            X=predictors, y=target, sample_weight=expected_weights
        )

    intercepts = np.atleast_2d(mock_regressor.intercept_).T
    coefficients = np.atleast_2d(mock_regressor.coef_)
    npt.assert_allclose(res, np.hstack([intercepts, coefficients]))
