import numpy as np
import pytest
import xarray as xr

from mesmer.core.utils import _check_dataarray_form
from mesmer.testing import (
    assert_dict_allclose,
    trend_data_1D,
    trend_data_2D,
    trend_data_3D,
)


def test_assert_dict_allclose_type_key_errors():

    a = {"a": 1}
    b = {"a": 1, "b": 2}
    c = {"a": "a"}

    # not passing a dict
    with pytest.raises(AssertionError):
        assert_dict_allclose(None, a)

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, None)

    # differing keys
    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    # different type of value
    with pytest.raises(AssertionError):
        assert_dict_allclose(a, c)

    with pytest.raises(AssertionError):
        assert_dict_allclose(c, a)


def test_assert_dict_allclose_nonnumeric():

    a = {"a": "a"}
    b = {"a": "b"}
    c = {"a": "a"}

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)


@pytest.mark.parametrize(
    "value1, value2",
    (
        (1.0, 2.0),
        (np.array([1.0, 2.0]), np.array([2.0, 1.0])),
        (xr.DataArray([1.0]), xr.DataArray([2.0])),
        (xr.Dataset(data_vars={"a": [1.0]}), xr.Dataset(data_vars={"a": [2.0]})),
    ),
)
def test_assert_dict_allclose_unequal_values(value1, value2):

    a = {"a": value1}
    b = {"a": value2}
    c = {"a": value1}
    d = {"a": value1 + 0.99e-7}

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)

    assert_dict_allclose(a, d)
    assert_dict_allclose(d, a)


def test_assert_dict_allclose_mixed_numpy():

    a = {"a": 1}
    b = {"a": np.array(1)}
    c = {"a": np.array([1])}

    assert_dict_allclose(a, b)
    assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)


def test_assert_dict_allclose_nested():

    a = {"a": {"a": 1}}
    b = {"a": {"a": 2}}
    c = {"a": {"a": 1}}

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)


@pytest.mark.parametrize("n_timesteps", [2, 3])
@pytest.mark.parametrize("intercept", [0, 3.14])
@pytest.mark.parametrize("slope", [0, 2.72])
def test_trend_data_1D_no_scale(n_timesteps, intercept, slope):

    result = trend_data_1D(
        n_timesteps=n_timesteps, intercept=intercept, slope=slope, scale=0
    )

    _check_dataarray_form(
        result, "trend_data_1D", ndim=1, required_dims="time", shape=(n_timesteps,)
    )

    assert result[0] == intercept
    assert result[1] == intercept + slope


def test_trend_data_1D_scale():

    n_timesteps = 5
    intercept, slope = 3.14, 2.72

    result = trend_data_1D(n_timesteps=n_timesteps, intercept=intercept, slope=slope)

    _check_dataarray_form(
        result, "trend_data_1D", ndim=1, required_dims="time", shape=(n_timesteps,)
    )

    assert result[0] != intercept
    assert result[1] != intercept + slope


@pytest.mark.parametrize("n_timesteps", [2, 3])
@pytest.mark.parametrize("n_lat", [2, 3])
@pytest.mark.parametrize("n_lon", [2, 3])
@pytest.mark.parametrize("intercept", [0, 1, 3.14])
@pytest.mark.parametrize("slope", [0, 2.72])
def test_trend_data_2D_3D_no_scale(n_timesteps, n_lat, n_lon, intercept, slope):

    result = trend_data_2D(
        n_timesteps=n_timesteps,
        n_lat=n_lat,
        n_lon=n_lon,
        intercept=intercept,
        slope=slope,
        scale=0,
    )

    _check_dataarray_form(
        result,
        "trend_data_2D",
        ndim=2,
        required_dims={"time", "cells"},
        shape=(n_lat * n_lon, n_timesteps),
    )

    np.testing.assert_allclose(result[..., 0], intercept)
    np.testing.assert_allclose(result[..., 1], intercept + slope)

    result = trend_data_3D(
        n_timesteps=n_timesteps,
        n_lat=n_lat,
        n_lon=n_lon,
        intercept=intercept,
        slope=slope,
        scale=0,
    )

    _check_dataarray_form(
        result,
        "trend_data_3D",
        ndim=3,
        required_dims={"time", "lat", "lon"},
        shape=(n_timesteps, n_lat, n_lon),
    )

    np.testing.assert_allclose(result[0, ...], intercept)
    np.testing.assert_allclose(result[1, ...], intercept + slope)


def test_trend_data_2D_3D_scale():

    n_timesteps, n_lat, n_lon = 5, 2, 3
    intercept, slope = 3.14, 2.72

    result = trend_data_2D(
        n_timesteps=n_timesteps,
        n_lat=n_lat,
        n_lon=n_lon,
        intercept=intercept,
        slope=slope,
    )

    _check_dataarray_form(
        result,
        "trend_data_2D",
        ndim=2,
        required_dims={"time", "cells"},
        shape=(n_lat * n_lon, n_timesteps),
    )

    assert not np.allclose(result[..., 0], intercept)
    assert not np.allclose(result[..., 1], intercept + slope)

    result = trend_data_3D(
        n_timesteps=n_timesteps,
        n_lat=n_lat,
        n_lon=n_lon,
        intercept=intercept,
        slope=slope,
    )

    _check_dataarray_form(
        result,
        "trend_data_3D",
        ndim=3,
        required_dims={"time", "lat", "lon"},
        shape=(n_timesteps, n_lat, n_lon),
    )

    assert not np.allclose(result[0, ...], intercept)
    assert not np.allclose(result[1, ...], intercept + slope)
