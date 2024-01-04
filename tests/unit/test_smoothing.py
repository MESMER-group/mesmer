import numpy as np
import pandas as pd
import pytest
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess

import mesmer
from mesmer.core.utils import _check_dataarray_form
from mesmer.testing import trend_data_1D, trend_data_2D


def test_lowess_errors():
    data = trend_data_2D()

    with pytest.raises(ValueError, match="Can only pass a single dimension."):
        mesmer.stats.lowess(data, ("lat", "lon"), frac=0.3)

    with pytest.raises(ValueError, match="data should be 1D"):
        mesmer.stats.lowess(data.to_dataset(), "data", frac=0.3)

    with pytest.raises(ValueError, match="Exactly one of ``n_steps`` and ``frac``"):
        mesmer.stats.lowess(data.to_dataset(), "lat")

    with pytest.raises(ValueError, match="Exactly one of ``n_steps`` and ``frac``"):
        mesmer.stats.lowess(data.to_dataset(), "lat", frac=0.5, n_steps=10)

    with pytest.raises(ValueError, match=r"``n_steps`` \(40\) cannot be be larger"):
        mesmer.stats.lowess(data.to_dataset(), "lat", n_steps=40)

    # numpy datetime
    time = pd.date_range("2000-01-01", periods=30)
    data = data.assign_coords(time=time)

    with pytest.raises(TypeError, match="Cannot convert coords"):
        mesmer.stats.lowess(data.to_dataset(), "time", frac=0.5)

    # cftime datetime
    time = xr.date_range("2000-01-01", periods=30, calendar="noleap")
    data = data.assign_coords(time=time)

    with pytest.raises(TypeError, match="Cannot convert coords"):
        mesmer.stats.lowess(data.to_dataset(), "time", frac=0.5)


@pytest.mark.parametrize("it", [0, 3])
@pytest.mark.parametrize("frac", [0.3, 0.5])
def test_lowess(it, frac):

    data = trend_data_1D()

    result = mesmer.stats.lowess(data, "time", frac=frac, it=it)

    expected = lowess(
        data.values, data.time.values, frac=frac, it=it, return_sorted=False
    )
    expected = xr.DataArray(expected, dims="time", coords={"time": data.time})

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("n_steps", [0, 10, 15, 30])
def test_lowess_n_steps(n_steps):

    data = trend_data_1D()

    result = mesmer.stats.lowess(data, "time", n_steps=n_steps)

    frac = n_steps / 30
    expected = mesmer.stats.lowess(data, "time", frac=frac)

    xr.testing.assert_allclose(result, expected)


def test_lowess_use_coords():

    data = trend_data_1D()
    time = data.time.values
    time[-1] = time[-1] + 10
    data = data.assign_coords(time=time)

    result = mesmer.stats.lowess(data, "time", frac=0.1)

    # time is not equally spaced: we do NOT want the same result as for use_coords=False
    not_expected = mesmer.stats.lowess(data, "time", frac=0.1, use_coords=False)

    # ensure it makes a difference
    assert not result.equals(not_expected)

    expected = lowess(
        data.values, data.time.values, frac=0.1, it=0, return_sorted=False
    )
    expected = xr.DataArray(expected, dims="time", coords={"time": data.time})

    xr.testing.assert_allclose(result, expected)


def test_lowess_dataset():

    data = trend_data_1D()

    result = mesmer.stats.lowess(data.to_dataset(), "time", frac=0.3)

    expected = lowess(
        data.values, data.time.values, frac=0.3, it=0, return_sorted=False
    )
    expected = xr.DataArray(
        expected, dims="time", coords={"time": data.time}, name="data"
    )
    expected = expected.to_dataset()

    xr.testing.assert_allclose(result, expected)


def test_lowess_dataset_missing_core_dims():

    data = trend_data_1D()
    da1 = xr.DataArray(1, name="extra1")
    da2 = xr.DataArray([3, 2, 1], dims="y", name="perpendicular")

    ds = xr.merge([data, da1, da2])

    result = mesmer.stats.lowess(ds, "time", frac=0.3)

    expected = lowess(
        data.values, data.time.values, frac=0.3, it=0, return_sorted=False
    )
    expected = xr.DataArray(
        expected, dims="time", coords={"time": data.time}, name="data"
    )
    expected = xr.merge([expected, da1, da2])

    xr.testing.assert_allclose(result, expected)


def test_lowess_2D():
    data = trend_data_2D()

    result = mesmer.stats.lowess(data, "time", frac=0.3)

    _check_dataarray_form(
        result, "result", ndim=2, required_dims=("time", "cells"), shape=data.shape
    )


def test_lowess_2D_combine_dim():

    data = trend_data_2D()

    arr = np.concatenate(np.vsplit(data.values, 6), 1).squeeze()
    x = np.concatenate([data.time.values.tolist()] * 6)

    expected = lowess(arr, x, xvals=data.time.values, frac=0.3, it=0)
    expected = xr.DataArray(expected, coords={"time": data.time.values})

    result = mesmer.stats.lowess(data, "time", combine_dim="cells", frac=0.3)

    xr.testing.assert_allclose(expected, result)

    # numpy datetime
    time = pd.date_range("2000-01-01", periods=30)
    data = data.assign_coords(time=time)

    result = mesmer.stats.lowess(
        data, "time", combine_dim="cells", frac=0.3, use_coords=False
    )
    expected = expected.assign_coords(time=time)

    xr.testing.assert_allclose(expected, result)

    # cftime datetime
    time = xr.date_range("2000-01-01", periods=30, calendar="noleap")
    data = data.assign_coords(time=time)

    result = mesmer.stats.lowess(
        data, "time", combine_dim="cells", frac=0.3, use_coords=False
    )
    expected = expected.assign_coords(time=time)

    xr.testing.assert_allclose(expected, result)


def test_lowess_2D_combine_dim_it():
    # interestingly, the result is the same for it=0

    data = trend_data_2D()

    r1 = mesmer.stats.lowess(data.mean("cells"), "time", frac=0.3)
    r2 = mesmer.stats.lowess(data, "time", combine_dim="cells", frac=0.3)
    r3 = mesmer.stats.lowess(data, "time", frac=0.3).mean("cells")

    xr.testing.assert_allclose(r1, r2)
    xr.testing.assert_allclose(r1, r3)
