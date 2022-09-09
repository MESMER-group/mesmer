import pytest
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess

import mesmer.core.smoothing
from mesmer.core.utils import _check_dataarray_form
from mesmer.testing import trend_data_1D, trend_data_2D


def test_lowess_errors():
    data = trend_data_2D()

    with pytest.raises(ValueError, match="Can only pass a single dimension."):
        mesmer.core.smoothing.lowess(data, ("lat", "lon"), frac=0.3)

    with pytest.raises(ValueError, match="data should be 1-dimensional"):
        mesmer.core.smoothing.lowess(data.to_dataset(), "data", frac=0.3)


@pytest.mark.parametrize("it", [0, 3])
@pytest.mark.parametrize("frac", [0.3, 0.5])
def test_lowess(it, frac):

    data = trend_data_1D()

    result = mesmer.core.smoothing.lowess(data, "time", frac=frac, it=it)

    expected = lowess(
        data.values, data.time.values, frac=frac, it=it, return_sorted=False
    )
    expected = xr.DataArray(expected, dims="time", coords={"time": data.time})

    xr.testing.assert_allclose(result, expected)


def test_lowess_dataset():

    data = trend_data_1D()

    result = mesmer.core.smoothing.lowess(data.to_dataset(), "time", frac=0.3)

    expected = lowess(
        data.values, data.time.values, frac=0.3, it=0, return_sorted=False
    )
    expected = xr.DataArray(
        expected, dims="time", coords={"time": data.time}, name="data"
    )
    expected = expected.to_dataset()

    xr.testing.assert_allclose(result, expected)


def test_lowess_2D():
    data = trend_data_2D()

    result = mesmer.core.smoothing.lowess(data, "time", frac=0.3)

    _check_dataarray_form(
        result, "result", ndim=2, required_dims=("time", "cells"), shape=data.shape
    )
