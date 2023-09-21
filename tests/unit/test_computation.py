import numpy as np
import pytest
import xarray as xr

from mesmer.core.computation import gaspari_cohn


def test_gaspari_cohn_error():

    ds = xr.Dataset()

    with pytest.raises(TypeError, match="Dataset is not supported"):
        gaspari_cohn(ds)


def test_gaspari_cohn():

    data = np.array([-0.5, 0, 0.5, 1, 1.5, 2]).reshape(2, 3)

    dims = ("y", "x")
    coords = {"x": [1, 2, 3], "y": ["a", "b"]}
    attrs = {"key": "value"}

    da = xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)

    result = gaspari_cohn(da)

    expected = np.array([0.68489583, 1.0, 0.68489583, 0.20833333, 0.01649306, 0.0])
    expected = expected.reshape(2, 3)
    expected = xr.DataArray(expected, dims=dims, coords=coords, attrs=attrs)

    xr.testing.assert_allclose(expected, result, rtol=1e-6)
    assert result.attrs == attrs


def test_gaspari_cohn_np():

    assert gaspari_cohn(0) == 1
    assert gaspari_cohn(2) == 0

    values = np.arange(0, 2.1, 0.5)
    expected = np.array([1.0, 0.68489583, 0.20833333, 0.01649306, 0.0])

    actual = gaspari_cohn(values)
    np.testing.assert_allclose(expected, actual, rtol=1e-6)

    # the function is symmetric around 0
    actual = gaspari_cohn(-values)
    np.testing.assert_allclose(expected, actual, rtol=1e-6)

    # make sure shape is conserved
    values = np.arange(9).reshape(3, 3)
    assert gaspari_cohn(values).shape == (3, 3)
