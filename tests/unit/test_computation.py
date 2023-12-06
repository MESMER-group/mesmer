import numpy as np
import pytest
import xarray as xr

from mesmer.core.geospatial import geodist_exact
from mesmer.stats import gaspari_cohn, gaspari_cohn_correlation_matrices
from mesmer.testing import assert_dict_allclose


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


def test_calc_geodist_dataset_error():

    ds = xr.Dataset()
    da = xr.DataArray()

    with pytest.raises(TypeError, match="Dataset is not supported"):
        geodist_exact(ds, ds)

    with pytest.raises(TypeError, match="Dataset is not supported"):
        geodist_exact(ds, da)

    with pytest.raises(TypeError, match="Dataset is not supported"):
        geodist_exact(da, ds)


def test_calc_geodist_dataarray_equal_dims_required():

    lon = xr.DataArray([0], dims="lon")
    lat = xr.DataArray([0], dims="lat")

    with pytest.raises(AssertionError, match="lon and lat have different dims"):
        geodist_exact(lon, lat)


@pytest.mark.parametrize("as_dataarray", [True, False])
def test_calc_geodist_not_same_shape_error(as_dataarray):

    lon, lat = [0, 0], [0]

    if as_dataarray:
        lon, lat = xr.DataArray(lon), xr.DataArray(lat)

    with pytest.raises(ValueError, match="lon and lat must be 1D arrays"):
        geodist_exact(lon, lat)


@pytest.mark.parametrize("as_dataarray", [True, False])
def test_calc_geodist_not_1D_error(as_dataarray):

    lon = lat = [[0, 0]]

    if as_dataarray:
        lon, lat = xr.DataArray(lon), xr.DataArray(lat)

    with pytest.raises(ValueError, match=".*of the same shape"):
        geodist_exact(lon, lat)


@pytest.mark.parametrize("lon", [[0, 0], [0, 360], [1, 361], [180, -180]])
@pytest.mark.parametrize("as_dataarray", [True, False])
def test_geodist_exact_equal(lon, as_dataarray):
    """test points with distance 0"""

    expected = np.array([[0, 0], [0, 0]])

    lat = [0, 0]

    if as_dataarray:
        lon = xr.DataArray(lon)

    result = geodist_exact(lon, lat)
    np.testing.assert_equal(result, expected)
    # when passing only one DataArray it's also returned as np.array
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("as_dataarray", [True, False])
def test_geodist_exact(as_dataarray):
    """test some random points"""

    lon = [-180, 0, 3]
    lat = [0, 0, 5]

    if as_dataarray:
        lon = xr.DataArray(lon, dims="gp", coords={"lon": ("gp", lon)})
        lat = xr.DataArray(lat, dims="gp", coords={"lat": ("gp", lat)})

    result = geodist_exact(lon, lat)
    expected = np.array(
        [
            [0.0, 20003.93145863, 19366.51816487],
            [20003.93145863, 0.0, 645.70051988],
            [19366.51816487, 645.70051988, 0.0],
        ]
    )

    if as_dataarray:

        expected = xr.DataArray(expected, dims=("gp_i", "gp_j"))
        xr.testing.assert_allclose(expected, result)

    else:

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("localisation_radii", [[1000, 2000], range(5000, 5001)])
@pytest.mark.parametrize("as_dataarray", [True, False])
def test_gaspari_cohn_correlation_matrices(localisation_radii, as_dataarray):

    lon = np.arange(0, 31, 5)
    lat = np.arange(-45, 46, 30)
    lat, lon = np.meshgrid(lat, lon)
    lon, lat = lon.flatten(), lat.flatten()

    if as_dataarray:
        lon = xr.DataArray(lon, dims="gridpoint")
        lat = xr.DataArray(lat, dims="gridpoint")

    geodist = geodist_exact(lon, lat)

    result = gaspari_cohn_correlation_matrices(geodist, localisation_radii)

    expected = {lr: gaspari_cohn(geodist / lr) for lr in localisation_radii}

    assert_dict_allclose(expected, result)
