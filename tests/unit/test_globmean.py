import numpy as np
import pytest
import xarray as xr

import mesmer.xarray_utils as mxu


def data_lon_lat(as_dataset, x_dim="lon", y_dim="lat"):

    lon = np.arange(0.5, 360, 2)
    lat = np.arange(90, -91, -2)
    time = np.arange(3)

    data = np.random.randn(*time.shape, *lat.shape, *lon.shape)

    da = xr.DataArray(
        data,
        dims=("time", y_dim, x_dim),
        coords={"time": time, x_dim: lon, y_dim: lat},
        attrs={"key": "da_attrs"},
    )

    ds = xr.Dataset(data_vars={"data": da, "scalar": 1}, attrs={"key": "ds_attrs"})

    if as_dataset:
        return ds
    return ds.data


def test_lat_weights_scalar():

    np.testing.assert_allclose(mxu.globmean.lat_weights(90), 0.0, atol=1e-7)
    np.testing.assert_allclose(mxu.globmean.lat_weights(45), np.sqrt(2) / 2)
    np.testing.assert_allclose(mxu.globmean.lat_weights(0), 1.0, atol=1e-7)
    np.testing.assert_allclose(mxu.globmean.lat_weights(-45), np.sqrt(2) / 2)
    np.testing.assert_allclose(mxu.globmean.lat_weights(-90), 0.0, atol=1e-7)


def test_lat_weights():

    lat_coords = np.arange(90, -91, -1)
    lat = xr.DataArray(lat_coords, dims=("lat"), coords={"lat": lat_coords}, name="lat")

    expected = np.cos(np.deg2rad(lat_coords))
    expected = xr.DataArray(expected, dims=("lat"), coords={"lat": lat}, name="lat")

    result = mxu.globmean.lat_weights(lat)

    xr.testing.assert_equal(result, expected)


def test_lat_weights_2D_warn_2D():

    lat = np.arange(10).reshape(2, 5)

    with pytest.warns(UserWarning, match="non-regular grids"):
        mxu.globmean.lat_weights(lat)


@pytest.mark.parametrize("lat", [-91, 90.1])
def test_lat_weights_2D_error_90(lat):

    with pytest.raises(ValueError, match="`lat_coords` must be between -90 and 90"):
        mxu.globmean.lat_weights(lat)


def _test_calc_globmean(as_dataset, **kwargs):
    # not checking the actual mask

    data = data_lon_lat(as_dataset, **kwargs)

    y_coords = kwargs.get("y_dim", "lat")
    weights = mxu.globmean.lat_weights(data[y_coords])

    result = mxu.globmean.weighted_mean(data, weights=weights, **kwargs)

    if as_dataset:
        # ensure scalar is not broadcast
        assert result.scalar.ndim == 0
        assert result.attrs == {"key": "ds_attrs"}

        result_da = result.data
    else:
        result_da = result

    assert result_da.ndim == 1
    assert result_da.notnull().all()

    assert result_da.attrs == {"key": "da_attrs"}


@pytest.mark.parametrize("as_dataset", [True, False])
def test_calc_globmean_default(as_dataset):

    _test_calc_globmean(as_dataset)


@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("x_dim", ("x", "lon"))
@pytest.mark.parametrize("y_dim", ("y", "lat"))
def test_ocean_land_fraction(as_dataset, x_dim, y_dim):

    _test_calc_globmean(
        as_dataset,
        x_dim=x_dim,
        y_dim=y_dim,
    )
