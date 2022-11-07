import numpy as np
import pytest
import xarray as xr

import mesmer.xarray_utils as mxu


def data_lon_lat(x_coords="lon", y_coords="lat"):

    lon = np.arange(0.5, 360, 2)
    lat = np.arange(90, -91, -2)

    data = np.random.randn(*lat.shape, *lon.shape)

    da = xr.DataArray(
        data,
        dims=(y_coords, x_coords),
        coords={x_coords: lon, y_coords: lat},
        attrs={"key": "da_attrs"},
    )

    ds = xr.Dataset(data_vars={"data": da, "scalar": 1}, attrs={"key": "ds_attrs"})

    return ds


@pytest.mark.parametrize("threshold", ([0, 1], -0.1, 1.1))
def test_mask_land_fraction_errors(threshold):

    data = data_lon_lat()

    with pytest.raises(
        ValueError, match="`threshold` must be a scalar between 0 and 1"
    ):
        mxu.mask_land_fraction(data, threshold=threshold)


def test_mask_land_fraction_irregular():

    lon = [0, 1, 3]
    lat = [0, 1, 3]

    data = xr.Dataset(coords={"lon": lon, "lat": lat})

    with pytest.raises(
        ValueError,
        match="Cannot calculate fractional mask for irregularly-spaced coords",
    ):
        mxu.mask_land_fraction(data, threshold=0.5)


def test_mask_land_fraction_threshold():
    # check that the threshold has an influence
    data = data_lon_lat()

    result_033 = mxu.mask_land_fraction(data, threshold=0.33)
    result_066 = mxu.mask_land_fraction(data, threshold=0.66)

    assert not (result_033.data == result_066.data).all()


def _test_mask(func, threshold=None, **kwargs):
    # not checking the actual mask

    data = data_lon_lat(**kwargs)

    kwargs = kwargs if threshold is None else {"threshold": threshold, **kwargs}
    result = func(data, **kwargs)

    # ensure scalar is not broadcast
    assert result.scalar.ndim == 0

    # ensure no nan in data
    assert result.data.notnull().all()

    # ensure mask is applied
    assert result.data.isnull().any()

    assert result.attrs == {"key": "ds_attrs"}
    assert result.data.attrs == {"key": "da_attrs"}


def test_mask_land_fraction_default():

    _test_mask(mxu.mask_land_fraction, threshold=0.5)


@pytest.mark.parametrize("x_coords", ("x", "lon"))
@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_mask_land_fraction(x_coords, y_coords):

    _test_mask(
        mxu.mask_land_fraction, threshold=0.5, x_coords=x_coords, y_coords=y_coords
    )


def test_mask_land_default():

    _test_mask(mxu.mask_land)


@pytest.mark.parametrize("x_coords", ("x", "lon"))
@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_mask_land(x_coords, y_coords):

    _test_mask(mxu.mask_land, x_coords=x_coords, y_coords=y_coords)


def test_mask_antarctiva_default():

    _test_mask(mxu.mask_antarctica)


@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_mask_antarctiva(y_coords):

    _test_mask(mxu.mask_antarctica, y_coords=y_coords)
