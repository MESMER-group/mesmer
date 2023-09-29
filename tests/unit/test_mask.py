import numpy as np
import pytest
import xarray as xr

import mesmer


def data_lon_lat(as_dataset, x_coords="lon", y_coords="lat"):

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

    if as_dataset:
        return ds
    return ds.data


@pytest.mark.parametrize("threshold", ([0, 1], -0.1, 1.1))
def test_ocean_land_fraction_errors(threshold):

    data = data_lon_lat(as_dataset=True)

    with pytest.raises(
        ValueError, match="`threshold` must be a scalar between 0 and 1"
    ):
        mesmer.mask.mask_ocean_fraction(data, threshold=threshold)


def test_ocean_land_fraction_irregular():

    lon = [0, 1, 3]
    lat = [0, 1, 3]

    data = xr.Dataset(coords={"lon": lon, "lat": lat})

    with pytest.raises(
        ValueError,
        match="Cannot calculate fractional mask for irregularly-spaced coords",
    ):
        mesmer.mask.mask_ocean_fraction(data, threshold=0.5)


def test_ocean_land_fraction_threshold():
    # check that the threshold has an influence
    data = data_lon_lat(as_dataset=True)

    result_033 = mesmer.mask.mask_ocean_fraction(data, threshold=0.33)
    result_066 = mesmer.mask.mask_ocean_fraction(data, threshold=0.66)

    assert not (result_033.data == result_066.data).all()


def _test_mask(func, as_dataset, threshold=None, **kwargs):
    # not checking the actual mask

    data = data_lon_lat(as_dataset=as_dataset, **kwargs)

    kwargs = kwargs if threshold is None else {"threshold": threshold, **kwargs}
    result = func(data, **kwargs)

    if as_dataset:
        # ensure scalar is not broadcast
        assert result.scalar.ndim == 0
        assert result.attrs == {"key": "ds_attrs"}

        result_da = result.data
    else:
        result_da = result

    # ensure mask is applied
    assert result_da.isnull().any()

    assert result_da.attrs == {"key": "da_attrs"}


@pytest.mark.parametrize("as_dataset", (True, False))
def test_ocean_land_fraction_default(as_dataset):

    _test_mask(mesmer.mask.mask_ocean_fraction, as_dataset, threshold=0.5)


@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("x_coords", ("x", "lon"))
@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_ocean_land_fraction(as_dataset, x_coords, y_coords):

    _test_mask(
        mesmer.mask.mask_ocean_fraction,
        as_dataset,
        threshold=0.5,
        x_coords=x_coords,
        y_coords=y_coords,
    )


@pytest.mark.parametrize("as_dataset", (True, False))
def test_ocean_land_default(
    as_dataset,
):

    _test_mask(mesmer.mask.mask_ocean, as_dataset)


@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("x_coords", ("x", "lon"))
@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_mask_land(as_dataset, x_coords, y_coords):

    _test_mask(mesmer.mask.mask_ocean, as_dataset, x_coords=x_coords, y_coords=y_coords)


@pytest.mark.parametrize("as_dataset", (True, False))
def test_mask_antarctiva_default(
    as_dataset,
):

    _test_mask(mesmer.mask.mask_antarctica, as_dataset)


@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_mask_antarctiva(as_dataset, y_coords):

    _test_mask(mesmer.mask.mask_antarctica, as_dataset, y_coords=y_coords)


def test_mask_ocean_2D_grid():

    lon = lat = np.arange(0, 30)
    LON, LAT = np.meshgrid(lon, lat)

    dims = ("rlat", "rlon")

    data = np.random.randn(*LON.shape)

    data_2D_grid = xr.Dataset(
        {"data": (dims, data)}, coords={"lon": (dims, LON), "lat": (dims, LAT)}
    )

    data_1D_grid = xr.Dataset(
        {"data": (("lat", "lon"), data)}, coords={"lon": lon, "lat": lat}
    )

    result = mesmer.mask.mask_ocean(data_2D_grid)
    expected = mesmer.mask.mask_ocean(data_1D_grid)

    # the Datasets don't have equal coords but their arrays should be the same
    np.testing.assert_equal(result.data.values, expected.data.values)
