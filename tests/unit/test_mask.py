import numpy as np
import pytest
import xarray as xr

import mesmer


def data_lon_lat(datatype, x_coords="lon", y_coords="lat"):

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

    if datatype == "Dataset":
        return ds
    elif datatype == "DataTree":
        return xr.DataTree.from_dict({"node": ds})
    return ds.data


@pytest.mark.parametrize("threshold", ([0, 1], -0.1, 1.1))
def test_ocean_land_fraction_errors(threshold):

    data = data_lon_lat("Dataset")

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
    data = data_lon_lat("Dataset")

    result_033 = mesmer.mask.mask_ocean_fraction(data, threshold=0.33)
    result_066 = mesmer.mask.mask_ocean_fraction(data, threshold=0.66)

    assert not (result_033.data == result_066.data).all()


def _test_mask(func, datatype, threshold=None, **kwargs):
    # not checking the actual mask

    data = data_lon_lat(datatype, **kwargs)

    print(data)

    kwargs = kwargs if threshold is None else {"threshold": threshold, **kwargs}
    result = func(data, **kwargs)

    if datatype == "DataTree":
        assert isinstance(result, xr.DataTree)
        result = result["node"].to_dataset()

    if datatype in ("DataTree", "Dataset"):
        # ensure scalar is not broadcast
        assert result.scalar.ndim == 0

        # TODO: enable again after https://github.com/pydata/xarray/pull/10219
        if datatype != "DataTree":
            assert result.attrs == {"key": "ds_attrs"}

        result_da = result.data
    else:
        result_da = result

    # ensure mask is applied
    assert result_da.isnull().any()

    assert result_da.attrs == {"key": "da_attrs"}


def test_ocean_land_fraction_default(datatype):

    _test_mask(mesmer.mask.mask_ocean_fraction, datatype, threshold=0.5)


@pytest.mark.parametrize("x_coords", ("x", "lon"))
@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_ocean_land_fraction(datatype, x_coords, y_coords):

    _test_mask(
        mesmer.mask.mask_ocean_fraction,
        datatype,
        threshold=0.5,
        x_coords=x_coords,
        y_coords=y_coords,
    )


def test_ocean_land_default(datatype):

    _test_mask(mesmer.mask.mask_ocean, datatype)


@pytest.mark.parametrize("x_coords", ("x", "lon"))
@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_mask_land(datatype, x_coords, y_coords):

    _test_mask(mesmer.mask.mask_ocean, datatype, x_coords=x_coords, y_coords=y_coords)


def test_mask_antarctiva_default(datatype):

    _test_mask(mesmer.mask.mask_antarctica, datatype)


@pytest.mark.parametrize("y_coords", ("y", "lat"))
def test_mask_antarctiva(datatype, y_coords):

    _test_mask(mesmer.mask.mask_antarctica, datatype, y_coords=y_coords)


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
