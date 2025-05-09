import numpy as np
import pytest
import regionmask
import shapely.geometry
import xarray as xr

from mesmer.core.regionmaskcompat import (
    InvalidCoordsError,
    mask_3D_frac_approx,
    sample_coord,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:`mask_3D_frac_approx` has been deprecated")
]


@pytest.fixture
def small_region():
    poly = shapely.geometry.box(0, 0, 1, 1)
    return regionmask.Regions([poly])


def test_sample_coord():

    actual = sample_coord([0, 10])
    expected = np.arange(-4.5, 14.6, 1)
    np.testing.assert_allclose(actual, expected)

    actual = sample_coord([0, 1, 2])
    expected = np.arange(-0.45, 2.46, 0.1)
    np.testing.assert_allclose(actual, expected)


def test_mask_3D_frac_approx_deprecated(small_region):

    lon = lat = np.array([0, 1, 2])

    with pytest.warns(FutureWarning, match="`mask_3D_frac_approx` has been deprecated"):
        mask_3D_frac_approx(small_region, lon, lat)


@pytest.mark.parametrize("dim", ["lon", "lat"])
@pytest.mark.parametrize("invalid_coords", ([0, 1, 3], [[0, 1, 2]]))
def test_mask_3D_frac_approx_wrong_coords(small_region, dim, invalid_coords):

    valid_coords = [0, 1, 2]
    latlon = {"lon": valid_coords, "lat": valid_coords}
    # replace one of the coords with invalid values
    latlon[dim] = invalid_coords

    with pytest.raises(
        InvalidCoordsError, match="'lon' and 'lat' must be 1D and equally spaced."
    ):
        mask_3D_frac_approx(small_region, **latlon)


@pytest.mark.parametrize("lat", ((-91, 90), (-90, 92), (-91, 92)))
def test_mask_3D_frac_approx_lon_beyond_90(small_region, lat):

    lat = np.arange(*lat)
    lon = np.arange(0, 360, 10)

    with pytest.raises(InvalidCoordsError, match=r"lat must be between \-90 and \+90"):
        mask_3D_frac_approx(small_region, lon, lat)


def test_mask_3D_frac_approx_coords():
    # ensure coords are the same (as they might by averaged)

    lat = np.arange(90, -90, -1)
    lon = np.arange(0, 120, 1)

    r = shapely.geometry.box(0, -90, 120, 90)
    r = regionmask.Regions([r])

    result = mask_3D_frac_approx(r, lon, lat)

    np.testing.assert_equal(result.lon.values, lon)
    np.testing.assert_equal(result.lat.values, lat)

    assert result.abbrevs.item() == "r0"
    assert result.names.item() == "Region0"
    assert result.region.item() == 0


def test_mask_3D_frac_approx_poles():
    # all points should be 1 for a global mask

    lat = np.arange(90, -91, -5)
    lon = np.arange(0, 360, 5)

    r = shapely.geometry.box(0, -90, 360, 90)
    r = regionmask.Regions([r])

    result = mask_3D_frac_approx(r, lon, lat)
    assert (result == 1).all()


def test_mask_3D_frac_approx_southpole():
    # all at the southpole should be 1 - irrespective of where exactly the southernmost
    # latitude is

    lon = np.arange(0, 31, 5)

    r = shapely.geometry.box(0, -90, 360, -85)
    r = regionmask.Regions([r])

    for offset in np.arange(0, 1, 0.05):
        lat = np.arange(-90, -80, 1) + offset
        result = mask_3D_frac_approx(r, lon, lat)
        assert (result.isel(lat=0) == 1).all()


def test_mask_3D_frac_approx_northpole():
    # all at the southpole should be 1 - irrespective of where exactly the southernmost
    # latitude is

    lon = np.arange(0, 31, 5)

    r = shapely.geometry.box(0, 85, 360, 90)
    r = regionmask.Regions([r])

    for offset in np.arange(0, 1, 0.05):
        lat = np.arange(90, 80, -1) - offset
        result = mask_3D_frac_approx(r, lon, lat)
        assert (result.isel(lat=0) == 1).all()


def test_mask_3D_frac_approx():

    lon = np.array([15, 30])
    lat = np.array([15, 30])

    # the center of the region is at 15°
    r = shapely.geometry.box(0, 0, 30, 30)
    r = regionmask.Regions([r])

    result = mask_3D_frac_approx(r, lon, lat)

    expected = [[[1, 0.5], [0.5, 0.25]]]
    expected = xr.DataArray(
        expected,
        dims=("region", "lat", "lon"),
        coords={
            "lat": lat,
            "lon": lon,
            "abbrevs": ("region", ["r0"]),
            "region": [0],
            "names": ("region", ["Region0"]),
        },
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lat_name", ("lat", "y"))
@pytest.mark.parametrize("lon_name", ("lon", "x"))
def test_mask_3D_frac_approx_coord_names(lat_name, lon_name):

    lon = np.array([15, 30])
    lat = np.array([15, 30])
    ds = xr.Dataset(coords={lon_name: lon, lat_name: lat})

    # the center of the region is at 15°
    r = shapely.geometry.box(0, 0, 30, 30)
    r = regionmask.Regions([r])

    result = mask_3D_frac_approx(r, ds[lon_name], ds[lat_name])

    expected = [[[1, 0.5], [0.5, 0.25]]]
    expected = xr.DataArray(
        expected,
        dims=("region", lat_name, lon_name),
        coords={
            lat_name: lat,
            lon_name: lon,
            "abbrevs": ("region", ["r0"]),
            "region": [0],
            "names": ("region", ["Region0"]),
        },
    )

    xr.testing.assert_allclose(result, expected)
