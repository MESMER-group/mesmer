import re

import pytest
import xarray as xr

import mesmer
from mesmer.testing import _convert


def _get_volcanic_params(slope):

    return xr.Dataset(
        data_vars={
            "aod": slope,
            "intercept": 0.0,
            "fit_intercept": False,
        },
        attrs={"version": "2022"},
    )


def test_fit_volcanic_influence_errors():

    with pytest.raises(TypeError, match="Expected tas_residuals to be an xr.DataArray"):
        mesmer.volc.fit_volcanic_influence(xr.Dataset(), slice("1850", "2014"))

    with pytest.raises(ValueError, match="tas_residuals should be 1D or 2D, but is 0D"):
        mesmer.volc.fit_volcanic_influence(xr.DataArray(), slice("1850", "2014"))

    with pytest.raises(ValueError, match="tas_residuals should be 1D or 2D, but is 3D"):
        data = mesmer.testing.trend_data_3D()
        mesmer.volc.fit_volcanic_influence(data, slice("1850", "2014"))

    with pytest.raises(
        ValueError, match="tas_residuals is missing the required dims: sample"
    ):
        data = mesmer.testing.trend_data_1D()
        mesmer.volc.fit_volcanic_influence(data, slice("1850", "2014"), dim="sample")


def test_load_stratospheric_aerosol_optical_depth_data_not_changed_inplace():

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    aod.loc[{"time": slice("1900", "2000")}] = 0.25

    aod_orig = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    assert aod is not aod_orig
    assert not aod.equals(aod_orig)


@pytest.mark.parametrize(
    "hist_period", (None, slice("1850", "2014"), slice("1900", "2000"))
)
def test_fit_volcanic_influence_self(hist_period):

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    result = mesmer.volc.fit_volcanic_influence(-aod, hist_period)
    expected = _get_volcanic_params(-1.0)

    xr.testing.assert_allclose(result, expected)


def test_fit_volcanic_influence_self_shorter():
    # test it works when passing data shorter than aod obs
    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    result = mesmer.volc.fit_volcanic_influence(
        -aod.sel(time=slice("1900", "2000")), None
    )
    expected = _get_volcanic_params(-1.0)

    xr.testing.assert_allclose(result, expected)


def test_fit_volcanic_influence_self_longer():
    # test it works when passing data longer than aod obs
    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    # ends later
    time = xr.Dataset(coords={"time": xr.date_range("1900", "2040", freq="YE")})
    aod, _ = xr.align(aod, time, join="right", fill_value=0.0)

    with pytest.raises(
        ValueError, match=re.escape("Time period of passed array (1900-2039) exceeds")
    ):
        mesmer.volc.fit_volcanic_influence(-aod, None)

    # starts before
    time = xr.Dataset(coords={"time": xr.date_range("1800", "2000", freq="YE")})
    aod, _ = xr.align(aod, time, join="right", fill_value=0.0)

    with pytest.raises(
        ValueError, match=re.escape("Time period of passed array (1800-1999) exceeds")
    ):
        mesmer.volc.fit_volcanic_influence(-aod, None)


def test_fit_volcanic_warns_positive():

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    with pytest.warns(UserWarning, match="The slope of 'aod' is positive"):
        result = mesmer.volc.fit_volcanic_influence(aod, slice("1850", "2014"))

    expected = _get_volcanic_params(1.0)

    xr.testing.assert_allclose(result, expected)


@pytest.mark.filterwarnings("ignore:The slope of")
def test_fit_volcanic_influence_2D():

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    aod_ens = xr.concat([aod, -aod], dim="ens")

    result = mesmer.volc.fit_volcanic_influence(-aod_ens, slice("1850", "2014"))
    expected = _get_volcanic_params(0)

    xr.testing.assert_allclose(result, expected)


@pytest.mark.filterwarnings("ignore:The slope of")
def test_fit_volcanic_influence_hist_period():

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    # adjusting aod -> the expected return value changes
    aod.loc[{"time": slice("1951", "2000")}] = 0.0

    result = mesmer.volc.fit_volcanic_influence(-aod, slice("1850", "2014"))
    expected = _get_volcanic_params(-0.52858)

    print(result.aod.values)

    xr.testing.assert_allclose(result, expected)

    # unless we also adjust the hist period
    result = mesmer.volc.fit_volcanic_influence(-aod, slice("1850", "1950"))
    expected = _get_volcanic_params(-1)

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("slope", [0, -1, -2])
@pytest.mark.parametrize("hist_period", (None, slice("1850", "2014")))
def test_superimpose_volcanic_influence(slope, hist_period):

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    data = xr.zeros_like(aod)

    params = _get_volcanic_params(slope)

    result = mesmer.volc.superimpose_volcanic_influence(data, params, hist_period)

    expected = slope * aod

    xr.testing.assert_allclose(result, expected)


def test_superimpose_volcanic_influence_loner_errors():

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    data = xr.zeros_like(aod)

    params = _get_volcanic_params(1)

    # ends later
    time = xr.Dataset(coords={"time": xr.date_range("1900", "2040", freq="YE")})
    data, _ = xr.align(data, time, join="right", fill_value=0.0)

    with pytest.raises(
        ValueError, match=re.escape("Time period of passed array (1900-2039) exceeds")
    ):
        mesmer.volc.superimpose_volcanic_influence(data, params)

    # starts before
    time = xr.Dataset(coords={"time": xr.date_range("1800", "2000", freq="YE")})
    data, _ = xr.align(data, time, join="right", fill_value=0.0)

    with pytest.raises(
        ValueError, match=re.escape("Time period of passed array (1800-1999) exceeds")
    ):
        mesmer.volc.superimpose_volcanic_influence(data, params)


def test_superimpose_volcanic_influence_hist_period():

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    data = xr.zeros_like(aod)

    params = _get_volcanic_params(-1)

    result = mesmer.volc.superimpose_volcanic_influence(
        data, params, slice("1850", "1950")
    )

    expected = -aod
    expected.loc[{"time": slice("1951", None)}] = 0

    xr.testing.assert_allclose(result, expected)


def test_superimpose_volcanic_influence_datatree():

    slope = -1.5

    aod = mesmer.volc.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    data = xr.zeros_like(aod)
    dt = _convert(data, "DataTree")

    params = _get_volcanic_params(slope)

    result = mesmer.volc.superimpose_volcanic_influence(
        dt, params, slice("1850", "2014")
    )
    expected = slope * aod

    assert isinstance(result, xr.DataTree)
    # allclose not implemented for DataTree, just check dataset
    xr.testing.assert_allclose(result["node"].to_dataset(), expected.to_dataset())
