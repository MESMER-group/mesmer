import pytest
import xarray as xr

import mesmer


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

    with pytest.raises(TypeError):
        mesmer.volc.fit_volcanic_influence(xr.Dataset(), slice("1850", "2014"))

    with pytest.raises(ValueError):
        mesmer.volc.fit_volcanic_influence(xr.DataArray(), slice("1850", "2014"))

    with pytest.raises(ValueError):
        data = mesmer.testing.trend_data_3D()
        mesmer.volc.fit_volcanic_influence(data, slice("1850", "2014"))

    with pytest.raises(ValueError):
        data = mesmer.testing.trend_data_1D()
        mesmer.volc.fit_volcanic_influence(data, slice("1850", "2014"), dim="sample")


def test_fit_volcanic_influence_self():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    result = mesmer.volc.fit_volcanic_influence(-aod, slice("1850", "2014"))
    expected = _get_volcanic_params(-1)

    xr.testing.assert_identical(result, expected)


def test_fit_volcanic_warns_positive():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    with pytest.warns(UserWarning, match="The slope of 'aod' is positive"):
        result = mesmer.volc.fit_volcanic_influence(aod, slice("1850", "2014"))

    expected = _get_volcanic_params(1)

    xr.testing.assert_identical(result, expected)


@pytest.mark.filterwarnings("ignore:The slope of")
def test_fit_volcanic_influence_2D():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    aod = xr.concat([aod, -aod], dim="ens")

    result = mesmer.volc.fit_volcanic_influence(-aod, slice("1850", "2014"))
    expected = _get_volcanic_params(0)

    xr.testing.assert_allclose(result, expected)


@pytest.mark.filterwarnings("ignore:The slope of")
def test_fit_volcanic_influence_hist_period():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    # adjusting aod -> the expected return value changes
    aod.loc[{"time": slice("1951", "2000")}] = 0.0

    result = mesmer.volc.fit_volcanic_influence(-aod, slice("1850", "2014"))
    expected = _get_volcanic_params(-0.52908)

    print(result.aod.values)

    xr.testing.assert_allclose(result, expected)

    # unless we also adjust the hist period
    result = mesmer.volc.fit_volcanic_influence(-aod, slice("1850", "1950"))
    expected = _get_volcanic_params(-1)

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("slope", [0, -1, -2])
def test_superimpose_volcanic_influence(slope):

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    data = xr.zeros_like(aod)

    params = _get_volcanic_params(slope)

    result = mesmer.volc.superimpose_volcanic_influence(
        data, params, slice("1850", "2014")
    )

    expected = slope * aod

    xr.testing.assert_allclose(result, expected)


def test_superimpose_volcanic_influence_hist_period():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
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
