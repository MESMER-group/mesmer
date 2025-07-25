import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mesmer


def test_load_stratospheric_aerosol_optical_depth_obs_wrong_version():

    with pytest.raises(
        ValueError, match="No version other than '2022' is currently available."
    ):
        mesmer.data.load_stratospheric_aerosol_optical_depth_obs(version="2021")


def test_load_stratospheric_aerosol_optical_depth_data():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    time = pd.date_range("1850", "2023", freq="YE")
    time = xr.DataArray(time, dims="time", coords={"time": time})

    xr.testing.assert_equal(aod.time, time)

    np.testing.assert_allclose(0.0036, aod[0])
    np.testing.assert_allclose(0.0, aod[-1])


def test_load_stratospheric_aerosol_optical_depth_data_not_changed_inplace():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    aod.loc[{"time": slice("1900", "2000")}] = 0.25

    aod_orig = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    assert aod is not aod_orig
    assert not aod.equals(aod_orig)


def test_load_stratospheric_aerosol_optical_depth_data_no_resample():

    aod = mesmer.data.load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=False
    )

    time = pd.date_range("1850", "2022-12", freq="MS")
    time = xr.DataArray(time, dims="time", coords={"time": time})

    xr.testing.assert_equal(aod.time, time)

    np.testing.assert_allclose(0.0044, aod[0])
    np.testing.assert_allclose(0.0, aod[-1])
