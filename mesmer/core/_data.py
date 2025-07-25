from functools import cache

import pandas as pd
import pooch
import xarray as xr

import mesmer


def load_stratospheric_aerosol_optical_depth_obs(*, version="2022", resample=True):
    """load stratospheric aerosol optical depth data - a proxy for volcanic activity

    Parameters
    ----------
    version : str, default: "2022"
        Which version of the dataset to load. Currently only "2022" is available.
    resample : bool, default: True
        Whether to resample the data to annual resolution.

    Returns
    -------
    stratospheric_aerosol_optical_depth_obs : xr.DataArray
        DataArray of stratospheric aerosol optical depth observations.
    """

    if version != "2022":
        raise ValueError("No version other than '2022' is currently available.")

    aod = _load_aod_obs(resample=resample)

    return aod.copy()


# use an inner function as @cache does not nicely preserve the signature
@cache
def _load_aod_obs(*, resample):

    filename = _fetch_remote_data("obs/tau.line_2012.12.txt")

    arr = pd.read_csv(filename, sep=r"\s+", header=2)["global"].to_numpy()

    time = pd.date_range("1850-01-01", "2012-09-01", freq="MS")
    time_full = pd.date_range("1850-01-01", "2022-12-01", freq="MS")

    aod = xr.DataArray(arr, coords={"time": time}, name="aod")
    aod = aod.reindex_like(xr.Dataset(coords={"time": time_full}), fill_value=0.0)

    if resample:
        aod = aod.resample(time="YE").mean()

    return aod


def _fetch_remote_data(name):
    """
    uses pooch to cache files
    """

    cache_dir = pooch.os_cache("mesmer")

    REMOTE_RESOURCE = pooch.create(
        path=cache_dir,
        # The remote data is on Github
        base_url="https://github.com/MESMER-group/mesmer/raw/{version}/data/",
        registry={
            "obs/tau.line_2012.12.txt": "9aa43f83bfc8e69b9e4c21c894a7a2e7b5ddf7ec32d2e9b55b12ce5bddc36451",
        },
        version=f"v{mesmer.__version__}",
        version_dev="main",
    )

    # the file will be downloaded automatically the first time this is run.
    return REMOTE_RESOURCE.fetch(name)
