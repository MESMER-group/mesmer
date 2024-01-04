# TODO: use cache instead of lru_cache once requiring python 3.9+
from functools import lru_cache

import pandas as pd
import pooch

import mesmer


def load_stratospheric_aerosol_optical_depth_obs(version="2022", resample=True):
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

    aod = _load_aod_obs(version=version, resample=resample)

    return aod.copy()


# use an inner function as @cache does not nicely preserve the signature
@lru_cache(None)
def _load_aod_obs(*, version, resample):

    filename = _fetch_remote_data(f"isaod_gl_{version}.dat")

    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        skiprows=11,
        names=("year", "month", "aod"),
        parse_dates=[["year", "month"]],
        date_format={"year_month": "%Y %m"},
        index_col="year_month",
    )

    aod = df.to_xarray().rename(year_month="time").aod

    if resample:
        aod = aod.resample(time="A").mean()

    return aod


def _fetch_remote_data(name):
    """
    uses pooch to cache files
    """

    cache_dir = pooch.os_cache("mesmer")

    REMOTE_RESSOURCE = pooch.create(
        path=cache_dir,
        # The remote data is on Github
        base_url="https://github.com/MESMER-group/mesmer/raw/{version}/data/",
        registry={
            "isaod_gl_2022.dat": "3d26e78bf0ee96a02c99e2a7a448dafda0ac847a5c914a75c7d9745e95fe68ee",
        },
        version=f"v{mesmer.__version__}",
        version_dev="main",
    )

    # the file will be downloaded automatically the first time this is run.
    return REMOTE_RESSOURCE.fetch(name)
