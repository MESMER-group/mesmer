import importlib
from functools import cache

import numpy as np
import pandas as pd
import pooch
import xarray as xr

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

    if version != "2022":
        raise ValueError("No version other than '2022' is currently available.")

    aod = _load_aod_obs(resample=resample)

    return aod.copy()


# use an inner function as @cache does not nicely preserve the signature
@cache
def _load_aod_obs(*, resample):

    # TODO: use pooch
    filename = importlib.resources.files("mesmer").parent / "data/tau.line_2012.12.txt"
    # filename = _fetch_remote_data(f"tau.line_2012.12.txt")

    arr = pd.read_csv(filename, sep=r"\s+", header=2)["global"].to_numpy()

    # TODO: remove rounding and re-generate output files
    # we originally used the same data from climate explorer (see https://github.com/MESMER-group/mesmer/blob/461a1d89db4ee2e93016c8a38d126d521d460fc9/data/isaod_gl_2022.dat)
    # climate explorer rounded the data to 3 digits (compared to 4 in the NASA file)
    # this was done in fortran using a REAL number, written as f6.3 - which leads to
    # inconsistent rounding ("5" is sometimes rounded up, sometimes down)
    # do the same to avoid re-generating all files
    # https://gitlab.com/KNMI-OSS/climexp/climexp_data/-/blob/3c9f735b0e8c7aabf5e4b6c351c4870182833ea7/NASAData/saod2dat.f90#L7
    # https://gitlab.com/KNMI-OSS/climexp/climexp_data/-/blob/3c9f735b0e8c7aabf5e4b6c351c4870182833ea7/NASAData/saod2dat.f90#L42
    rounded = np.array([float(f"{v:6.3f}") for v in arr.astype(np.float32)])

    time = pd.date_range("1850-01-01", "2012-09-01", freq="MS")
    time_full = pd.date_range("1850-01-01", "2022-12-01", freq="MS")

    aod = xr.DataArray(rounded, coords={"time": time}, name="aod")
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
            "tau.line_2012.12.txt": "40b245c8fc871b75da40803c8dffee78fe9707758702297b6b9945e5ed003393",
        },
        version=f"v{mesmer.__version__}",
        version_dev="main",
    )

    # the file will be downloaded automatically the first time this is run.
    return REMOTE_RESOURCE.fetch(name)
