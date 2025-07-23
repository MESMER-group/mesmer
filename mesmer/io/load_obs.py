# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in observations which are saved locally.
"""

import warnings

from mesmer._core._data import load_stratospheric_aerosol_optical_depth_obs


def load_strat_aod(time, dir_obs=None):
    """Load observed global stratospheric aerosol optical depth time series.

    Parameters
    ----------
    time : np.ndarray
        1d array of years the AOD time series is required for
    dir_obs : None
        Deprecated.

    Returns
    -------
    aod_obs : np.ndarray
        1d array of observed global stratospheric AOD time series

    Notes
    -----
    - Assumption: time covers max full extend historical period (i.e., 1850 - 2014 for
      cimp6, 1850 - 2005 for cmip5)
    """

    if dir_obs is not None:
        warnings.warn(
            "The aerosol data is now shipped with mesmer. Passing `dir_obs` to "
            "``load_strat_aod`` is no longer necessary",
            FutureWarning,
        )

    aod_obs = load_stratospheric_aerosol_optical_depth_obs(
        version="2022", resample=True
    )

    aod_obs = aod_obs.sel(time=slice(str(time[0]), str(time[-1])))

    return aod_obs
