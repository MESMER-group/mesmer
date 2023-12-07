# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr


def gaspari_cohn_correlation_matrices(geodist, localisation_radii):
    """Gaspari-Cohn correlation matrices for a range of localisation radii

    Parameters
    ----------
    geodist : xr.DataArray, np.ndarray
        2D array of great circle distances. Calculated from e.g. ``geodist_exact``.
    localisation_radii : iterable of float
        Localisation radii to test (in km)

    Returns
    -------
    gaspari_cohn_correlation_matrices: dict[float : :obj:`xr.DataArray`, :obj:`np.ndarray`]
        Gaspari-Cohn correlation matrix (values) for each localisation radius (keys)

    Notes
    -----
    Values in ``localisation_radii`` should not exceed 10'000 km by much because
    it can lead to correlation matrices which are not positive semidefinite.

    See Also
    --------
    gaspari_cohn, geodist_exact

    """

    out = {lr: gaspari_cohn(geodist / lr) for lr in localisation_radii}

    return out


def gaspari_cohn(r):
    """smooth, exponentially decaying Gaspari-Cohn correlation function

    Parameters
    ----------
    r : xr.DataArray, np.ndarray
        Values for which to calculate the value of the Gaspari-Cohn correlation function
        (e.g. normalised geographical distances)

    Returns
    -------
    out : xr.DataArray, , np.ndarray
        Gaspari-Cohn correlation function

    Notes
    -----
    - Smooth exponentially decaying correlation function which mimics a Gaussian
      distribution but vanishes at r = 2, i.e., 2 x the localisation radius (L)

    - based on Gaspari-Cohn 1999 [1]_ (as taken from Carrassi et al., 2018 [2]_)

    - r = d / L, with d = geographical distance in km, L = localisation radius in km

    .. [1] Gaspari, G. and Cohn, S.E. (1999), Construction of correlation functions in
       two and three dimensions. Q.J.R. Meteorol. Soc., 125: 723-757.
       https://doi.org/10.1002/qj.49712555417

    .. [2] Carrassi, A, Bocquet, M, Bertino, L, Evensen, G. Data assimilation in the
       geosciences: An overview of methods, issues, and perspectives. WIREs Clim Change.
       2018; 9:e535. https://doi.org/10.1002/wcc.535

    """

    if isinstance(r, xr.Dataset):
        raise TypeError("Dataset is not supported, please pass a DataArray")

    # make it work for numpy arrays
    if not isinstance(r, xr.DataArray):
        return _gaspari_cohn_np(r)

    out = _gaspari_cohn_np(r.values)
    out = xr.DataArray(out, dims=r.dims, coords=r.coords, attrs=r.attrs)

    return out


def _gaspari_cohn_np(r):

    r = np.abs(r)

    out = np.zeros(r.shape)

    # compute for 0 <= r < 1
    sel = (r >= 0) & (r < 1)
    r_sel = r[sel]

    out[sel] = (
        1
        - 5 / 3 * r_sel**2
        + 5 / 8 * r_sel**3
        + 1 / 2 * r_sel**4
        - 1 / 4 * r_sel**5
    )

    # compute for 1 <= r < 2
    sel = (r >= 1) & (r < 2)
    r_sel = r[sel]

    out[sel] = (
        4
        - 5 * r_sel
        + 5 / 3 * r_sel**2
        + 5 / 8 * r_sel**3
        - 1 / 2 * r_sel**4
        + 1 / 12 * r_sel**5
        - 2 / (3 * r_sel)
    )

    return out
