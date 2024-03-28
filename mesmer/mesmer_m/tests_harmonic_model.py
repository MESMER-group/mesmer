# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Tests for monthly trend module of MESMER-M. We check:

    1. The monthly cycle is centred around the yearly temp.

    2. Approximately the correct order is chosen when using the BIC:
        a. How this depends on length of time series: using nr_runs
        b. How this evolves with order
"""

import numpy as np
import xarray as xr

from mesmer.mesmer_m import harmonic_model as hm

# from tqdm import tqdm_notebook as tqdm


def dummy_yearly_trend(nr_runs):
    """
    Generate random non-linear yearly time series from 1870-2100

    Parameters
    ----------
    nr_runs : Integer
             Number of hypothetical initial-condition ensemble members.

    Outputs
    -------

    y : array-like of size (231*nr_runs)
       Hypothetical yearly temperature time series

    """

    coeffs = np.random.randint(low=5, high=20, size=3) / 10
    x = np.linspace(0, 2, 231)

    y = np.tile(coeffs[0] * (x**2) + coeffs[1] * x + coeffs[2], nr_runs)
    y += np.random.normal(loc=0, scale=1, size=y.shape[0])

    return y


def dummy_harmonic_data(nr_runs, n=2, x=None):
    """
    Generate dummy data with a seasonal cycle

    Parameters
    ----------

    nr_runs : Integer
             Number of hypothetical initial-condition ensemble members.

    n : Integer
       Order of the fourier series (Optional, default=3).

    x : None or array-like of size (n_samples)
        Yearly values to calculate monthly values with (Optional, default=None).


    Returns
    -------

    y : array-like of size (231*12*nr_runs)
       Data consisting of a seasonal cycle of order n with normally distributed noise added ontop.

    """

    if x is None:
        x = dummy_yearly_trend(nr_runs)  # generate random yearly data

    mon = np.tile(
        np.arange(1, 13), x.shape[0]
    )  # generate month values for length of yearly data
    x = np.repeat(
        x, 12
    )  # expand yearly data so each month has its corresponding yearly temperature

    coeffs = np.random.randint(low=5, high=20, size=n * 4) / 10
    coeffs[2] = 1  # impose centring around yearly temperatures
    coeffs[3] = 0  # impose centring around yearly temperatures

    y = hm.generate_fourier_series_np(coeffs, n, x, mon)  # generate fourier series

    y += np.random.normal(loc=0, scale=2, size=y.shape[0])  # add noise

    return x, coeffs, y


def dummy_fit(nr_runs, n=3, fit_to_bic=False):
    """
    Fits on dummy data with a seasonal cycle

    Parameters
    ----------

    nr_runs: Integer
             Number of hypothetical initial-condition ensemble members.

    n: Integer
       Order of the fourier series (Optional, default=3).

    fit_to_bic: Boolean
                Optional, whether to fit to bic or simply fit (default = False).

    Returns
    -------

    x : array-like of size (n_samples,)
       Original yearly temperature.

    y : array-like of size (n_samples,)
       Original monthly temperatures.

    coeffs_org : array-like of size (n*4-2)
                Original coefficients.

    outputs of fit_to_bic or fit_fourier_series functions (see harmonic_model.py)

    """

    x, coeffs_org, y = dummy_harmonic_data(nr_runs, n)

    if fit_to_bic:

        return x, y, coeffs_org, hm.fit_to_bic_np(x[::12], y, 8, repeat=True)

    else:

        return x, y, coeffs_org, hm.fit_fourier_series_np(x[::12], y, n, repeat=True)


def BIC_gridsearch_domain(max_n, max_nr_runs):
    """
    Fits on dummy data with a seasonal cycle

    Parameters
    ----------

    max_n : Integer
           Maximum order of the fourier series to go up to.

    max_nr_runs : Integer
                 Maximum number of hypothetical initial-condition ensemble members to check for.

    Returns
    -------

    n_sel : array-like of size (max_n, max_nr_runs,)
           The selected order based on BIC score across the whole max_n and max_nr_run domain.

    mse : array-like of size (max_n, max_nr_runs,)
         Error in the chosen model's predictions across the whole max_n and max_nr_run domain.

    y_acts : Dict. with keys 1 to max_n
            Target y variable as dictionary of arrays dims (n_runs,231*12) and keys representing order of harmonics.

    y_preds : Dict. with keys 1 to max_n
             Same as y_acts but with predicted values from harmonic model instead

    """

    n_sel = np.zeros([max_n, max_nr_runs])
    mse = np.zeros([max_n, max_nr_runs, 12])
    y_acts = {}
    y_preds = {}

    for i_nr in range(1, max_nr_runs + 1):

        y_acts[i_nr] = np.zeros([max_n + 1, i_nr * 231 * 12])
        y_preds[i_nr] = np.zeros([max_n + 1, i_nr * 231 * 12])

        for i_n in range(1, max_n + 1):  # tqdm(range(1, max_n + 1)):

            _, y, _, bic_results = dummy_fit(i_nr, n=i_n, fit_to_bic=True)

            y_acts[i_nr][i_n, :] = y
            y_preds[i_nr][i_n, :] = bic_results[2]

            n_sel[i_n - 1, i_nr - 1] = bic_results[0]

            mse[i_n - 1, i_nr - 1, :] = np.mean(
                (y.reshape(-1, 12) - y_preds[i_nr][i_n, :].reshape(-1, 12)) ** 2, axis=0
            )

    return n_sel, mse, y_acts, y_preds


def check_fit_to_bic_xr(nr_runs, cells, max_n):
    """
    Performs dummy fit of fit_to_bic_xr xarray wrapper function

    Parameters
    ----------

    nr_runs : Integer
             Hypothetical number of initial condition ensemble members to fit over.

    cells : Integer
           Hypothetical number of grid points to generate xarray dataArray for.

    max_n : Integer
           Maximum order of Fourier Series to fit for.

    Outputs
    -------

    output : xr.DataArray
             Test fitted Fourier Series composed of (n_sel, coefficients, predictions)

    """

    times_mon = np.arange(
        "1870-01", "2101-01", np.timedelta64(1, "M"), dtype="datetime64"
    )

    X = xr.DataArray(
        np.hstack(
            (
                [
                    np.repeat(dummy_yearly_trend(nr_runs), 12).reshape(-1, 1)
                    for i_cell in np.arange(cells)
                ]
            )
        ),
        dims=["time", "cell"],
        coords={"time": np.tile(times_mon, nr_runs)},
    )

    Y = xr.DataArray(
        np.hstack(
            (
                [
                    dummy_harmonic_data(nr_runs, x=X.values[::12, i_cell])[2].reshape(
                        -1, 1
                    )
                    for i_cell in np.arange(cells)
                ]
            )
        ),
        dims=["time", "cell"],
        coords={"time": np.tile(times_mon, nr_runs)},
    )

    output = hm.fit_to_bic_xr(X, Y, max_n)

    return output
