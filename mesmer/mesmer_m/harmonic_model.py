# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M
"""

import numpy as np
import xarray as xr
from scipy import optimize

import mesmer


def generate_fourier_series_np(coeffs, n, year, mon):
    """construct the Fourier Series

    Parameters
    ----------
    coeffs : array-like of shape (4*n-2)
        coefficients of Fourier Series.
    n : Integer
        Order of the Fourier Series.
    year : array-like of shape (n_samples,)
        yearly temperature values.
    mon : array-like of shape (n_samples,)
        month values (0-11).

    Returns
    -------
    predictions: array-like of shape (n_samples,)
        Fourier Series of order n calculated over x and mon with coeffs.

    """

    return sum(
        [
            (coeffs[idx] * year + coeffs[idx + 1]) * np.sin(np.pi * i * (mon % 12 + 1) / 6)
            + (coeffs[idx + 2] * year + coeffs[idx + 3])
            * np.cos(np.pi * i * (mon % 12 + 1) / 6)
            for i, idx in enumerate(np.arange(n * 4, step=4))
        ]
    )


def fit_fourier_series_np(yearly_predictor, monthly_target, n):
    """execute fitting of the harmonic model/fourier series

    Parameters
    ----------

    yearly_predictor : array-like of shape (n_samples,)
        Yearly temperature values to predict with.

    monthly_target : array-like of shape (n_samples*12,)
        Target monthly temperature values.

    n : Integer
        Order of the Fourier Series.

    Method
    ------

    We use np.linalg.lstsq as a simple solver, given we have the equation:

    sum_{i=0}^{n} [(a{i}*x + b{i})*np.cos(\frac{np.pi*i*(mon%12+1)}{6}+(c{i}*x + d{i})*np.cos(\frac{np.pi*i*(mon%12+1)}{6})]

    we expect the input A to be of size n_samples, n*4-2 such that each column contains each coefficient's respective variable


    Returns
    -------
    coeffs : array-like of shape (4*n-2)
        Fitted coefficients of Fourier series.

    y : array-like of shape (n_samples*12,)
        Predicted monthly temperature values.

    """

    # each month scales to the yearly value at that timestep so need to repeat
    x_train = yearly_predictor #np.repeat(yearly_predictor, 12)

    # also get monthly values
    mon_train = np.tile(np.arange(1, 13), int(x_train.shape[0] / 12))
    # for simplicity's sake we take month values in there harmonic form
    mon_train = (np.pi * (mon_train % 12 + 1)) / 6

    def fun(coefs, n, x_train, mon_train, mon_target):
        """loss function for fitting fourier series in scipy.optimize.least_squares"""
        loss = np.mean((generate_fourier_series_np(coefs, n, x_train, mon_train) - mon_target) ** 2)

        return loss

    c0 = np.zeros(n * 4)
    c0[2] = 1 # why?
    # c0[3] = 0 # not necessary ?

    # NOTE: this seems to select less 'orders' than the scipy one
    # np.linalg.lstsq(A, y)[0]

    coeffs = optimize.least_squares(
        fun, c0, args=(n, x_train, mon_train, monthly_target), loss="cauchy"
    ).x

    preds = generate_fourier_series_np(coeffs, n, x_train, mon_train)

    return coeffs, preds


def calculate_bic(n_samples, n_order, mse):
    """calculate Bayesian Information Criteria (BIC)

    Parameters
    ----------
    n_samples : Integer
        size of training set.
    n_order : Integer
        Order of Fourier Series.
    mse : Float
        Mean-squared error.

    Returns
    -------
    BIC score : Float

    """

    n_params = n_order * 4 - 2 # why - 2?

    return n_samples * np.log(mse) + n_params * np.log(n_samples)


def fit_to_bic_np(yearly_predictor, monthly_target, max_order):
    """choose order of Fourier Series to fit for by minimising BIC score

    Parameters
    ----------
    yearly_predictor : array-like of shape (n_samples,)
        Yearly temperature to predict with.
    monthly_target : array-like of shape (n_samples*12,)
        Target monthly temperature values.
    max_order : Integer
        Maximum order of Fourier Series.

    Returns
    -------
    n_sel : Integer
        Selected order of Fourier Series.
    coeffs_fit : array-like of size (4*n_Sel-2,)
        Fitted coefficients for the selected order of Fourier Series.
    preds : array-like of size (n_samples,)
        Predicted monthly values from final model.

    """

    bic_score = np.zeros([max_order])

    for i_n in range(1, max_order + 1):

        _, preds = fit_fourier_series_np(yearly_predictor, monthly_target, i_n)
        mse = np.mean((preds - monthly_target) ** 2)

        bic_score[i_n - 1] = calculate_bic(len(monthly_target), i_n, mse)

    n_sel = np.argmin(bic_score) + 1
    coeffs_fit, preds = fit_fourier_series_np(yearly_predictor, monthly_target, n_sel)

    coeffs = np.zeros([max_order * 4]) # removed -2, because it threw and error, why was that here?
    coeffs[: len(coeffs_fit)] = coeffs_fit # need the coeff array to be the same size for all orders

    return n_sel, coeffs, preds


def fit_to_bic_xr(yearly_predictor, monthly_target, max_order):
    """fit Fourier Series using BIC score to select order - xarray wrapper

    Parameters
    ----------
    yearly_predictor : xr.DataArray
        Yearly temperature values used as predictors, must contain dims: ("sample","cell").
        Containing one value per year.
    monthly_target : xr.DataArray
        Monthly temperature values to fit for, must contain dims: ("sample","cell").
    max_order : Integer
        Maximum order of Fourier Series to fit for.

    Returns
    -------
    data_vars : `xr.Dataset`
        Dataset containing the selected order of Fourier Series (n_sel), estimated
        coefficients of the Fourier Series (coeffs) and the resulting predictions for
        monthly temperatures (predictions).

    """

    if not isinstance(yearly_predictor, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_predictor)}")

    if not isinstance(monthly_target, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_target)}")
    
    yearly_predictor = mesmer.mesmer_m.upsample_yearly_data(yearly_predictor, monthly_target)

    n_sel, coeffs, preds = xr.apply_ufunc(
        fit_to_bic_np,
        yearly_predictor,
        monthly_target,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=([], ["coeff"], ["time"]),
        vectorize=True,
        output_dtypes=[int, float, float],
        kwargs={"max_order": max_order},
    )

    data_vars = {
        "n_sel": n_sel,
        "coeffs": coeffs,
        "predictions": preds,
    }

    return xr.Dataset(data_vars)
