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


def generate_fourier_series_np(coeffs, n, x, mon):

    """
    Construct the Fourier Series

    Parameters
    ----------

    coeffs: array-like of shape (4*n-2)
            coefficients of Fourier Series.

    n : Integer
        Order of the Fourier Series.

    x : array-like of shape (n_samples,)
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
            (coeffs[idx] * x + coeffs[idx + 1]) * np.sin(np.pi * i * (mon % 12 + 1) / 6)
            + (coeffs[idx + 2] * x + coeffs[idx + 3])
            * np.cos(np.pi * i * (mon % 12 + 1) / 6)
            for i, idx in enumerate(np.arange(n * 4, step=4))
        ]
    )


def fit_fourier_series_np(x, y, n, repeat=False):
    """execute fitting of the harmonic model/fourier series

    Parameters
    ----------

    x : array-like of shape (n_samples,)
        Yearly temperature values to predict with.

    y : array-like of shape (n_samples*12,)
        Target monthly temperature values.

    n : Integer
        Order of the Fourier Series.

    repeat : bool, default: True
        Whether x data should be expanded.

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

    if repeat:
        # each month scales to the yearly value at that timestep so need to repeat
        x_train = np.repeat(x, 12)
        # yearly temp. array for fitting if not already done
    else:
        x_train = x

    mon_train = np.tile(
        np.arange(1, 13), int(x_train.shape[0] / 12)
    )  # also get monthly values
    mon_train = (
        np.pi * (mon_train % 12 + 1)
    ) / 6  # for simplicity's sake we take month values in there harmonic form

    # construct predictor matrix

    # A = np.hstack(
    #    (
    #        [
    #            np.array(
    #                [
    #                    x_train * np.sin(i_n * mon_train),
    #                    np.sin(i_n * mon_train),
    #                    x_train * np.cos(i_n * mon_train),
    #                    np.sin(i_n * mon_train),
    #                ]
    #            ).T
    #            for i_n in range(n)
    #        ]
    #    )
    # )

    def fun(x, n, x_train, mon_train, y):

        """
        Loss function for fitting fourier series needed as input to scipy.optimize.least_squares
        """
        loss = np.mean((generate_fourier_series_np(x, n, x_train, mon_train) - y) ** 2)

        return loss

    # print(A.shape,)
    x0 = np.zeros(n * 4)
    x0[2] = 1
    x0[3] = 0

    coeffs = optimize.least_squares(
        fun, x0, args=(n, x_train, mon_train, y), loss="cauchy"
    ).x  # np.linalg.lstsq(A, y)[0]

    # print(coeffs.shape)
    y_pred = generate_fourier_series_np(coeffs, n, x_train, mon_train)

    return coeffs, y_pred


def calculate_bic(n_samples, n_order, mse):

    """
    Calculate Bayesian Information Criteria (BIC)

    Input
    -----

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

    n_params = n_order * 4 - 2

    return n_samples * np.log(mse) + n_params * np.log(n_samples)


def fit_to_bic_np(x, y, max_n, repeat=False):

    """
    Choose order of Fourier Series to fit for by minimising BIC score

    Input
    -----

    x : array-like of shape (n_samples/12,)
        Yearly temperature values to predict with.

    y : array-like of shape (n_samples,)
        Target monthly temperature values.

    n : Integer
        Maximum order of Fourier Series.

    repeat: Boolean
            Passed on to fit_fourier_series_np , default=False

    Returns
    -------

    n_sel : Integer
           Selected order of Fourier Series.

    coeffs_fit : array-like of size (4*n_Sel-2,)
                Fitted coefficients for the selected order of Fourier Series.

    y_pred : array-like of size (n_samples,)
            Predicted y values from final model.

    """

    bic_score = np.zeros([max_n])

    for i_n in range(1, max_n + 1):

        _, y_pred = fit_fourier_series_np(x, y, i_n, repeat=repeat)
        mse = np.mean((y_pred - y) ** 2)

        bic_score[i_n - 1] = calculate_bic(len(y), i_n, mse)

    n_sel = np.argmin(bic_score) + 1
    coeffs_fit, y_pred = fit_fourier_series_np(x, y, n_sel, repeat=repeat)

    coeffs = np.zeros([max_n * 4 - 2])
    coeffs[: len(coeffs_fit)] = coeffs_fit

    return n_sel, coeffs, y_pred


def fit_to_bic_xr(X, Y, max_n):

    """
    Fit Fourier Series using BIC score to select order - xarray wrapper

    Parameters
    ----------

    X : xr.DataArray
        Yearly temperature values used as predictors, must contain dims: ("sample","cell").

    Y : xr.DataArray
        Monthly temperature values to fit for, must contain dims: ("sample","cell").

    max_n : Integer
            Maximum order of Fourier Series to fit for.


    Returns
    -------
    data_vars : `xr.Dataset`
           Dataset containing the selected order of Fourier Series (n_sel), estimated coefficients of the Fourier Series (coeffs)
           and the resulting predictions for monthly temperatures (predictions).

    """

    if not isinstance(X, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(X)}")

    if not isinstance(Y, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(Y)}")

    n_sel, coeffs, preds = xr.apply_ufunc(
        fit_to_bic_np,
        X,
        Y,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=([], ["coeff"], ["time"]),
        vectorize=True,
        output_dtypes=[int, float, float],
        kwargs={"max_n": max_n, "repeat": False},
    )

    data_vars = {
        "n_sel": n_sel,
        "coeffs": coeffs,
        "predictions": preds,
    }

    return xr.Dataset(data_vars)
