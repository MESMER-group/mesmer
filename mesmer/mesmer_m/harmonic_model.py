# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M
"""

import numpy as np
import xarray as xr
import scipy as sp

import mesmer


def generate_fourier_series_np(yearly_T, coeffs, months):
    """construct the Fourier Series

    Parameters
    ----------
    yearly_T : array-like of shape (n_years*12,)
        yearly temperature values.
    coeffs : array-like of shape (4*order)
        coefficients of Fourier Series.
    months : array-like of shape (n_years*12,)
        month values (1-12).

    Returns
    -------
    predictions: array-like of shape (n_years*12,)
        Fourier Series of order n calculated over yearly_T and months with coeffs.

    """
    order = int(coeffs.size / 4)

    # fix these parameters, according to paper
    # we could also fit them and give an inital guess of 0 and 1 in the coeffs array as before
    beta0 = 0
    beta1 = 1

    seasonal_cycle = sum(
        [
            (coeffs[idx * 4] * yearly_T + coeffs[idx * 4 + 1])
            * np.sin(np.pi * i * (months) / 6)
            + (coeffs[idx * 4 + 2] * yearly_T + coeffs[idx * 4 + 3])
            * np.cos(np.pi * i * (months) / 6)
            for idx, i in enumerate(range(1, order + 1))
        ]
    )
    return beta0 + beta1 * yearly_T + seasonal_cycle


def fit_fourier_series_np(yearly_predictor, monthly_target, first_guess):
    """execute fitting of the harmonic model/fourier series

    Parameters
    ----------

    yearly_predictor : array-like of shape (n_years*12,)
        Repeated yearly temperature values to predict with.

    monthly_target : array-like of shape (n_years*12,)
        Target monthly temperature values.

    first_guess : array-like of shape (4*order)
        Initial guess for the coefficients of the Fourier Series.

    Method
    ------

    We use scipy.optimize.least_squares as a simple solver, given we have the equation:

    sum_{i=0}^{n} yearly_predictor + [(a{i}*x + b{i})*np.cos(\frac{np.pi*i*(mon)}{6}+(c{i}*x + d{i})*np.cos(\frac{np.pi*i*(mon)}{6})]

    We expect the yearly_predictor and monthly target to both be of size n_years*12, such that the yearly predictor contains each yearly value repeated 12 times to represent each month.


    Returns
    -------
    coeffs : array-like of shape (4*order)
        Fitted coefficients of Fourier series.

    preds : array-like of shape (n_years*12,)
        Predicted monthly temperature values.

    """

    # get monthly values
    mon_train = np.tile(np.arange(1, 13), int(yearly_predictor.shape[0] / 12))

    def func(coeffs, yearly_predictor, mon_train, mon_target):
        """loss function for fitting fourier series in scipy.optimize.least_squares"""

        loss = np.mean(
            (
                generate_fourier_series_np(yearly_predictor, coeffs, mon_train)
                - mon_target
            )
            ** 2
        )

        return loss

    # NOTE: this seems to select less 'orders' than the scipy one
    # np.linalg.lstsq(A, y)[0]

    minimize_result = sp.optimize.least_squares(
        func, # TODO: func should return residuals
        first_guess,
        args=(yearly_predictor, mon_train, monthly_target),
        loss="cauchy", # TODO: when returning residuals we should use 'linear'
    )

    coeffs = minimize_result.x
    mse = minimize_result.fun 
    # NOTE: when we switch to returning the residuals .fun no longer returns the mse

    preds = generate_fourier_series_np(
        yearly_T=yearly_predictor, coeffs=coeffs, months=mon_train
    )

    return coeffs, preds, mse


def calculate_bic(n_samples, order, mse):
    """calculate Bayesian Information Criterion (BIC)

    Parameters
    ----------
    n_samples : Integer
        size of training set.
    order : Integer
        Order of Fourier Series.
    mse : Float
        Mean-squared error.

    Returns
    -------
    BIC score : Float

    """

    n_params = order * 4
    # NOTE: removed -2 now because for order=0 the first two coefficients dissapear as sin(0) == 0
    # removed this now because we no longer fit beta0 and beta1

    return n_samples * np.log(mse) + n_params * np.log(n_samples)


def fit_to_bic_np(yearly_predictor, monthly_target, max_order):
    """choose order of Fourier Series to fit for by minimising BIC score

    Parameters
    ----------
    yearly_predictor : array-like of shape (n_years*12,)
        Repeated yearly temperature to predict with. Containing the repeated yearly value for every month.
    monthly_target : array-like of shape (n_years*12,)
        Target monthly temperature values.
    max_order : Integer
        Maximum considered order of Fourier Series for which to fit.

    Returns
    -------
    selected_order : Integer
        Selected order of Fourier Series.
    coeffs : array-like of size (4*n_sel,)
        Fitted coefficients for the selected order of Fourier Series.
    predictions : array-like of size (n_years*12,)
        Predicted monthly values from final model.

    """

    current_min_score = float("inf")
    last_coeffs = []
    selected_order = 0

    for i_order in range(1, max_order + 1):

        coeffs, predictions, mse = fit_fourier_series_np(
            yearly_predictor,
            monthly_target,
            i_order,
            # use coeffs from last iteration as first guess
            first_guess=np.append(last_coeffs, np.zeros(4)),
        )
        bic_score = calculate_bic(len(monthly_target), i_order, mse)

        if bic_score < current_min_score:
            current_min_score = bic_score
            last_coeffs = coeffs
            selected_order = i_order
        else:
            break

    # need the coeff array to be the same size for all orders
    coeffs = np.full(max_order * 4, fill_value=np.nan)
    coeffs[: selected_order * 4] = last_coeffs

    return selected_order, coeffs, predictions


def fit_to_bic_xr(yearly_predictor, monthly_target, max_order=6, time_dim="time"):
    """fit Fourier Series to every gridcell using BIC score to select order - xarray wrapper
    Repeats yearly values for every month before passing to :func:`fit_to_bic_np`

    Parameters
    ----------
    yearly_predictor : xr.DataArray of shape (n_years, n_gridcells)
        Yearly temperature values used as predictors
        Containing one value per year.
    monthly_target : xr.DataArray of shape (n_months, n_gridcells)
        Monthly temperature values to fit for, containing one value per month, for every year in yearly_predictor.
        So n_months = 12*n_years
    max_order : Integer, default 6
        Maximum order of Fourier Series to fit for. Default is 6 since highest meaningful
        maximum order is sample_frequency/2, i.e. 12/2 to fit for monthly data.

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

    yearly_predictor = mesmer.core.utils.upsample_yearly_data(
        yearly_predictor, monthly_target[time_dim], time_dim=time_dim
    )

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
