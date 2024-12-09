# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M
"""

import numpy as np
import scipy as sp
import xarray as xr

import mesmer
from mesmer.core.utils import _check_dataarray_form


def _generate_fourier_series_np(yearly_predictor, coeffs):
    """construct a Fourier Series from the yearly predictor with given coeffs.

    The order of the Fourier series is inferred from the size of the coeffs array.

    Parameters
    ----------
    yearly_predictor : array-like of shape (n_years*12,)
        yearly predictor values.
    coeffs : array-like of shape (4*order)
        coefficients of Fourier Series.
    months : array-like of shape (n_years*12,)
        month values (1-12).

    Returns
    -------
    predictions: array-like of shape (n_years*12,)
        Fourier Series of order n calculated over yearly_predictor and months with coeffs.

    """

    # NOTE: performance-optimized fourier series generation

    coeffs = coeffs[~np.isnan(coeffs)]
    order = coeffs.size // 4

    # create 2D array of angles with shape (months, order)
    # as 2 * np.pi * k * np.arange(12).reshape(-1, 1) / 12 but faster

    factor = 2 * np.pi / 12
    k = np.arange(1.0, order + 1)
    alpha = np.arange(12 * factor, step=factor).reshape(-1, 1) * k

    # combine cosine and sine into one array
    cos_sin = np.empty((12, order * 2))
    cos_sin[:, :order] = np.cos(alpha)
    cos_sin[:, order:] = np.sin(alpha)

    # sum coefficients - equivalent to np.sum(cos_sin * coeffs[0::2], axis=1)
    coeff_a = cos_sin @ coeffs[0::2]
    coeff_b = cos_sin @ coeffs[1::2]

    # reshape yearly_predictor so the coeffs are correctly broadcast
    yearly_predictor = yearly_predictor.reshape(-1, 12)
    seasonal_cycle = coeff_a * yearly_predictor + coeff_b

    return seasonal_cycle.flatten()


def predict_harmonic_model(yearly_predictor, coeffs, time, time_dim="time"):
    """construct a Fourier Series from yearly predictors with fitted coeffs.

    Parameters
    ----------
    yearly_predictor : xr.DataArray of shape (n_years, n_gridcells)
        Predictor containing one value per year.
    coeffs : xr.DataArray of shape (n_gridcells, n_coeffs)
        coefficients of Fourier Series for each gridcell. Note that coeffs
        may contain nans (for higher orders, that have not been fit).
    time: xr.DataArray of shape (n_years * 12)
        A ``xr.DataArray`` containing cftime objects which will be used as coordinates
        for the monthly output values
    time_dim: str, default: "time"
        Name for the time dimension of the output ``xr.DataArray``.

    Returns
    -------
    predictions: xr.DataArray of shape (n_years * 12, n_gridcells)
        Fourier Series calculated over `yearly_predictor` with `coeffs`.

    """
    _, n_gridcells = yearly_predictor.shape
    _check_dataarray_form(
        yearly_predictor,
        "yearly_predictor",
        ndim=2,
        required_dims=time_dim,
        shape=(time.size // 12, n_gridcells),
    )
    upsampled_y = mesmer.core.utils.upsample_yearly_data(
        yearly_predictor, time, time_dim
    )

    predictions = upsampled_y + xr.apply_ufunc(
        _generate_fourier_series_np,
        upsampled_y,
        coeffs,
        input_core_dims=[[time_dim], ["coeff"]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        output_dtypes=[float],
    )

    return predictions.transpose(time_dim, ...)


def _fit_fourier_coeffs_np(yearly_predictor, monthly_target, first_guess):
    """fit the coefficients of a Fourier Series to the data using least squares for the
    given order, which is inferred from the size of the `first_guess` array.

    Parameters
    ----------

    yearly_predictor : array-like of shape (n_years*12,)
        Repeated yearly predictor.

    monthly_target : array-like of shape (n_years*12,)
        Target monthly values.

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
        Predicted monthly values.

    """

    def residuals_from_fourier_series(coeffs, yearly_predictor, mon_target):
        return _generate_fourier_series_np(yearly_predictor, coeffs) - mon_target

    # use least_squares to optimize the coefficients
    minimize_result = sp.optimize.least_squares(
        residuals_from_fourier_series,
        first_guess,
        args=(yearly_predictor, monthly_target),
        loss="linear",
        method="lm",
    )

    coeffs = minimize_result.x
    mse = np.mean(minimize_result.fun**2)

    return coeffs, mse


def _calculate_bic(n_samples, order, mse):
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

    # assume mse smaller 10 eps is 'perfect' - only relevant for noiseless test data
    if mse < np.finfo(float).eps * 10:
        return -np.inf

    return n_samples * np.log(mse) + n_params * np.log(n_samples)


def _fit_fourier_order_np(yearly_predictor, monthly_target, max_order):
    """determine order of Fourier Series for by minimizing BIC score.
    For each order, the coefficients are fit using least squares.

    Parameters
    ----------
    yearly_predictor : array-like of shape (n_years*12,)
        Repeated yearly values, i.e. containing the repeated yearly value for every month.
    monthly_target : array-like of shape (n_years*12,)
        Target monthly values.
    max_order : Integer
        Maximum considered order of Fourier Series for which to fit.

    Returns
    -------
    selected_order : Integer
        Selected order of Fourier Series.
    coeffs : array-like of size (4 * max_order,)
        Fitted coefficients for the selected order of Fourier Series.
    predictions : array-like of size (n_years*12,)
        Predicted monthly values from final model.

    """

    current_min_score = float("inf")
    last_coeffs = []
    selected_order = 0

    for i_order in range(1, max_order + 1):

        coeffs, mse = _fit_fourier_coeffs_np(
            yearly_predictor,
            monthly_target,
            # use coeffs from last iteration as first guess
            first_guess=np.append(last_coeffs, np.zeros(4)),
        )
        bic_score = _calculate_bic(len(monthly_target), i_order, mse)

        if bic_score < current_min_score:
            current_min_score = bic_score
            last_coeffs = coeffs
            selected_order = i_order
        else:
            break

    predictions = _generate_fourier_series_np(
        yearly_predictor=yearly_predictor, coeffs=last_coeffs
    )

    # need the coeff array to be the same size for all orders
    coeffs = np.full(max_order * 4, fill_value=np.nan)
    coeffs[: selected_order * 4] = last_coeffs

    return selected_order, coeffs, predictions


def fit_harmonic_model(yearly_predictor, monthly_target, max_order=6, time_dim="time"):
    """fit harmonic model i.e. a Fourier Series to every gridcell using BIC score to
    select the order and least squares to fit the coefficients for each order.

    Parameters
    ----------
    yearly_predictor : xr.DataArray of shape (n_years, n_gridcells)
        Yearly values used as predictors, containing one value per year.
    monthly_target : xr.DataArray of shape (n_months, n_gridcells)
        Monthly values to fit to, containing one value per month, for every year in
        ´yearly_predictor´ (starting with January!). So `n_months` = 12 :math:`\\cdot` `n_years`.
    max_order : Integer, default 6
        Maximum order of Fourier Series to fit for. Default is 6 since highest meaningful
        maximum order is sample_frequency/2, i.e. 12/2 to fit for monthly data.

    Returns
    -------
    data_vars : `xr.Dataset`
        Dataset containing the selected order of Fourier Series (`selected_order`),
        the estimated coefficients of the Fourier Series (`coeffs`) and the resulting
        residuals of the model (`residuals`).

    """

    if not isinstance(yearly_predictor, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_predictor)}")

    if not isinstance(monthly_target, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_target)}")

    if not monthly_target[time_dim].isel({time_dim: 0}).dt.month == 1:
        raise ValueError("Monthly target data must start with January.")

    yearly_predictor = mesmer.core.utils.upsample_yearly_data(
        yearly_predictor, monthly_target[time_dim], time_dim=time_dim
    )

    # subtract annual mean to have seasonal anomalies around 0
    seasonal_deviations = monthly_target - yearly_predictor

    selected_order, coeffs, preds = xr.apply_ufunc(
        _fit_fourier_order_np,
        yearly_predictor,
        seasonal_deviations,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=([], ["coeff"], [time_dim]),
        vectorize=True,
        output_dtypes=[int, float, float],
        kwargs={"max_order": max_order},
    )

    coeffs = coeffs.assign_coords({"coeff": np.arange(coeffs.sizes["coeff"])})

    resids = monthly_target - (yearly_predictor + preds)

    data_vars = {
        "selected_order": selected_order,
        "coeffs": coeffs,
        "residuals": resids.transpose(time_dim, ...),
    }

    return xr.Dataset(data_vars)
