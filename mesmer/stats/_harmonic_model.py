# MESMER-M, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

"""
Functions to train monthly trend module of MESMER-M
"""

from functools import lru_cache

import numpy as np
import scipy as sp
import xarray as xr

import mesmer
from mesmer.core.utils import _check_dataarray_form


@lru_cache
def _get_cos_sin(order):
    # cos_sin is constant with order, cache it for a considerable speed gain

    # create 2D array of angles with shape (months, order)
    # as 2 * np.pi * k * np.arange(12).reshape(-1, 1) / 12 but faster

    factor = 2 * np.pi / 12
    k = np.arange(1.0, order + 1)
    alpha = np.arange(0, 12 * factor, step=factor).reshape(-1, 1) * k

    # combine cosine and sine into one array
    cos_sin = np.empty((12, order * 2))
    cos_sin[:, :order] = np.cos(alpha)
    cos_sin[:, order:] = np.sin(alpha)

    return cos_sin


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

    return _generate_fourier_series_order_np(yearly_predictor, coeffs, order)


def _generate_fourier_series_order_np(yearly_predictor, coeffs, order):

    cos_sin = _get_cos_sin(order)

    # sum coefficients - equivalent to np.sum(cos_sin * coeffs[0::2], axis=1)
    coeff_a = cos_sin @ coeffs[0::2]
    coeff_b = cos_sin @ coeffs[1::2]

    # reshape yearly_predictor so the coeffs are correctly broadcast
    yearly_predictor = yearly_predictor.reshape(-1, 12)
    seasonal_cycle = coeff_a * yearly_predictor + coeff_b

    return seasonal_cycle.ravel()


def predict_harmonic_model(
    yearly_predictor: xr.DataArray,
    coeffs: xr.DataArray,
    time: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """construct a Fourier Series from yearly predictors with fitted coeffs.

    Parameters
    ----------
    yearly_predictor : xr.DataArray
        yearly values used as predictors, must contain `time_dim` but can have
        additional dimensions for example gridcells or members.
    coeffs : xr.DataArray
        coefficients of Fourier Series, must have "coeff" dim and additional dims of
        `yearly_predictor`. Note that coeffs may contain nans (for higher orders, that have not been fit).
    time: xr.DataArray
        A ``xr.DataArray`` containing cftime objects which will be used as coordinates
        for the monthly output values
    time_dim: str, default: "time"
        Name of the time dimension on `yearly_predictor`. Will also be the name of the time_dim
        of the output ``xr.DataArray``.

    Returns
    -------
    predictions: xr.DataArray
        Fourier Series calculated over `yearly_predictor` with `coeffs`, has `time_dim` with values of `time` and
        any additional dimensions of `yearly_predictor`.

    """

    _check_dataarray_form(
        yearly_predictor,
        "yearly_predictor",
        required_coords=time_dim,
    )
    (sample_dim,) = yearly_predictor[time_dim].dims
    dims = set(yearly_predictor.dims) - {sample_dim}
    _check_dataarray_form(
        coeffs,
        "coeffs",
        required_dims=dims | {"coeff"},
    )

    upsampled_y = mesmer.core.utils.upsample_yearly_data(
        yearly_predictor, time, time_dim
    )

    predictions = upsampled_y + xr.apply_ufunc(
        _generate_fourier_series_np,
        upsampled_y,
        coeffs,
        input_core_dims=[[sample_dim], ["coeff"]],
        output_core_dims=[[sample_dim]],
        vectorize=True,
        output_dtypes=[float],
    )

    return predictions.transpose(sample_dim, ...)


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

    def _residuals_from_fourier_series(coeffs, yearly_predictor, mon_target, order):
        return (
            _generate_fourier_series_order_np(yearly_predictor, coeffs, order)
            - mon_target
        )

    order = first_guess.size // 4

    # use least_squares to optimize the coefficients
    minimize_result = sp.optimize.least_squares(
        _residuals_from_fourier_series,
        first_guess,
        args=(yearly_predictor, monthly_target, order),
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


def fit_harmonic_model(
    yearly_predictor: xr.DataArray,
    monthly_target: xr.DataArray,
    *,
    max_order: int = 6,
    time_dim: str = "time",
) -> xr.Dataset:
    """fit harmonic model i.e. a Fourier Series to every gridcell using BIC score to
    select the order and least squares to fit the coefficients for each order.

    Parameters
    ----------
    yearly_predictor : xr.DataArray
        Yearly values used as predictors, containing one value per year. Contains `time_dim`
        and possibly additional dimensions for example for gridcells or members.
    monthly_target : xr.DataArray
        Monthly values to fit to, containing one value per month, for every year in
        `yearly_predictor` (starting with January). So `n_months` = 12 :math:`\\cdot` `n_years`.
        Must contain `time_dim` and possibly additional dimensions as `yearly_predictor`.
    max_order : Integer, default 6
        Maximum order of Fourier Series to fit for. Default is 6 since highest meaningful
        maximum order is sample_frequency/2, i.e. 12/2 to fit for monthly data.
    time_dim: str, default: "time"
        Name of the time dimension on `yearly_predictor` and `monthly_target`.

    Returns
    -------
    data_vars : `xr.Dataset`
        Dataset containing

        - the selected order of Fourier Series (`selected_order`),
        - the estimated coefficients of the Fourier Series (`coeffs`), and
        - the residuals of the model (`residuals`).

    """

    _check_dataarray_form(
        yearly_predictor, "yearly_predictor", required_coords={time_dim}
    )

    _check_dataarray_form(monthly_target, "monthly_target", required_coords={time_dim})

    # we need to pass the dim (which may be `time_dim` or `sample_dim`)
    (sample_dim,) = monthly_target[time_dim].dims

    if set(yearly_predictor.dims) != set(monthly_target.dims):

        msg = (
            "DataArray objects have different dimensions:\n"
            f"- `{yearly_predictor.dims}` in `yearly_predictor`\n"
            f"- `{monthly_target.dims}` in `monthly_target`"
        )
        raise ValueError(msg)

    for dim in set(yearly_predictor.dims) - {sample_dim}:

        if yearly_predictor[dim].size != monthly_target[dim].size:
            msg = (
                f"The '{dim}' coords of `yearly_predictor` and `monthly_target` have a "
                f"different size: {yearly_predictor[dim].size} vs. "
                f"{monthly_target[dim].size}"
            )
            raise ValueError(msg)

    if not monthly_target[time_dim].isel({sample_dim: 0}).dt.month == 1:
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
        input_core_dims=[[sample_dim], [sample_dim]],
        output_core_dims=([], ["coeff"], [sample_dim]),
        vectorize=True,
        output_dtypes=[int, float, float],
        kwargs={"max_order": max_order},
    )

    coeffs = coeffs.assign_coords({"coeff": np.arange(coeffs.sizes["coeff"])})

    resids = monthly_target - (yearly_predictor + preds)

    data_vars = {
        "selected_order": selected_order,
        "coeffs": coeffs,
        "residuals": resids.transpose(sample_dim, ...),
    }

    return xr.Dataset(data_vars)
