import numpy as np
import pandas as pd
import pytest
import xarray as xr
from packaging.version import Version

from mesmer.core.utils import upsample_yearly_data
from mesmer.mesmer_m.harmonic_model import (
    fit_to_bic_np,
    fit_to_bic_xr,
    generate_fourier_series_np,
    generate_fourier_series_xr
)
from mesmer.testing import trend_data_1D, trend_data_2D
from mesmer.core.utils import _check_dataarray_form


def test_generate_fourier_series_np():
    # test if fourier series is generated correctly
    n_years = 10
    n_months = n_years * 12

    yearly_predictor = np.zeros(n_months)
    months = np.tile(np.arange(1, 13), n_years)

    # dummy yearly cycle
    expected = -np.sin(2 * np.pi * (months) / 12) - 2 * np.cos(
        2 * np.pi * (months) / 12
    )
    result = generate_fourier_series_np(
        yearly_predictor, np.array([0, -1, 0, -2]), months
    )
    np.testing.assert_equal(result, expected)

    yearly_predictor = np.ones(n_months)
    result = generate_fourier_series_np(
        yearly_predictor, np.array([0, -1, 0, -2]), months
    )
    # NOTE: yearly_predictor is added to the Fourier series
    expected += 1
    np.testing.assert_equal(result, expected)

    result = generate_fourier_series_np(
        yearly_predictor, np.array([3.14, -1, 1, -2]), months
    )
    expected += 3.14 * np.sin(np.pi * months / 6) + 1 * np.cos(np.pi * months / 6)
    np.testing.assert_allclose(result, expected, atol=1e-10)

def test_generate_fourier_series_xr():
    n_years = 10
    n_lat, n_lon, n_gridcells = 2, 3, 2*3
    freq = "AS" if Version(pd.__version__) < Version("2.2") else "YS"
    time = xr.cftime_range(
        start="2000-01-01", periods=n_years, freq=freq
    )
    yearly_predictor = xr.DataArray(np.zeros((n_years, n_gridcells)), dims=["time", "cells"], coords={"time": time})

    time = xr.cftime_range(start="2000-01-01", periods=n_years * 12, freq="MS")
    monthly_time = xr.DataArray(time, dims=["time"], coords={"time": time})

    coeffs = get_2D_coefficients(order_per_cell=[1, 2, 3], n_lat=n_lat, n_lon=n_lon)

    result = generate_fourier_series_xr(yearly_predictor, coeffs, monthly_time, time_dim="time")

    _check_dataarray_form(result, "result", ndim = 2, required_dims=["time", "cells"], shape=(n_years * 12, n_gridcells))


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([0, -1, 0, -2]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    ],
)
def test_fit_to_bic_np(coefficients):
    # ensure original coeffs and series is recovered from noiseless fourier series
    max_order = 6
    n_years = 100
    months = np.tile(np.arange(1, 13), n_years)
    yearly_predictor = trend_data_1D(n_timesteps=n_years, intercept=0, slope=1).values
    yearly_predictor = np.repeat(yearly_predictor, 12)
    monthly_target = generate_fourier_series_np(yearly_predictor, coefficients, months)

    selected_order, estimated_coefficients, predictions = fit_to_bic_np(
        yearly_predictor, monthly_target, max_order=max_order
    )

    np.testing.assert_equal(selected_order, int(len(coefficients) / 4))

    # fill up all coefficient arrays with zeros to have the same length 4*max_order
    # to also be able to compare coefficients of higher orders than the original one
    original_coefficients = np.concatenate(
        [coefficients, np.zeros(4 * max_order - len(coefficients))]
    )
    estimated_coefficients = np.nan_to_num(
        estimated_coefficients,
        0,
    )

    np.testing.assert_allclose(original_coefficients, estimated_coefficients, atol=1e-7)
    np.testing.assert_allclose(predictions, monthly_target, atol=1e-7)


def get_2D_coefficients(order_per_cell, n_lat=3, n_lon=2):
    n_cells = n_lat * n_lon
    max_order = 6

    # generate coefficients that resemble real ones
    # generate rapidly decreasing coefficients for increasing orders
    trend = np.repeat(np.linspace(1.2, 0.2, max_order) ** 2, 4)
    # the first coefficients are rather small  (scaling of seasonal variability with temperature change)
    # while the second ones are large (constant distance of each month from the yearly mean)
    scale = np.tile([0.01, 5.0], (n_cells, max_order * 2))
    # generate some variability so not all coefficients are exactly the same
    rng = np.random.default_rng(0)
    variability = rng.normal(loc=0, scale=0.1, size=(n_cells, max_order * 4))
    # put it together
    coeffs = trend * scale + variability
    coeffs = np.round(coeffs, 1)

    # replace superfluous orders with nans
    for cell, order in enumerate(order_per_cell):
        coeffs[cell, order * 4 :] = np.nan

    LON, LAT = np.meshgrid(np.arange(n_lon), np.arange(n_lat))

    coords = {
        "lon": ("cells", LON.flatten()),
        "lat": ("cells", LAT.flatten()),
    }

    return xr.DataArray(coeffs, dims=("cells", "coeff"), coords=coords)


def test_fit_to_bic_xr():
    n_ts = 100
    orders = [1, 2, 3, 4, 5, 6]

    coefficients = get_2D_coefficients(order_per_cell=orders, n_lat=3, n_lon=2)

    yearly_predictor = trend_data_2D(n_timesteps=n_ts, n_lat=3, n_lon=2)

    freq = "AS" if Version(pd.__version__) < Version("2.2") else "YS"
    yearly_predictor["time"] = xr.cftime_range(
        start="2000-01-01", periods=n_ts, freq=freq
    )

    time = xr.cftime_range(start="2000-01-01", periods=n_ts * 12, freq="MS")
    monthly_time = xr.DataArray(time, dims=["time"], coords={"time": time})
    upsampled_yearly_predictor = upsample_yearly_data(yearly_predictor, monthly_time)

    months = upsampled_yearly_predictor.time.dt.month
    monthly_target = xr.apply_ufunc(
        generate_fourier_series_np,
        upsampled_yearly_predictor,
        coefficients,
        input_core_dims=[["time"], ["coeff"]],
        output_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[float],
        kwargs={"months": months},
    )

    # test if the model can recover the monthly target from perfect fourier series
    result = fit_to_bic_xr(yearly_predictor, monthly_target)
    np.testing.assert_equal(result.n_sel.values, orders)
    xr.testing.assert_allclose(result["predictions"], monthly_target)

    # test if the model can recover the underlying cycle with noise on top of monthly target
    rng = np.random.default_rng(0)
    noisy_monthly_target = monthly_target + rng.normal(
        loc=0, scale=0.1, size=monthly_target.values.shape
    )

    result = fit_to_bic_xr(yearly_predictor, noisy_monthly_target)
    xr.testing.assert_allclose(result["predictions"], monthly_target, atol=0.1)

    # compare numerically one cell of one year
    expected = np.array(
        [
            9.99630445,
            9.98829217,
            7.32212458,
            2.73123514,
            -2.53876124,
            -7.07931947,
            -9.69283667,
            -9.6945128,
            -7.08035255,
            -2.53178204,
            2.74790275,
            7.34046832,
        ]
    )
    np.testing.assert_allclose(
        result.predictions.isel(cells=0, time=slice(0, 12)).values, expected
    )


def test_fit_to_bix_xr_instance_checks():
    yearly_predictor = trend_data_2D(n_timesteps=10, n_lat=3, n_lon=2)
    monthly_target = trend_data_2D(n_timesteps=10 * 12, n_lat=3, n_lon=2)

    with pytest.raises(TypeError):
        fit_to_bic_xr(yearly_predictor.values, monthly_target)

    with pytest.raises(TypeError):
        fit_to_bic_xr(yearly_predictor, monthly_target.values)
