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
)
from mesmer.testing import trend_data_2D


def test_generate_fourier_series_np():
    n_years = 10
    n_months = n_years * 12

    yearly_predictor = np.zeros(n_months)
    months = np.tile(np.arange(1, 13), n_years)

    # dummy yearly cycle
    expected = -np.sin(2 * np.pi * (months) / 12) - 2 * np.cos(
        2 * np.pi * (months) / 12
    )
    result = generate_fourier_series_np(yearly_predictor, np.array([0, -1, 0, -2]), months)
    np.testing.assert_equal(result, expected)

    yearly_predictor = np.ones(n_months)
    result = generate_fourier_series_np(yearly_predictor, np.array([0, -1, 0, -2]), months)
    expected += 1
    np.testing.assert_equal(result, expected)

    result = generate_fourier_series_np(yearly_predictor, np.array([3.14, -1, 1, -2]), months)
    expected += 3.14 * np.sin(np.pi * months / 6) + 1 * np.cos(np.pi * months / 6)
    np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([0, -1, 0, -2]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    ],
)
@pytest.mark.parametrize(
    "yearly_predictor",
    [np.zeros(10 * 12), np.ones(10 * 12), np.linspace(-1, 1, 10 * 12) * 10],
)
def test_fit_to_bic_np(coefficients, yearly_predictor):

    # fill up all coefficient arrays with zeros to have the same length 4*6
    coefficients = np.concatenate([coefficients, np.zeros(4 * 6 - len(coefficients))])

    months = np.tile(np.arange(1, 13), 10)

    monthly_target = generate_fourier_series_np(yearly_predictor, coefficients, months)
    selected_order, estimated_coefficients, predictions = fit_to_bic_np(
        yearly_predictor, monthly_target, max_order=6
    )

    # assert selected_order == int(len(coefficients) / 4)
    # the model does not necessarily select the "correct" order
    # but the coefficients of higher orders should be close to zero

    # linear combination of the coefficients with the predictor should be similar
    np.testing.assert_allclose(
        [
            coefficients[i] * yearly_predictor[i] + coefficients[i + 1]
            for i in range(0, len(coefficients), 2)
        ],
        [
            estimated_coefficients[i] * yearly_predictor[i]
            + estimated_coefficients[i + 1]
            for i in range(0, len(estimated_coefficients), 2)
        ],
        atol=1e-2,
    )

    # actually all what really counts is that the predictions are close to the target
    np.testing.assert_allclose(predictions, monthly_target, atol=0.1)


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([0, -1, 0, -2]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    ],
)
def test_fit_to_bic_xr(coefficients):
    yearly_predictor = trend_data_2D(n_timesteps=10, n_lat=3, n_lon=2)

    freq = "AS" if Version(pd.__version__) < Version("2.2") else "YS"
    yearly_predictor["time"] = xr.cftime_range(
        start="2000-01-01", periods=10, freq=freq
    )

    time = xr.cftime_range(start="2000-01-01", periods=10 * 12, freq="MS")
    monthly_time = xr.DataArray(
        time,
        dims=["time"],
        coords={"time": time},
    )
    upsampled_yearly_predictor = upsample_yearly_data(yearly_predictor, monthly_time)

    months = np.tile(np.arange(1, 13), 10)
    monthly_target = xr.apply_ufunc(
        generate_fourier_series_np,
        upsampled_yearly_predictor,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[float],
        kwargs={"coeffs": coefficients, "months": months},
    )

    result = fit_to_bic_xr(yearly_predictor, monthly_target)

    xr.testing.assert_allclose(result["predictions"], monthly_target, atol=1e-1)
