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
    result = generate_fourier_series_np(
        yearly_predictor, np.array([0, -1, 0, -2]), months
    )
    np.testing.assert_equal(result, expected)

    yearly_predictor = np.ones(n_months)
    result = generate_fourier_series_np(
        yearly_predictor, np.array([0, -1, 0, -2]), months
    )
    expected += 1
    np.testing.assert_equal(result, expected)

    result = generate_fourier_series_np(
        yearly_predictor, np.array([3.14, -1, 1, -2]), months
    )
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
    [np.repeat([-1, 1], 10 * 6), np.linspace(-1, 1, 10 * 12) * 10],
)
def test_fit_to_bic_np(coefficients, yearly_predictor):
    max_order = 6
    months = np.tile(np.arange(1, 13), 10)

    monthly_target = generate_fourier_series_np(yearly_predictor, coefficients, months)
    selected_order, estimated_coefficients, predictions = fit_to_bic_np(
        yearly_predictor, monthly_target, max_order=max_order
    )

    # assert selected_order == int(len(coefficients) / 4)
    # the model does not necessarily select the "correct" order
    # (i.e. the third coef array in combination with the second predictor)
    # but the coefficients of higher orders should be close to zero

    # fill up all coefficient arrays with zeros to have the same length 4*max_order
    # to also be able to compare coefficients of higher orders than the original one
    original_coefficients = np.concatenate(
        [coefficients, np.zeros(4 * max_order - len(coefficients))]
    )
    estimated_coefficients = np.nan_to_num(
        estimated_coefficients,
        0,
    )
    # NOTE: if we would use a constant predictor only the linear combination of coefficients needs to be close
    # np.testing.assert_allclose(
    #     [
    #         original_coefficients[i] * yearly_predictor[i]
    #         + original_coefficients[i + 1]
    #         for i in range(0, len(original_coefficients), 2)
    #     ],
    #     [
    #         estimated_coefficients[i] * yearly_predictor[i]
    #         + estimated_coefficients[i + 1]
    #         for i in range(0, len(estimated_coefficients), 2)
    #     ],
    #     atol=1e-2,
    # )

    np.testing.assert_allclose(original_coefficients, estimated_coefficients, atol=1e-2)

    np.testing.assert_allclose(predictions, monthly_target, atol=0.1)


def test_fit_to_bic_numerical_stability():
    coefficients = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    n_years = 3
    yearly_predictor = np.ones(12 * n_years)

    max_order = 6
    months = np.tile(np.arange(1, 13), n_years)

    monthly_target = generate_fourier_series_np(yearly_predictor, coefficients, months)
    selected_order, estimated_coefficients, predictions = fit_to_bic_np(
        yearly_predictor, monthly_target, max_order=max_order
    )

    assert selected_order == 2

    expected_coefficients = np.full(4 * max_order, np.nan)
    expected_coefficients[: selected_order * 4] = np.array(
        [
            1.49981711,
            1.49981711,
            3.49957326,
            3.49957326,
            5.4993294,
            5.4993294,
            7.49908555,
            7.49908555,
        ]
    )
    expected_predictions = np.array(
        [
            25.58545928,
            9.12336508,
            -10.99853688,
            -16.9260173,
            -5.58765396,
            8.99902459,
            10.46294769,
            -3.07130031,
            -16.99780532,
            -15.12238966,
            3.53558919,
            22.99731761,
            25.58545928,
            9.12336508,
            -10.99853688,
            -16.9260173,
            -5.58765396,
            8.99902459,
            10.46294769,
            -3.07130031,
            -16.99780532,
            -15.12238966,
            3.53558919,
            22.99731761,
            25.58545928,
            9.12336508,
            -10.99853688,
            -16.9260173,
            -5.58765396,
            8.99902459,
            10.46294769,
            -3.07130031,
            -16.99780532,
            -15.12238966,
            3.53558919,
            22.99731761,
        ]
    )

    np.testing.assert_allclose(expected_coefficients, estimated_coefficients)
    np.testing.assert_allclose(predictions, expected_predictions)


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

    months = upsampled_yearly_predictor.time.dt.month
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

    xr.testing.assert_allclose(result["predictions"], monthly_target, atol=0.1)


def test_fit_to_bix_xr_instance_checks():
    yearly_predictor = trend_data_2D(n_timesteps=10, n_lat=3, n_lon=2)
    monthly_target = trend_data_2D(n_timesteps=10 * 12, n_lat=3, n_lon=2)

    with pytest.raises(TypeError):
        fit_to_bic_xr(yearly_predictor.values, monthly_target)

    with pytest.raises(TypeError):
        fit_to_bic_xr(yearly_predictor, monthly_target.values)
