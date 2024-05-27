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
    # NOTE: yearly_predictor is added to the Fourier series
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
            1.5,
            1.5,
            3.5,
            3.5,
            5.499359,
            5.499359,
            7.499126,
            7.499126,
        ]
    )
    expected_predictions = np.array(
        [
            25.58647411,
            9.12411969,
            -10.99825277,
            -16.92622002,
            -5.58822129,
            8.99825274,
            10.46211833,
            -3.07203284,
            -16.99825283,
            -15.12237243,
            3.53613445,
            22.99825286,
            25.58647411,
            9.12411969,
            -10.99825277,
            -16.92622002,
            -5.58822129,
            8.99825274,
            10.46211833,
            -3.07203284,
            -16.99825283,
            -15.12237243,
            3.53613445,
            22.99825286,
            25.58647411,
            9.12411969,
            -10.99825277,
            -16.92622002,
            -5.58822129,
            8.99825274,
            10.46211833,
            -3.07203284,
            -16.99825283,
            -15.12237243,
            3.53613445,
            22.99825286,
        ]
    )

    np.testing.assert_allclose(expected_coefficients, estimated_coefficients)
    np.testing.assert_allclose(predictions, expected_predictions)


def get_2D_coefficients(order_per_cell, n_lat=3, n_lon=2):
    n_cells = n_lat * n_lon
    max_order = 6

    # generate coefficients that reseble real ones
    # generate rapidly decreasing coefficients for increasing orders
    trend = np.repeat(np.linspace(1.2, 0.2, max_order)**2, 4)
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
    n_ts = 10
    orders = [1, 2, 3, 4, 5, 6]

    coefficients = get_2D_coefficients(order_per_cell=orders, n_lat=3, n_lon=2)

    yearly_predictor = trend_data_2D(n_timesteps=n_ts, n_lat=3, n_lon=2)

    freq = "AS" if Version(pd.__version__) < Version("2.2") else "YS"
    yearly_predictor["time"] = xr.cftime_range(
        start="2000-01-01", periods=n_ts, freq=freq
    )

    time = xr.cftime_range(start="2000-01-01", periods=n_ts * 12, freq="MS")
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
        coefficients,
        input_core_dims=[["time"], ["coeff"]],
        output_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[float],
        kwargs={"months": months},
    )

    # test if the model can recover the monthly target from perfect fourier series
    result = fit_to_bic_xr(yearly_predictor, monthly_target)
    assert (result.n_sel.values == orders).all()
    xr.testing.assert_allclose(result["predictions"], monthly_target, atol=0.1)

    # test if the model can recover the underlying cycle with noise on top of monthly target
    rng = np.random.default_rng(0)
    noisy_monthly_target = monthly_target + rng.normal(
        loc=0, scale=0.1, size=monthly_target.values.shape
    )
    result = fit_to_bic_xr(yearly_predictor, noisy_monthly_target)
    xr.testing.assert_allclose(result["predictions"], monthly_target, atol=0.2)


def test_fit_to_bix_xr_instance_checks():
    yearly_predictor = trend_data_2D(n_timesteps=10, n_lat=3, n_lon=2)
    monthly_target = trend_data_2D(n_timesteps=10 * 12, n_lat=3, n_lon=2)

    with pytest.raises(TypeError):
        fit_to_bic_xr(yearly_predictor.values, monthly_target)

    with pytest.raises(TypeError):
        fit_to_bic_xr(yearly_predictor, monthly_target.values)
