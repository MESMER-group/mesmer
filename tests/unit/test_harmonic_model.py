import numpy as np
import pytest

from mesmer.mesmer_m.harmonic_model import (
    fit_to_bic_np,
    generate_fourier_series_np,
)


def test_generate_fourier_series_np():
    n_years = 10
    n_months = n_years * 12

    yearly_predictor = np.zeros(n_months)
    months = np.tile(np.arange(1, 13), n_years)

    # dummy yearly cycle
    expected = -np.sin(2 * np.pi * (months) / 12) - 2 * np.cos(
        2 * np.pi * (months) / 12
    )
    result = generate_fourier_series_np([0, -1, 0, -2], 1, yearly_predictor, months)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "coefficients", [np.array([0, -1, 0, -2]), np.array([0, -1, 0, -2, 0, 3, 0, -4])]
)
def test_fit_to_bic_np(coefficients):
    n_years = 10
    n_months = n_years * 12

    yearly_predictor = np.zeros(n_months)
    months = np.tile(np.arange(1, 13), n_years)

    monthly_target = generate_fourier_series_np(
        coefficients, int(len(coefficients) / 4), yearly_predictor, months
    )
    selected_order, estimated_coefficients, predictions = fit_to_bic_np(
        yearly_predictor, monthly_target, max_order=6
    )

    assert selected_order == int(len(coefficients) / 4)
    np.testing.assert_allclose(
        estimated_coefficients[0 : selected_order * 4], coefficients, atol=1e-2
    )
    np.testing.assert_allclose(predictions, monthly_target, atol=1e-2)
