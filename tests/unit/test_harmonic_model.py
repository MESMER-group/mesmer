import numpy as np
import pytest

from mesmer.mesmer_m.harmonic_model import fit_to_bic_np, generate_fourier_series_np


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
    "coefficients", [np.array([0, -1, 0, -2]),
                     np.array([1,2,3,4,5,6,7,8]),
                     np.array([1,2,3,4,5,6,7,8,9,10,11,12]),],
)
@pytest.mark.parametrize(
    "yearly_predictor", [np.zeros(10*12), 
                         np.ones(10*12),
                         np.linspace(-1, 1, 10*12)*10],
)
def test_fit_to_bic_np(coefficients, yearly_predictor):

    # fill up all coefficient arrays with zeros to have the same length 4*6
    coefficients = np.concatenate([coefficients, np.zeros(4*6-len(coefficients))])

    months = np.tile(np.arange(1, 13), 10)

    monthly_target = generate_fourier_series_np(
        coefficients, int(len(coefficients) / 4), yearly_predictor, months
    )
    selected_order, estimated_coefficients, predictions = fit_to_bic_np(
        yearly_predictor, monthly_target, max_order=6
    )
    
    # assert selected_order == int(len(coefficients) / 4) 
    # the model does not necessarily select the "correct" order
    # but the coefficients of higher orders should be close to zero

    # linear combination of the coefficients with the predictor should be similar
    np.testing.assert_allclose([coefficients[i]*yearly_predictor[i]+coefficients[i+1] for i in range(0, len(coefficients), 2)],
                           [estimated_coefficients[i]*yearly_predictor[i]+estimated_coefficients[i+1] for i in range(0, len(estimated_coefficients), 2)], atol=1e-2)

    # actually all what really counts is that the predictions are close to the target
    np.testing.assert_allclose(predictions, monthly_target, atol=1e-1)
