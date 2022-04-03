from unittest import mock

import numpy as np
import pytest
import xarray as xr

import mesmer.core.auto_regression
from mesmer.core.utils import _check_dataarray_form, _check_dataset_form

from .utils import trend_data_1D, trend_data_2D


@pytest.mark.parametrize("obj", [xr.Dataset(), None])
def test_fit_auto_regression_xr_errors(obj):

    with pytest.raises(TypeError, match="Expected a `xr.DataArray`"):
        mesmer.core.auto_regression._fit_auto_regression_xr(obj, "dim", lags=1)


def test_fit_auto_regression_xr_1D_values():
    # values obtained by running the example - to ensure there are no changes in
    # statsmodels.tsa.ar_model.AutoReg

    data = trend_data_1D()
    result = mesmer.core.auto_regression._fit_auto_regression_xr(data, "time", lags=1)

    expected = xr.Dataset(
        {
            "intercept": 1.04728995,
            "coeffs": ("lags", [0.99682459]),
            "standard_deviation": 1.02655342,
        }
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("lags", [1, 2])
def test_fit_auto_regression_xr_1D(lags):

    data = trend_data_1D()
    res = mesmer.core.auto_regression._fit_auto_regression_xr(data, "time", lags=lags)

    _check_dataset_form(
        res,
        "_fit_auto_regression_result",
        required_vars=["intercept", "coeffs", "standard_deviation"],
    )

    _check_dataarray_form(res.intercept, "intercept", ndim=0, shape=())
    _check_dataarray_form(
        res.coeffs, "coeffs", ndim=1, required_dims={"lags"}, shape=(lags,)
    )
    _check_dataarray_form(
        res.standard_deviation, "standard_deviation", ndim=0, shape=()
    )


@pytest.mark.parametrize("lags", [1, 2])
def test_fit_auto_regression_xr_2D(lags):

    data = trend_data_2D()
    res = mesmer.core.auto_regression._fit_auto_regression_xr(data, "time", lags=lags)

    (n_cells,) = data.cells.shape

    _check_dataset_form(
        res,
        "_fit_auto_regression_result",
        required_vars=["intercept", "coeffs", "standard_deviation"],
    )

    _check_dataarray_form(res.intercept, "intercept", ndim=1, shape=(n_cells,))
    _check_dataarray_form(
        res.coeffs,
        "coeffs",
        ndim=2,
        required_dims={"cells", "lags"},
        shape=(n_cells, lags),
    )
    _check_dataarray_form(
        res.standard_deviation, "standard_deviation", ndim=1, shape=(n_cells,)
    )


@pytest.mark.parametrize("lags", [1, 2])
def test_fit_auto_regression_np(lags):

    data = np.array([0, 1, 3.14])

    mock_auto_regressor = mock.Mock()
    mock_auto_regressor.params = np.array([0.1, 0.25])
    mock_auto_regressor.sigma2 = 3.14

    with mock.patch(
        "statsmodels.tsa.ar_model.AutoReg"
    ) as mocked_auto_regression, mock.patch(
        "statsmodels.tsa.ar_model.AutoRegResults"
    ) as mocked_auto_regression_result:

        mocked_auto_regression.return_value = mocked_auto_regression_result
        mocked_auto_regression_result.return_value = mock_auto_regressor

        mesmer.core.auto_regression._fit_auto_regression_np(data, lags=lags)

        mocked_auto_regression.assert_called_once()
        mocked_auto_regression.assert_called_with(data, lags=lags, old_names=False)

        mocked_auto_regression_result.fit.assert_called_once()
        mocked_auto_regression_result.fit.assert_called_with()
