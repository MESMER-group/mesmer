import numpy as np
import xarray as xr


def _predict_auto_regression_np(
    intercept, coefs, covariance, n_emus, n_ts, n_cells, seed, buffer
):

    intercept = np.array(intercept)
    covariance = np.atleast_2d(covariance)

    # ensure reproducibility
    np.random.seed(seed)

    innovs = np.random.multivariate_normal(
        mean=np.zeros(n_cells),
        cov=covariance,
        size=[n_emus, n_ts + buffer],
    )

    emus_lv_tmp = np.zeros([n_emus, n_ts + buffer, n_cells])

    coefs = coefs
    AR_order = coefs.shape[0]
    ar_lags = np.arange(1, AR_order + 1, dtype=int)

    for t in range(AR_order + 1, n_ts + buffer):

        ar = np.sum(coefs * emus_lv_tmp[:, t - ar_lags, :], axis=1)

        emus_lv_tmp[:, t, :] = intercept + ar + innovs[:, t, :]

    return emus_lv_tmp[:, buffer:, :]


def _fit_auto_regression_xr(data, dim, lags):
    """
    fit an auto regression - xarray wrapper

    Parameters
    ----------
    data : xr.DataArray
        A ``xr.DataArray`` to estimate the auto regression over.
    dim : str
        Dimension along which to fit the auto regression over.
    lags : int
        The number of lags to include in the model.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the estimated parameters of the ``intercept``, the AR ``coeffs``
        and the ``standard_deviation`` of the residuals.
    """

    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(data)}")

    intercept, coeffs, std = xr.apply_ufunc(
        _fit_auto_regression_np,
        data,
        input_core_dims=[[dim]],
        output_core_dims=((), ("lags",), ()),
        vectorize=True,
        output_dtypes=[float, float, float],
        kwargs={"lags": lags},
    )

    data_vars = {"intercept": intercept, "coeffs": coeffs, "standard_deviation": std}

    # TODO: add coords for lags?
    return xr.Dataset(data_vars)


def _fit_auto_regression_np(data, lags):
    """
    fit an auto regression - numpy wrapper

    Parameters
    ----------
    data : np.array
        A numpy array to estimate the auto regression over. Must be 1D.
    lags : int
        The number of lags to include in the model.

    Returns
    -------
    intercept : :obj:`np.array`
        Intercept of the fitted AR model.
    coeffs : :obj:`np.array`
        Coefficients if the AR model. Will have as many entries as ``lags``.
    std :obj:`np.array`
        Standard deviation of the residuals.
    """

    from statsmodels.tsa.ar_model import AutoReg

    AR_model = AutoReg(data, lags=lags, old_names=False)
    AR_result = AR_model.fit()

    intercept = AR_result.params[0]
    coeffs = AR_result.params[1:]

    # standard deviation of the residuals
    std = np.sqrt(AR_result.sigma2)

    return intercept, coeffs, std
