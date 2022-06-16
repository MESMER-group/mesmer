import numpy as np
import xarray as xr


def _draw_auto_regression_np(
    *, intercept, coefs, covariance, n_samples, n_ts, seed, buffer
):
    """draw time series of an autoregressive process

    Parameters
    ----------
    intercept : ndarray, float
        Intercept of the model. Must be scalar or have size n_cells.
    coefs : ndarray
        The coefficients of the autoregressive process. Must be a 2D array with shape
        ar_order x n_cells, i.e., the autoregressive coefficients are aligned along
        axis=0, while axis=1 contains all grid cells.
    covariance : float or ndarray
        The (co-)variance array. Needs to have shape n_cells x n_cells.
    n_samples : int
        Number of samples to draw.
    n_ts : int
        Number of time steps to draw.
    seed : int
        Seed used to initialize the pseudo-random number generator.
    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).

    Returns
    -------
    out : ndarray
        Drawn realizations of the specified autoregressive process. The array has shape
        n_samples x n_ts x n_cells.

    Notes
    -----
    As this is not an deterministic function it is not called `predict`. "Predicting"
    an autoregressive process does not include the innovations and therefore asymptotes
    towards certain value (in constrast to this function).
    """

    intercept = np.array(intercept)
    covariance = np.atleast_2d(covariance)

    # coeffs must be ar_order x n_cells
    ar_order, n_cells = coefs.shape

    # TODO: allow arbitrary lags? (e.g., [1, 12]) -> need to pass `ar_lags`
    ar_lags = np.arange(1, ar_order + 1, dtype=int)

    # ensure reproducibility
    np.random.seed(seed)

    innovations = np.random.multivariate_normal(
        mean=np.zeros(n_cells),
        cov=covariance,
        size=[n_samples, n_ts + buffer],
    )

    out = np.zeros([n_samples, n_ts + buffer, n_cells])
    for t in range(ar_order + 1, n_ts + buffer):

        ar = np.sum(coefs * out[:, t - ar_lags, :], axis=1)

        out[:, t, :] = intercept + ar + innovations[:, t, :]

    return out[:, buffer:, :]


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
