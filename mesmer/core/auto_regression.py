import numpy as np
import xarray as xr


def _select_ar_order_xr(data, dim, maxlag, ic="bic"):
    """Select the order of an autoregressive AR-X(p) process - xarray wrapper

    Parameters
    ----------
    data : array_like
        A 1-d endogenous response variable. The independent variable.
    dim : str
        Dimension along which to determine the order.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}, default 'bic'
        The information criterion to use in the selection.

    Returns
    -------
    selected_order : DataArray
        Array indicating the selected order with the same size as the input but ``dim``
        removed.

    Notes
    -----
    Only full models can be selected.
    """

    selected_order = xr.apply_ufunc(
        _select_order_np,
        data,
        input_core_dims=[[dim]],
        output_core_dims=((),),
        vectorize=True,
        output_dtypes=[int],
        kwargs={"maxlag": maxlag, "ic": ic},
    )

    # remove zeros
    selected_order[selected_order == 0] = np.NaN

    return selected_order


def _select_order_np(data, maxlag, ic="bic"):
    """Select the order of an autoregressive AR-X(p) process - numpy wrapper

    Parameters
    ----------
    data : array_like
        A 1-d endogenous response variable. The independent variable.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}, default 'bic'
        The information criterion to use in the selection.

    Returns
    -------
    selected_order : int
        The selected order.

    Notes
    -----
    Only full models can be selected.
    """

    from statsmodels.tsa.ar_model import ar_select_order

    ar_lags = ar_select_order(data, maxlag=maxlag, ic=ic, old_names=False).ar_lags

    return ar_lags[-1]


def _draw_auto_regression_correlated_np(
    *, intercept, coefs, covariance, n_samples, n_ts, seed, buffer
):
    """
    Draw time series of an auto regression process with possibly spatially-correlated
    innovations

    Creates `n_samples` auto-correlated time series of order `ar_order` and length
    `n_ts` for each set of `n_coefs` coefficients (typically one set for each grid
    point), the resulting array has shape n_samples x n_ts x n_coefs. The innovations
    can be spatially correlated.

    Parameters
    ----------
    intercept : float or ndarray of length n_coefs
        Intercept of the model.
    coefs : ndarray of shape ar_order x n_coefs
        The coefficients of the autoregressive process. Must be a 2D array with the
        autoregressive coefficients along axis=0, while axis=1 contains all idependent
        coefficients.
    covariance : float or ndarray of shape n_coefs x n_coefs
        The (co-)variance array. Must be symmetric and positive-semidefinite.
    n_samples : int
        Number of samples to draw for each set of coefficients.
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
        n_samples x n_ts x n_coefs.

    Notes
    -----
    The 'innovations' is the error or noise term.

    As this is not a deterministic function it is not called `predict`. "Predicting"
    an autoregressive process does not include the innovations and therefore asymptotes
    towards a certain value (in contrast to this function).
    """
    intercept = np.array(intercept)
    covariance = np.atleast_2d(covariance)

    # coeffs assumed to be ar_order x n_coefs
    ar_order, n_coefs = coefs.shape

    # TODO: allow arbitrary lags? (e.g., [1, 12]) -> need to pass `ar_lags` (see #164)
    ar_lags = np.arange(1, ar_order + 1, dtype=int)

    # ensure reproducibility (TODO: clarify approach to this, see #35)
    np.random.seed(seed)

    innovations = np.random.multivariate_normal(
        mean=np.zeros(n_coefs),
        cov=covariance,
        size=[n_samples, n_ts + buffer],
    )

    out = np.zeros([n_samples, n_ts + buffer, n_coefs])
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

    if np.ndim(lags) == 0:
        lags = np.arange(lags) + 1

    data_vars = {
        "intercept": intercept,
        "coeffs": coeffs,
        "standard_deviation": std,
        "lags": lags,
    }

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
