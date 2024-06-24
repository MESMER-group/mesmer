import warnings

import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import xarray as xr

from mesmer.core.utils import LinAlgWarning, _check_dataarray_form, _check_dataset_form


def _select_ar_order_scen_ens(*objs, dim, ens_dim, maxlag, ic="bic"):
    """
    Select the order of an autoregressive process and potentially calculate the median
    over ensemble members and scenarios

    Parameters
    ----------
    *objs : iterable of DataArray
        A list of ``xr.DataArray`` to estimate the auto regression order over.
    dim : str
        Dimension along which to determine the order.
    ens_dim : str
        Dimension name of the ensemble members.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}, default 'bic'
        The information criterion to use in the selection.

    Returns
    -------
    selected_ar_order : DataArray
        Array indicating the selected order with the same size as the input but ``dim``
        removed.

    Notes
    -----
    Calculates the median auto regression order, first over the ensemble members,
    then over all scenarios.
    """

    ar_order_scen = list()
    for obj in objs:
        res = select_ar_order(obj, dim=dim, maxlag=maxlag, ic=ic)

        if ens_dim in res.dims:
            res = res.quantile(dim=ens_dim, q=0.5, method="nearest")

        ar_order_scen.append(res)

    ar_order_scen = xr.concat(ar_order_scen, dim="scen")

    ar_order = ar_order_scen.quantile(0.5, dim="scen", method="nearest")

    if not np.isnan(ar_order).any():
        ar_order = ar_order.astype(int)

    return ar_order


def _fit_auto_regression_scen_ens(*objs, dim, ens_dim, lags):
    """
    fit an auto regression and potentially calculate the mean over ensemble members
    and scenarios

    Parameters
    ----------
    *objs : iterable of DataArray
        A list of ``xr.DataArray`` to estimate the auto regression over.
    dim : str
        Dimension along which to fit the auto regression.
    ens_dim : str
        Dimension name of the ensemble members.
    lags : int
        The number of lags to include in the model.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the estimated parameters of the ``intercept``, the AR
        ``coeffs`` and the ``variance`` of the residuals.

    Notes
    -----
    Calculates the mean auto regression, first over the ensemble members, then over all
    scenarios.
    """

    ar_params_scen = list()
    for obj in objs:
        ar_params = fit_auto_regression(obj, dim=dim, lags=int(lags))

        # BUG/ TODO: fix for v1, see https://github.com/MESMER-group/mesmer/issues/307
        ar_params["standard_deviation"] = np.sqrt(ar_params.variance)

        if ens_dim in ar_params.dims:
            ar_params = ar_params.mean(ens_dim)

        ar_params_scen.append(ar_params)

    ar_params_scen = xr.concat(ar_params_scen, dim="scen")

    # return the mean over all scenarios
    ar_params = ar_params_scen.mean("scen")

    return ar_params


# ======================================================================================


def select_ar_order(data, dim, maxlag, ic="bic"):
    """Select the order of an autoregressive process

    Parameters
    ----------
    data : DataArray
        A ``xr.DataArray`` to estimate the auto regression order.
    dim : str
        Dimension along which to determine the order.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}, default 'bic'
        The information criterion to use in the selection.

    Returns
    -------
    selected_ar_order : DataArray
        Array indicating the selected order with the same size as the input but ``dim``
        removed.

    Notes
    -----
    Thin wrapper around ``statsmodels.tsa.ar_model.ar_select_order``. Only full models
    can be selected.
    """

    selected_ar_order = xr.apply_ufunc(
        _select_ar_order_np,
        data,
        input_core_dims=[[dim]],
        output_core_dims=((),),
        vectorize=True,
        output_dtypes=[float],
        kwargs={"maxlag": maxlag, "ic": ic},
    )

    # remove zeros
    selected_ar_order.data[selected_ar_order.data == 0] = np.nan

    selected_ar_order.name = "selected_ar_order"

    return selected_ar_order


def _select_ar_order_np(data, maxlag, ic="bic"):
    """Select the order of an autoregressive AR(p) process - numpy wrapper

    Parameters
    ----------
    data : array_like
        A numpy array to estimate the auto regression order. Must be 1D.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}, default 'bic'
        The information criterion to use in the selection.

    Returns
    -------
    selected_ar_order : int
        The selected order.

    Notes
    -----
    Thin wrapper around ``statsmodels.tsa.ar_model.ar_select_order``. Only full models
    can be selected.
    """

    from statsmodels.tsa.ar_model import ar_select_order

    ar_lags = ar_select_order(data, maxlag=maxlag, ic=ic).ar_lags

    # None is returned if no lag is selected
    selected_ar_order = np.nan if ar_lags is None else ar_lags[-1]

    return selected_ar_order


def _get_size_and_coord_dict(coords_or_size, dim, name):

    if isinstance(coords_or_size, int):
        size, coord_dict = coords_or_size, {}

        return size, coord_dict

    # TODO: use public xr.Index when the minimum xarray version is v2023.08.0
    xr_Index = xr.core.indexes.Index

    if not isinstance(coords_or_size, (xr.DataArray, xr_Index, pd.Index)):
        raise TypeError(
            f"expected '{name}' to be an `int`, pandas or xarray Index or a `DataArray`"
            f" got {type(coords_or_size)}"
        )

    if coords_or_size.ndim != 1:
        raise ValueError(f"Coords must be 1D but have {coords_or_size.ndim} dimensions")

    size = coords_or_size.size
    coord_dict = {dim: np.asarray(coords_or_size)}

    return size, coord_dict


def draw_auto_regression_uncorrelated(
    ar_params,
    *,
    time,
    realisation,
    seed,
    buffer,
    time_dim="time",
    realisation_dim="realisation",
):
    """draw time series of an auto regression process

    Parameters
    ----------
    ar_params : Dataset
        Dataset containing the estimated parameters of the AR process. Must contain the
        following DataArray objects:

        - intercept
        - coeffs
        - variance

    time : int | DataArray | Index
        Defines the number of auto-correlated samples to draw and possibly its coordinates.

        - ``int``: defines the number of time steps to draw
        - ``DataArray`` or ``Index``: defines the coordinates and its length the number
          of samples along the time dimension to draw.

    realisation : int | DataArray
        Defines the number of uncorrelated samples to draw and possibly its coordinates.
        See ``time`` for details.

    seed : int
        Seed used to initialize the pseudo-random number generator.

    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).

    Returns
    -------
    out : DataArray
        Drawn realizations of the specified autoregressive process. The array has shape
        n_time x n_coeffs x n_realisations.

    """

    # check the input
    _check_dataset_form(
        ar_params, "ar_params", required_vars=("intercept", "coeffs", "variance")
    )

    if (
        ar_params.intercept.ndim != 0
        or ar_params.coeffs.ndim != 1
        or ar_params.variance.ndim != 0
    ):
        raise ValueError(
            "``_draw_auto_regression_uncorrelated`` can currently only handle single points"
        )

    # _draw_ar_corr_xr_internal expects 2D arrays
    ar_params = ar_params.expand_dims("__gridpoint__")

    result = _draw_ar_corr_xr_internal(
        intercept=ar_params.intercept,
        coeffs=ar_params.coeffs,
        covariance=ar_params.variance,
        time=time,
        realisation=realisation,
        seed=seed,
        buffer=buffer,
        time_dim=time_dim,
        realisation_dim=realisation_dim,
    )

    # remove the "__gridpoint__" dim again
    result = result.squeeze(dim="__gridpoint__", drop=True)

    return result


def draw_auto_regression_correlated(
    ar_params,
    covariance,
    *,
    time,
    realisation,
    seed,
    buffer,
    time_dim="time",
    realisation_dim="realisation",
):
    """
    draw time series of an auto regression process with spatially-correlated innovations

    Parameters
    ----------
    ar_params : Dataset
        Dataset containing the estimated parameters of the AR process. Must contain the
        following DataArray objects:

        - intercept
        - coeffs

    covariance : DataArray
        The (co-)variance array. Must be symmetric and positive-semidefinite.

    time : int | DataArray | Index
        Defines the number of auto-correlated samples to draw and possibly its coordinates.

        - ``int``: defines the number of time steps to draw
        - ``DataArray`` or ``Index``: defines the coordinates and its length the number
          of samples along the time dimension to draw.

    realisation : int | DataArray
        Defines the number of uncorrelated samples to draw and possibly its coordinates.
        See ``time`` for details.

    seed : int
        Seed used to initialize the pseudo-random number generator.

    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).

    Returns
    -------
    out : DataArray
        Drawn realizations of the specified autoregressive process. The array has shape
        n_time x n_coeffs x n_realisations.

    Notes
    -----
    The number of (spatially-)correlated samples is defined by the size of ``ar_params``
    (``n_coeffs``, i.e. the number of gridpoints) and ``covariance`` (which must be
    equal).

    """

    # check the input
    _check_dataset_form(ar_params, "ar_params", required_vars=("intercept", "coeffs"))
    _check_dataarray_form(ar_params.intercept, "intercept", ndim=1)

    (dim,), size = ar_params.intercept.dims, ar_params.intercept.size
    _check_dataarray_form(
        ar_params.coeffs, "coeffs", ndim=2, required_dims=("lags", dim)
    )
    _check_dataarray_form(covariance, "covariance", ndim=2, shape=(size, size))

    result = _draw_ar_corr_xr_internal(
        intercept=ar_params.intercept,
        coeffs=ar_params.coeffs,
        covariance=covariance,
        time=time,
        realisation=realisation,
        seed=seed,
        buffer=buffer,
        time_dim=time_dim,
        realisation_dim=realisation_dim,
    )

    return result


def _draw_ar_corr_xr_internal(
    intercept,
    coeffs,
    covariance,
    *,
    time,
    realisation,
    seed,
    buffer,
    time_dim="time",
    realisation_dim="realisation",
):

    # get the size and coords of the new dimensions
    n_ts, time_coords = _get_size_and_coord_dict(time, time_dim, "time")
    n_realisations, realisation_coords = _get_size_and_coord_dict(
        realisation, realisation_dim, "realisation"
    )

    # the dimension name of the gridpoints
    (gridpoint_dim,) = set(intercept.dims)
    # make sure non-dimension coords are properly caught
    gridpoint_coords = dict(coeffs[gridpoint_dim].coords)

    out = _draw_auto_regression_correlated_np(
        intercept=intercept.values,
        coeffs=coeffs.transpose(..., gridpoint_dim).values,
        covariance=covariance.values,
        n_samples=n_realisations,
        n_ts=n_ts,
        seed=seed,
        buffer=buffer,
    )

    dims = (realisation_dim, time_dim, gridpoint_dim)

    # TODO: use dict union once requiring py3.9+
    # coords = gridpoint_coords | time_coords | realisation_coords
    coords = {**time_coords, **realisation_coords, **gridpoint_coords}

    out = xr.DataArray(out, dims=dims, coords=coords)

    # for consistency we transpose to time x gridpoint x realisation
    out = out.transpose(time_dim, gridpoint_dim, realisation_dim)

    return out


def _draw_auto_regression_correlated_np(
    *, intercept, coeffs, covariance, n_samples, n_ts, seed, buffer
):
    """
    Draw time series of an auto regression process with possibly spatially-correlated
    innovations

    Creates `n_samples` auto-correlated time series of order `ar_order` and length
    `n_ts` for each set of `n_coeffs` coefficients (typically one set for each grid
    point), the resulting array has shape n_samples x n_ts x n_coeffs. The innovations
    can be spatially correlated.

    Parameters
    ----------
    intercept : float or ndarray of length n_coeffs
        Intercept of the model.
    coeffs : ndarray of shape ar_order x n_coeffs
        The coefficients of the autoregressive process. Must be a 2D array with the
        autoregressive coefficients along axis=0, while axis=1 contains all independent
        coefficients.
    covariance : float or ndarray of shape n_coeffs x n_coeffs
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
        n_samples x n_ts x n_coeffs.

    Notes
    -----
    As this is not a deterministic function it is not called `predict`. "Predicting"
    an autoregressive process does not include the innovations and therefore asymptotes
    towards a certain value (in contrast to this function).
    """
    intercept = np.asarray(intercept)
    covariance = np.atleast_2d(covariance)

    # coeffs assumed to be ar_order x n_coeffs
    ar_order, n_coeffs = coeffs.shape

    # arbitrary lags? no, see: https://github.com/MESMER-group/mesmer/issues/164
    ar_lags = np.arange(1, ar_order + 1, dtype=int)

    # ensure reproducibility (TODO: https://github.com/MESMER-group/mesmer/issues/35)
    np.random.seed(seed)

    # NOTE: 'innovations' is the error or noise term.
    # innovations has shape (n_samples, n_ts + buffer, n_coeffs)
    try:
        cov = scipy.stats.Covariance.from_cholesky(np.linalg.cholesky(covariance))
    except np.linalg.LinAlgError as e:
        if "Matrix is not positive definite" in str(e):
            w, v = np.linalg.eigh(covariance)
            cov = scipy.stats.Covariance.from_eigendecomposition((w, v))
            warnings.warn(
                "Covariance matrix is not positive definite, using eigh instead of cholesky.",
                LinAlgWarning,
            )
        else:
            raise

    innovations = scipy.stats.multivariate_normal.rvs(
        mean=np.zeros(n_coeffs),
        cov=cov,
        size=[n_samples, n_ts + buffer],
    ).reshape(n_samples, n_ts + buffer, n_coeffs)

    out = np.zeros([n_samples, n_ts + buffer, n_coeffs])
    for t in range(ar_order + 1, n_ts + buffer):

        ar = np.sum(coeffs * out[:, t - ar_lags, :], axis=1)

        out[:, t, :] = intercept + ar + innovations[:, t, :]

    return out[:, buffer:, :]


def fit_auto_regression(data, dim, lags):
    """fit an auto regression

    Parameters
    ----------
    data : xr.DataArray
        A ``xr.DataArray`` to estimate the auto regression over.
    dim : str
        Dimension along which to fit the auto regression.
    lags : int
        The number of lags to include in the model.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the estimated parameters of the ``intercept``, the AR
        ``coeffs``, the ``variance`` of the residuals and the number of observations ``nobs``.
    """

    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(data)}")

    # NOTE: this is slowish, see https://github.com/MESMER-group/mesmer/pull/290
    intercept, coeffs, variance, nobs = xr.apply_ufunc(
        _fit_auto_regression_np,
        data,
        input_core_dims=[[dim]],
        output_core_dims=((), ("lags",), (), ()),
        vectorize=True,
        output_dtypes=[float, float, float, int],
        kwargs={"lags": lags},
    )

    if np.ndim(lags) == 0:
        lags = np.arange(lags) + 1

    data_vars = {
        "intercept": intercept,
        "coeffs": coeffs,
        "variance": variance,
        "lags": lags,
        "nobs": nobs,
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
    variance : :obj:`np.array`
       Variance of the residuals.
    nobs: :obj:`np.array``
        Number of observations.
    """

    from statsmodels.tsa.ar_model import AutoReg

    AR_model = AutoReg(data, lags=lags)
    AR_result = AR_model.fit()

    intercept = AR_result.params[0]
    coeffs = AR_result.params[1:]

    # variance of the residuals
    variance = AR_result.sigma2

    nobs = AR_result.nobs

    return intercept, coeffs, variance, nobs


def fit_auto_regression_monthly(monthly_data, time_dim = "time"):
    """fit an auto regression of lag one on monthly data
    Autoregression parameters are estimated for each month and gridpoint separately.
    This is based on the assuption that e.g. June depends on May differently 
    than July on June. Autoregression is fit along `time_dim`.

    Parameters
    ----------
    monthly_data : xr.DataArray
        A ``xr.DataArray`` to estimate the auto regression over. Each month has a value.
    time_dim : str
        Name of the time dimension (dimension along which to fit the auto regression).

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the estimated parameters of the AR(1) process, the ``intercept``and the
        ``slope``.
    """
    if not isinstance(monthly_data, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_data)}")
    
    monthly_data = monthly_data.groupby(time_dim + ".month")
    coeffs = []
    
    for month in range(1,13):
        if month == 1:
            # first January has no previous December
            # and last December has no following January
            prev_month = monthly_data[12].isel(time=slice(0, len(monthly_data[12].time)-1))
            cur_month = monthly_data[month].isel(time=slice(1, len(monthly_data[1].time)))
        else:
            prev_month = monthly_data[month - 1]
            cur_month = monthly_data[month]

        prev_month[time_dim] = cur_month[time_dim]
        
        slope, intercept = xr.apply_ufunc(_fit_autoregression_monthly_np, 
                    cur_month, 
                    prev_month,
                    input_core_dims=[["time"], ["time"]],
                    output_core_dims=[[], []],
                    vectorize=True,
                    )

        coeffs.append(xr.Dataset({"slope": slope, "intercept": intercept}))
    
    return xr.concat(coeffs, dim="month")


def _fit_autoregression_monthly_np(data_month, data_prev_month):
    """fit an auto regression of lag one on monthly data - numpy wrapper
    We use a linear function to relate the independent previous month's 
    data to the dependent current month's data.

    Parameters
    ----------
    data_month : np.array
        A numpy array of the current month's data.
    data_prev_month : np.array
        A numpy array of the previous month's data.
    
    Returns
    -------
    slope : :obj:`np.array`
        The slope of the AR(1) process.
    intercept : :obj:`np.array`
        The intercept of the AR(1) proces.
    """
    
    def lin_func(x, a, b):
        return a * x + b
    
    slope, intercept = scipy.optimize.curve_fit(lin_func, 
                     data_prev_month, # independent variable
                     data_month, # dependent variable
                     bounds=([-1,-np.inf], [1, np.inf]))[0]
    
    return slope, intercept

def predict_auto_regression_monthly(intercept, slope, time, buffer, month_dim = "month"):
    """predict time series of an auto regression process with lag one.

    Parameters
    ----------
    intercept : xr.DataArray of shape (12, n_gridpoints)
        The intercept of the AR(1) process for each month and gridpoint.
    slope : xr.DataArray of shape (12, n_gridpoints)
        The slope of the AR(1) process for each month and gridpoint.
    time : xr.DataArray
        The time coordinates that determines the length of the predicted timeseries and
        that will be the assigned time dimension of the predictions.
    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).
    month_dim : str, default "month"
        Name of the month dimension of the input data needed to stack the predictions.
    
    Returns
    -------
    AR_predictions : xr.DataArray
        Predicted time series of the specified AR(1). The array has shape
        n_time x n_gridpoints.

    """
    # the AR process alone is deterministic so we dont need realisations here
    # or a seed

    AR_predictions = xr.apply_ufunc(_predict_auto_regression_monthly_np, 
                intercept, 
                slope,
                input_core_dims=[[month_dim], [month_dim]],
                output_core_dims=[["year", month_dim]],
                vectorize=True,
                #dask="parallelized",
                output_dtypes=[float],
                kwargs={"n_ts": len(time), "buffer": buffer}
                )

    AR_predictions = AR_predictions.stack({"time": ["year", month_dim]})
    AR_predictions['time'] = time

    return AR_predictions


def _predict_auto_regression_monthly_np(intercept, slope, n_ts, buffer):
    """ predict time series of an auto regression process with lag one - numpy wrapper

    Parameters
    ----------
    intercept : np.array of shape (12,)
        The intercept of the AR(1) process for each month.
    slope : np.array of shape (12,)
        The slope of the AR(1) process for each month.
    n_ts : int
        The number of time steps to predict.
    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).
    
    Returns
    -------
    out : np.array of shape (n_ts/12, 12)
        Predicted time series of the specified AR(1).
    """
    if not n_ts%12 == 0:
        raise ValueError("The number of time steps must be a multiple of 12.")
    n_years = int(n_ts/12)

    out = np.zeros([n_years+buffer, 12])

    for y in range(n_years+buffer):
        for month in range(12):
            prev_month = 11 if month == 0 else month - 1
            if month == 0:
                prev_month = 11
            else:
                prev_month = month - 1
            
            out[y,month] = intercept[month] + slope[month] * out[y-1,prev_month]
        
    return out[buffer:,:]