import numpy as np
import pandas as pd
import xarray as xr

from mesmer.core.utils import _check_dataarray_form, _check_dataset_form


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
    selected_ar_order.data[selected_ar_order.data == 0] = np.NaN

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

    ar_lags = ar_select_order(data, maxlag=maxlag, ic=ic, old_names=False).ar_lags

    # None is returned if no lag is selected
    selected_ar_order = np.NaN if ar_lags is None else ar_lags[-1]

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
    innovations = np.random.multivariate_normal(
        mean=np.zeros(n_coeffs),
        cov=covariance,
        size=[n_samples, n_ts + buffer],
    )

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
        ``coeffs`` and the ``variance`` of the residuals.
    """

    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(data)}")

    # NOTE: this is slowish, see https://github.com/MESMER-group/mesmer/pull/290
    intercept, coeffs, variance = xr.apply_ufunc(
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
        "variance": variance,
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
    std : :obj:`np.array`
        Standard deviation of the residuals.
    """

    from statsmodels.tsa.ar_model import AutoReg

    AR_model = AutoReg(data, lags=lags, old_names=False)
    AR_result = AR_model.fit()

    intercept = AR_result.params[0]
    coeffs = AR_result.params[1:]

    # variance of the residuals
    variance = AR_result.sigma2

    return intercept, coeffs, variance
