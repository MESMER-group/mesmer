import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scipy
import xarray as xr

from mesmer.core.datatree import (
    _datatree_wrapper,
    collapse_datatree_into_dataset,
    map_over_datasets,
)
from mesmer.core.utils import (
    LinAlgWarning,
    _check_dataarray_form,
    _check_dataset_form,
    _set_threads_from_options,
)


def _scen_ens_inputs_to_dt(objs: Sequence) -> xr.DataTree:
    """Helper function to convert a sequence of objects to a DataTree"""

    if isinstance(objs[0], xr.DataTree):
        if len(objs) != 1:
            raise ValueError("Only one DataTree can be passed.")
        dt = objs[0]

    elif isinstance(objs[0], xr.DataArray):
        # TODO: in the future might be able to use DataTree.from_array_dict
        # see https://github.com/pydata/xarray/issues/9486
        # with just da_dict = {f"da_{i}": da for i, da in enumerate(objs)}
        ds_dict = {f"scen_{i}": da._to_temp_dataset() for i, da in enumerate(objs)}
        dt = xr.DataTree.from_dict(ds_dict)

    elif isinstance(objs[0], dict):
        if len(objs) != 1:
            raise ValueError("Only one dictionary can be passed.")
        da_dict = objs[0]
        # TODO: in the future might be able to use DataTree.from_array_dict(da_dict)
        # see https://github.com/pydata/xarray/issues/9486
        ds_dict = {f"{key}": da._to_temp_dataset() for key, da in da_dict.items()}
        dt = xr.DataTree.from_dict(ds_dict)

    else:
        raise ValueError(
            "Expected either a DataTree, a dictionary of xr.DataArrays",
            f"or several DataArrays as objs, got {type(objs[0])}.",
        )

    return dt


def _extract_and_apply_to_da(func: Callable) -> Callable:

    def _inner(ds: xr.Dataset, **kwargs) -> xr.Dataset:

        name, *others = ds.data_vars
        if others:
            raise ValueError("Dataset must have only one data variable.")

        x = func(ds[name], **kwargs)

        return x.to_dataset() if isinstance(x, xr.DataArray) else x

    return _inner


def select_ar_order_scen_ens(
    *objs: xr.DataArray | dict[str, xr.DataArray] | xr.DataTree,
    dim: str,
    ens_dim: str | None,
    maxlag: int,
    ic: Literal["bic", "aic", "hqic"] = "bic",
) -> xr.DataArray:
    """
    Select the order of an autoregressive process and potentially calculate the median
    over ensemble members and scenarios

    Parameters
    ----------
    objs : DataTree, xr.DataArrays or dict of DataArrays
        A DataTree, ``xr.DataArray``s or dict of ``xr.DataArray`` to estimate the auto regression order over.
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
    dt = _scen_ens_inputs_to_dt(objs)
    return _select_ar_order_scen_ens_dt(dt, dim, ens_dim, maxlag, ic)


def _select_ar_order_scen_ens_dt(
    dt: xr.DataTree,
    dim: str,
    ens_dim: str | None,
    maxlag: int,
    ic: Literal["bic", "aic", "hqic"] = "bic",
) -> xr.DataArray:
    """
    Select the order of an autoregressive process and potentially calculate the median
    over ensemble members and scenarios

    Parameters
    ----------
    dt : a DataTree
        A DataTree holding one or several ``xr.Dataset`` to estimate the auto regression order over,
        each representing one scenario, potentially with several ensemble members along `ens_dim`.
        Each ``xr.DataSet`` should only hold one variable, the one for which to estimate the autoregression.
    dim : str
        Dimension along which to determine the order.
    ens_dim : str
        Dimension name of the ensemble members. Must be the same for all scenarios and have coordinates if not None.
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

    ar_order_scen = map_over_datasets(
        _extract_and_apply_to_da(select_ar_order),
        dt,
        kwargs={"dim": dim, "maxlag": maxlag, "ic": ic},
    )

    # TODO: think about weighting?
    def _ens_quantile(ds, ens_dim):
        if ens_dim in ds.dims:
            return ds.quantile(dim=ens_dim, q=0.5, method="nearest")
        return ds

    ar_order_ens_median = map_over_datasets(_ens_quantile, ar_order_scen, ens_dim)

    ar_order_ens_median_ds = collapse_datatree_into_dataset(
        ar_order_ens_median, dim="scen"
    )

    ar_order = ar_order_ens_median_ds.quantile(
        dim="scen", q=0.5, method="nearest"
    ).selected_ar_order

    if not np.isnan(ar_order).any():
        ar_order = ar_order.astype(int)

    return ar_order


def fit_auto_regression_scen_ens(
    *objs: xr.DataArray | dict[str, xr.DataArray] | xr.DataTree,
    dim: str,
    ens_dim: str | None,
    lags: int | xr.DataArray,
) -> xr.Dataset:
    """
    fit an auto regression and potentially calculate the mean over ensemble members
    and scenarios

    Parameters
    ----------
    obj : DataTree, xr.DataArrays or dict of DataArrays
        A ``DataTree`` holding one or several ``xr.Dataset``, ``xr.DataArray``s, or dict of ``xr.DataArray``s to estimate the auto regression order over,
        each representing one scenario, potentially with several ensemble members along `ens_dim`.
        If a ``DataTree``, each ``xr.Dataset`` should only hold one variable, the one for which to estimate the autoregression.
    dim : str
        Dimension along which to fit the auto regression (often time).
    ens_dim : str
        Dimension name of the ensemble members, None if no ensemble is provided.  Must be the same for all scenarios and have coordinates if not None.
    lags : int
        The number of lags to include in the model.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the estimated parameters of the ``intercept``, the AR
        ``coeffs`` and the ``variance`` of the residuals.

    Notes
    -----
    If `ens_dim` is not `None`, calculates the mean auto regression first over all ensemble
    members and then over scenarios. This is done to weight scenarios equally, consequently
    ensemble members are not weighted equally, if the number of members differs between scenarios.
    If no ensemble members are provided, the mean is calculated over scenarios only.
    """
    dt = _scen_ens_inputs_to_dt(objs)
    return _fit_auto_regression_scen_ens_dt(dt, dim, ens_dim, lags)


def _fit_auto_regression_scen_ens_dt(
    dt: xr.DataTree, dim: str, ens_dim: str | None, lags: int | xr.DataArray
) -> xr.Dataset:
    """
    fit an auto regression and potentially calculate the mean over ensemble members
    and scenarios

    Parameters
    ----------
    dt : a DataTree
        A ``DataTree`` holding one or several ``xr.Dataset`` to estimate the auto regression order over,
        each representing one scenario, potentially with several ensemble members along `ens_dim`.
        Each ``xr.DataSet`` should only hold one variable, the one for which to estimate the autoregression.
    dim : str
        Dimension along which to fit the auto regression (often time).
    ens_dim : str
        Dimension name of the ensemble members, None if no ensemble is provided.  Must be the same for all scenarios and have coordinates if not None.
    lags : int
        The number of lags to include in the model.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the estimated parameters of the ``intercept``, the AR
        ``coeffs`` and the ``variance`` of the residuals.

    Notes
    -----
    If `ens_dim` is not `None`, calculates the mean auto regression first over all ensemble
    members and then over scenarios. This is done to weight scenarios equally, consequently
    ensemble members are not weighted equally, if the number of members differs between scenarios.
    If no ensemble members are provided, the mean is calculated over scenarios only.
    """

    ar_params_scen = map_over_datasets(
        _extract_and_apply_to_da(fit_auto_regression),
        dt,
        kwargs={"dim": dim, "lags": int(lags)},
    )

    # TODO: think about weighting! see https://github.com/MESMER-group/mesmer/issues/307
    def _ens_mean(ds, ens_dim):
        if ens_dim in ds.dims:
            return ds.mean(ens_dim)
        return ds

    ar_params_scen = map_over_datasets(_ens_mean, ar_params_scen, ens_dim)

    ar_params_scen = collapse_datatree_into_dataset(ar_params_scen, dim="scen")

    # return the mean over all scenarios
    ar_params = ar_params_scen.mean("scen")

    return ar_params


# ======================================================================================


def select_ar_order(
    data: xr.DataArray, dim: str, maxlag: int, ic: Literal["bic", "aic", "hqic"] = "bic"
) -> xr.DataArray:
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


def _get_size_and_coord_dict(coords_or_size, dim, name) -> tuple[int, dict]:

    if isinstance(coords_or_size, int):
        size = coords_or_size
        coord_dict: dict = {}

        return size, coord_dict

    if not isinstance(coords_or_size, xr.DataArray | pd.Index):
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
    ar_params: xr.Dataset,
    *,
    time: int | xr.DataArray | pd.Index,
    realisation: int | xr.DataArray | pd.Index,
    seed: int | xr.DataTree,
    buffer: int,
    time_dim: str = "time",
    realisation_dim: str = "realisation",
) -> xr.Dataset:
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

    seed : int | xr.DataTree
        Seed used to initialize the pseudo-random number generator. Can be an int or a xr.DataTree that
        contains Datasets with a single variable "seed" with the seed value, to draw samples for multiple scenarios.

    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).

    Returns
    -------
    out : Dataset
        Drawn realizations of the specified autoregressive process. The array has shape
        n_time x n_coeffs x n_realisations.

    """

    return _draw_auto_regression_uncorrelated(
        seed,
        ar_params,
        time=time,
        realisation=realisation,
        buffer=buffer,
        time_dim=time_dim,
        realisation_dim=realisation_dim,
    )


@_datatree_wrapper
def _draw_auto_regression_uncorrelated(
    seed: int | xr.DataTree,
    ar_params: xr.Dataset,
    *,
    time: int | xr.DataArray | pd.Index,
    realisation: int | xr.DataArray | pd.Index,
    buffer: int,
    time_dim: str = "time",
    realisation_dim: str = "realisation",
) -> xr.Dataset:

    # NOTE: we use variance and not std since we use multivariate normal
    # also to draw univariate realizations
    # check the input
    _check_dataset_form(
        ar_params, "ar_params", required_vars={"intercept", "coeffs", "variance"}
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

    if isinstance(seed, xr.Dataset):
        seed = int(seed.seed.item())

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

    return result.rename("samples").to_dataset()


def draw_auto_regression_correlated(
    ar_params: xr.Dataset,
    covariance: xr.DataArray,
    *,
    time: int | xr.DataArray | pd.Index,
    realisation: int | xr.DataArray | pd.Index,
    seed: int | xr.DataTree,
    buffer: int,
    time_dim: str = "time",
    realisation_dim: str = "realisation",
) -> xr.Dataset:
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

    seed : int | xr.DataTree
        Seed used to initialize the pseudo-random number generator. Can be an int or a xr.DataTree that
        contains Datasets with a single variable "seed" with the seed value, used to draw samples for multiple scenarios.

    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).

    Returns
    -------
    out : Dataset
        Drawn realizations of the specified autoregressive process. The array has shape
        n_time x n_coeffs x n_realisations.

    Notes
    -----
    The number of (spatially-)correlated samples is defined by the size of ``ar_params``
    (``n_coeffs``, i.e. the number of gridpoints) and ``covariance`` (which must be
    equal).

    """

    return _draw_auto_regression_correlated(
        seed,
        ar_params,
        covariance,
        time=time,
        realisation=realisation,
        buffer=buffer,
        time_dim=time_dim,
        realisation_dim=realisation_dim,
    )


@_datatree_wrapper
def _draw_auto_regression_correlated(
    seed: int | xr.DataTree,
    ar_params: xr.Dataset,
    covariance: xr.DataArray,
    *,
    time: int | xr.DataArray | pd.Index,
    realisation: int | xr.DataArray | pd.Index,
    buffer: int,
    time_dim: str = "time",
    realisation_dim: str = "realisation",
) -> xr.Dataset:

    # check the input
    _check_dataset_form(ar_params, "ar_params", required_vars={"intercept", "coeffs"})
    _check_dataarray_form(ar_params.intercept, "intercept", ndim=1)

    (dim,), size = ar_params.intercept.dims, ar_params.intercept.size
    _check_dataarray_form(
        ar_params.coeffs, "coeffs", ndim=2, required_dims={"lags", dim}
    )
    _check_dataarray_form(covariance, "covariance", ndim=2, shape=(size, size))

    if isinstance(seed, xr.Dataset):
        seed = int(seed.seed.item())

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

    return result.rename("samples").to_dataset()


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

    coords = gridpoint_coords | time_coords | realisation_coords

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

    # ensure reproducibility
    rng = np.random.default_rng(seed)

    innovations = _draw_innovations_correlated_np(
        covariance, rng, n_coeffs, n_samples, n_ts, buffer
    )

    out = np.zeros([n_samples, n_ts + buffer, n_coeffs])
    for t in range(ar_order + 1, n_ts + buffer):

        ar = np.sum(coeffs * out[:, t - ar_lags, :], axis=1)

        out[:, t, :] = intercept + ar + innovations[:, t, :]

    return out[:, buffer:, :]


@_set_threads_from_options()
def _draw_innovations_correlated_np(
    covariance, rng, n_gridcells, n_samples, n_ts, buffer
):
    # NOTE: 'innovations' is the error or noise term.
    # innovations has shape (n_samples, n_ts + buffer, n_coeffs)
    cov = None
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

    innovations = scipy.stats.multivariate_normal.rvs(
        mean=np.zeros(n_gridcells),
        cov=cov,
        size=[n_samples, n_ts + buffer],
        random_state=rng,
    ).reshape(n_samples, n_ts + buffer, n_gridcells)

    return innovations


def fit_auto_regression(
    data: xr.DataArray, dim: str, lags: int | Sequence[int]
) -> xr.Dataset:
    """fit an auto regression

    Parameters
    ----------
    data : xr.DataArray
        A ``xr.DataArray`` to estimate the auto regression over.
    dim : str
        Dimension along which to fit the auto regression.
    lags : int | Sequence[int]
        The number of lags or list of lags to include in the model.
        If int, then all lags up to ``lags`` will be included.

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

    if isinstance(lags, int):
        lags = list(range(1, lags + 1))

    # return intercept, coeffs, variance, lags, nobs
    data_vars = {
        "intercept": intercept,
        "coeffs": coeffs,
        "variance": variance,
        "lags": lags,
        "nobs": nobs,
    }

    return xr.Dataset(data_vars)


def _fit_auto_regression_np(data: np.ndarray, lags: int | Sequence[int]):
    """
    fit an auto regression - numpy wrapper

    Parameters
    ----------
    data : np.array
        A numpy array to estimate the auto regression over. Must be 1D.
    lags : int | Sequence[int]
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
    variance, nobs = AR_result.sigma2, AR_result.nobs

    return intercept, coeffs, variance, nobs


def fit_auto_regression_monthly(
    monthly_data: xr.DataArray, time_dim: str = "time"
) -> xr.Dataset:
    """fit a cyclo-stationary auto-regressive process of lag one (AR(1)) on monthly
    data. The parameters are estimated for each month and gridpoint separately.
    This is based on the assumption that e.g. June depends on May differently
    than July on June. The auto regression is fit along `time_dim`.

    A cyclo-stationary AR(1) process is defined as follows:

    .. math::

        \\mathbf{X}_{t, \\tau} = \\alpha_{0, \\tau} + \\alpha_{1, \\tau} \\mathbf{X}_{t, \\tau -1}
        + \\epsilon_{t, \\tau}


    where :math:`\\tau \\in \\{1, \\ldots, N\\}` counts the seasons of some seasonal cycle, here the
    months of a year :math:`(N=12)` and :math:`t` counts the repetitions of this seasonal cycle,
    here the years. Here :math:`\\epsilon` is a white noise process, i.e. :math:`\\epsilon \\sim N(0, \\sigma^2)`.
    The covariance matrix of the driving white noise process should be estimated on the residuals of the AR(1)
    process. The residuals are returned here and should be passed to
    :func:`find_localized_empirical_covariance_monthly <mesmer.stats.find_localized_empirical_covariance_monthly>`.

    For more information refer to Storch and Zwiers (1999) Chapter 10.3.8 [1].

    [1] Storch H von, Zwiers FW. Statistical Analysis in Climate Research.
        **Cambridge University Press; 1999,** `DOI:10.1017/CBO9780511612336 <https://doi.org/10.1017/CBO9780511612336>`_.


    Parameters
    ----------
    monthly_data : ``xr.DataArray``
        A ``xr.DataArray`` to estimate the auto regression over, must contain `time_dim` and can have more dims,
        for example a gridcell and/or a member dim. Each month has a value.
    time_dim : str
        Name of the time dimension (dimension along which to fit the auto regression).

    Returns
    -------
    obj : ``xr.Dataset``
        Dataset containing

        - the ``intercept`` for each month of the AR(1) process,
        - the ``slope`` for each month and
        - the ``residuals`` (needed for the estimation of the covariance matrices).

        ``intercept`` and ``slope`` have `"month"` and the additional dims of the input data as dimensions,
        the residuals have `time_dim` and the additional dims of the input data as dimensions.
    """
    _check_dataarray_form(monthly_data, "monthly_data", required_coords=time_dim)
    monthly_groups = monthly_data.groupby(f"{time_dim}.month")
    ar_params_res = []

    (sample_dim,) = monthly_data[time_dim].dims

    residuals = xr.full_like(monthly_data, fill_value=np.nan)
    # we loose one timestep
    residuals = residuals.isel({sample_dim: slice(1, None)})
    residuals.name = "residuals"

    for month in range(1, 13):
        if month == 1:
            # first January has no previous December
            # and last December has no following January
            prev_month = monthly_groups[12].isel({sample_dim: slice(None, -1)})
            cur_month = monthly_groups[1].isel({sample_dim: slice(1, None)})
            i = 11  # values only start the second year & first ts is removed
        else:
            prev_month = monthly_groups[month - 1]
            cur_month = monthly_groups[month]
            i = month - 2

        slope, intercept, resids = xr.apply_ufunc(
            _fit_auto_regression_monthly_np,
            cur_month,
            prev_month,
            input_core_dims=[[sample_dim], [sample_dim]],
            output_core_dims=[[], [], [sample_dim]],
            exclude_dims={sample_dim},
            vectorize=True,
        )

        # assign residuals, so the order is kept
        residuals[{sample_dim: slice(i, None, 12)}] = resids

        ar_params_res.append(xr.Dataset({"slope": slope, "intercept": intercept}))

    month_dim = xr.Variable("month", np.arange(1, 13))
    ar_params = xr.concat(ar_params_res, dim=month_dim)

    return xr.merge([ar_params, residuals])


def _fit_auto_regression_monthly_np(data_month, data_prev_month):
    """fit an auto regression of lag one (AR(1)) on monthly data
    We use a linear function to relate the previous month's data
    (predictor/independent variable) to the current month's data
    (target/ dependent variable).

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
        The intercept of the AR(1) process.
    """

    slope, intercept = np.polyfit(data_prev_month, data_month, deg=1)

    residuals = data_month - (intercept + slope * data_prev_month)

    return slope, intercept, residuals


def draw_auto_regression_monthly(
    ar_params: xr.Dataset,
    covariance: xr.DataArray,
    *,
    time: xr.DataArray | pd.Index,
    n_realisations: int,
    seed: int | xr.DataTree,
    buffer: int,
    time_dim: str = "time",
    realisation_dim: str = "realisation",
) -> xr.Dataset:
    """draw time series of a cyclo-stationary auto-regressive process of lag one (AR(1))
    using individual parameters for each month including spatially-correlated innovations.
    For more information on the cyclo-stationary AR(1) process please refer to
    :func:`fit_auto_regression_monthly <mesmer.stats.fit_auto_regression_monthly>`.

    Parameters
    ----------
    ar_params : xr.Dataset
        Dataset containing the estimated parameters of the AR1 process. Must contain the
        following DataArray objects:

        - intercept
        - slope

        both of shape (12, n_gridpoints).
    covariance : xr.DataArray of shape (12, n_gridpoints, n_gridpoints)
        The covariance matrix representing the spatially correlated driving
        white noise process for each month. Must be symmetric and at least
        positive-semidefinite.
        Used to draw spatially-correlated innovations using a multivariate normal.
    time : xr.DataArray | pd.Index
        The time coordinates that determines the length of the predicted timeseries and
        that will be the assigned time dimension of the predictions.
    n_realisations : int
        The number of realisations to draw.
    seed : int | xr.DataTree
        Seed used to initialize the pseudo-random number generator. Can be an int or a xr.DataTree that
        contains Datasets with a single variable "seed" with the seed value, used to draw samples for multiple scenarios.
    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result).
    time_dim : str, default "time"
        Name of the time dimension for the output data.
    realisation_dim : str, default "realisation"
        Name of the realisation dimension for the output data.

    Returns
    -------
    result : xr.Dataset with samples of shape (n_realisations, n_timesteps, n_gridpoints)
        Predicted time series of the specified AR(1) process including spatially
        correlated innovations. The array has shape n_timesteps x n_gridpoints.

    """

    return _draw_auto_regression_monthly(
        seed,
        ar_params,
        covariance,
        time=time,
        n_realisations=n_realisations,
        buffer=buffer,
        time_dim=time_dim,
        realisation_dim=realisation_dim,
    )


@_datatree_wrapper
def _draw_auto_regression_monthly(
    seed: int | xr.DataTree,
    ar_params: xr.Dataset,
    covariance: xr.DataArray,
    *,
    time: xr.DataArray | pd.Index,
    n_realisations: int,
    buffer: int,
    time_dim: str = "time",
    realisation_dim: str = "realisation",
) -> xr.Dataset:

    # NOTE: seed must be the first positional argument for map_over_datasets to work

    # check input
    _check_dataset_form(ar_params, "ar_params", required_vars={"intercept", "slope"})
    month_dim, gridcell_dim = ar_params.intercept.dims
    n_months, size = ar_params.intercept.shape
    _check_dataarray_form(
        ar_params.intercept,
        "intercept",
        ndim=2,
        required_dims={month_dim, gridcell_dim},
    )
    _check_dataarray_form(
        ar_params.slope, "slope", ndim=2, required_dims={month_dim, gridcell_dim}
    )
    _check_dataarray_form(
        covariance, "covariance", ndim=3, shape=(n_months, size, size)
    )

    if isinstance(seed, xr.Dataset):
        seed = int(seed.seed.item())

    result = _draw_ar_corr_monthly_xr_internal(
        intercept=ar_params.intercept,
        slope=ar_params.slope,
        covariance=covariance,
        time=time,
        realisation=n_realisations,
        seed=seed,
        buffer=buffer,
        time_dim=time_dim,
        realisation_dim=realisation_dim,
    )

    return result.rename("samples").to_dataset()


def _draw_ar_corr_monthly_xr_internal(
    intercept,
    slope,
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
    (_, gridpoint_dim) = intercept.dims
    # make sure non-dimension coords are properly caught
    gridpoint_coords = dict(slope[gridpoint_dim].coords)

    out = _draw_auto_regression_monthly_np(
        intercept=intercept.values,
        slope=slope.transpose(..., gridpoint_dim).values,
        covariance=covariance.values,
        n_samples=n_realisations,
        n_ts=n_ts,
        seed=seed,
        buffer=buffer,
    )

    dims = (realisation_dim, time_dim, gridpoint_dim)

    coords = gridpoint_coords | time_coords | realisation_coords

    out = xr.DataArray(out, dims=dims, coords=coords)

    # for consistency we transpose to time x gridpoint x realisation
    out = out.transpose(time_dim, gridpoint_dim, realisation_dim)

    return out


def _draw_auto_regression_monthly_np(
    intercept, slope, covariance, n_samples, n_ts, seed, buffer
):
    """draw time series of an auto regression process with lag one
    (AR(1)) using individual parameters for each month - numpy wrapper

    Parameters
    ----------
    intercept : np.array of shape (12, gridpoints)
        The intercept of the AR(1) process for each month.
    slope : np.array of shape (12, gridpoints)
        The slope of the AR(1) process for each month.
    covariance: np.array of shape (12, n_gridpoints, n_gridpoints)
        The covariance matrix representing spatial correlation between gridpoints for each month.
    n_samples : int
        The number of realisations to draw.
    n_ts : int
        The number of time steps to draw (i.e. the number of months in the whole timeseries).
    buffer : int
        Buffer to initialize the autoregressive process (ensures that start at 0 does
        not influence overall result). The number given is used for every month such
        that at the end 12*buffer months are cut off.

    Returns
    -------
    out : np.array of shape (n_samples, n_ts, n_gridpoints)
        Predicted time series of the specified AR(1) including spatially correlated innovations.
    """
    intercept = np.asarray(intercept)
    covariance = np.atleast_3d(covariance)

    _, n_gridcells = intercept.shape

    # ensure reproducibility
    rng = np.random.default_rng(seed)

    # draw innovations for each month
    innovations = np.zeros([n_samples, n_ts // 12 + buffer, 12, n_gridcells])

    for month in range(12):
        cov_month = covariance[month, :, :]
        innovations[:, :, month, :] = _draw_innovations_correlated_np(
            cov_month, rng, n_gridcells, n_samples, n_ts // 12, buffer
        )

    # reshape innovations into continuous time series
    innovations = innovations.reshape(n_samples, n_ts + buffer * 12, n_gridcells)

    # predict auto-regressive process using innovations
    out = np.zeros([n_samples, n_ts + buffer * 12, n_gridcells])
    for t in range(1, n_ts + buffer * 12):
        month = t % 12
        out[:, t, :] = (
            intercept[month, :]
            + slope[month, :] * out[:, t - 1, :]
            + innovations[:, t, :]
        )

    return out[:, buffer * 12 :, :]
