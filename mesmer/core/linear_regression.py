from typing import Mapping, Optional

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from .utils import _check_dataarray_form


def linear_regression(
    predictors: Mapping[str, xr.DataArray],
    target: xr.DataArray,
    dim: str,
    weights: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """
    Perform a linear regression

    Parameters
    ----------
    predictors : dict of xr.DataArray
        A dict of DataArray objects used as predictors. Must be 1D and contain `dim`.

    target : xr.DataArray
        Target DataArray. Must be 2D and contain `dim`.

    dim : str
        Dimension along which to fit the polynomials.

    weights : xr.DataArray, default: None.
        Individual weights for each sample. Must be 1D and contain `dim`.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset of intercepts and coefficients. The intercepts and each predictor is an
        individual DataArray.
    """

    if not isinstance(predictors, Mapping):
        raise TypeError(f"predictors should be a dict, got {type(predictors)}.")

    for key, pred in predictors.items():
        _check_dataarray_form(pred, ndim=1, required_dims=dim, name=f"predictor: {key}")

    predictors_concat = xr.concat(
        tuple(predictors.values()), dim="predictor", join="exact"
    )

    _check_dataarray_form(target, ndim=2, required_dims=dim, name="target")

    # ensure `dim` is equal
    xr.align(predictors_concat, target, join="exact")

    if weights is not None:
        _check_dataarray_form(weights, ndim=1, required_dims=dim, name="weights")
        xr.align(weights, target, join="exact")

    target_dim = list(set(target.dims) - {dim})[0]

    out = _linear_regression(
        predictors_concat.transpose(dim, "predictor"),
        target.transpose(dim, target_dim),
        weights,
    )

    # split `out` into individual DataArrays
    keys = ["intercept"] + list(predictors)
    dataarrays = {key: (target_dim, out[:, i]) for i, key in enumerate(keys)}
    out = xr.Dataset(dataarrays, coords=target.coords).drop_vars(dim)

    if weights is not None:
        out["weights"] = weights

    return out


def _linear_regression(predictors, target, weights=None):
    """
    Perform a linear regression - numpy wrapper

    Parameters
    ----------
    predictors : array-like of shape (n_samples, n_predictors)
        Array of predictors

    target : array-like of shape (n_samples, n_targets)
        Array of targets where each row is a sample and each column is a
        different target i.e. variable to be predicted

    weights : array-like of shape (n_samples,)
        Weights for each sample

    Returns
    -------
    :obj:`np.ndarray` of shape (n_targets, n_predictors + 1)
        Array of intercepts and coefficients. Each row is the intercept and
        coefficients for a different target (rows are in same order as the
        columns of ``target``). In each row, the intercept of the regression is
        followed by the intercept for each predictor (in the same order as the
        columns of ``predictors``).
    """
    reg = LinearRegression()
    reg.fit(X=predictors, y=target, sample_weight=weights)

    intercepts = np.atleast_2d(reg.intercept_).T
    coefficients = np.atleast_2d(reg.coef_)

    return np.hstack([intercepts, coefficients])
