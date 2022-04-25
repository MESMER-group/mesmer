from typing import Mapping, Optional

import numpy as np
import xarray as xr

from .utils import _check_dataarray_form, _check_dataset_form


class LinearRegression:
    """Ordinary least squares Linear Regression for xr.DataArray objects."""

    def __init__(self):
        self._params = None

    def fit(
        self,
        predictors: Mapping[str, xr.DataArray],
        target: xr.DataArray,
        dim: str,
        weights: Optional[xr.DataArray] = None,
    ):
        """
        Fit a linear model

        Parameters
        ----------
        predictors : dict of xr.DataArray
            A dict of DataArray objects used as predictors. Must be 1D and contain
            `dim`.

        target : xr.DataArray
            Target DataArray. Must be 2D and contain `dim`.

        dim : str
            Dimension along which to fit the polynomials.

        weights : xr.DataArray, default: None.
            Individual weights for each sample. Must be 1D and contain `dim`.
        """

        params = _fit_linear_regression_xr(
            predictors=predictors,
            target=target,
            dim=dim,
            weights=weights,
        )

        self._params = params

    def predict(
        self,
        predictors: Mapping[str, xr.DataArray],
    ):
        """
        Predict using the linear model.

        Parameters
        ----------
        predictors : dict of xr.DataArray
            A dict of DataArray objects used as predictors. Must be 1D and contain `dim`.

        Returns
        -------
        prediction : xr.DataArray
            Returns predicted values.
        """

        params = self.params

        required_predictors = set(params.data_vars) - set(["intercept", "weights"])
        available_predictors = set(predictors.keys())

        if required_predictors != available_predictors:
            raise ValueError("Missing or superflous predictors.")

        prediction = params.intercept
        for key in required_predictors:
            prediction = prediction + predictors[key] * params[key]

        return prediction

    def residuals(
        self,
        predictors: Mapping[str, xr.DataArray],
        target: xr.DataArray,
    ):
        """
        Calculate the residuals of the fitted linear model

        Parameters
        ----------
        predictors : dict of xr.DataArray
            A dict of DataArray objects used as predictors. Must be 1D and contain `dim`.

        target : xr.DataArray
            Target DataArray. Must be 2D and contain `dim`.

        Returns
        -------
        residuals : xr.DataArray
            Returns residuals - the difference between the predicted values and target.

        """

        prediction = self.predict(predictors)

        residuals = target - prediction

        return residuals

    @property
    def params(self):
        """The parameters of this estimator."""

        if self._params is None:
            raise ValueError(
                "'params' not set - call `fit` or assign them to "
                "`LinearRegression().params`."
            )

        return self._params

    @params.setter
    def params(self, params):

        _check_dataset_form(
            params,
            "params",
            required_vars="intercept",
            optional_vars="weights",
            requires_other_vars=True,
        )

        self._params = params

    @classmethod
    def from_netcdf(cls, filename, **kwargs):
        """read params from a netCDF file

        Parameters
        ----------
        filename : str
            Name of the netCDF file to open.
        kwargs : Any
            Additional keyword arguments passed to ``xr.open_dataset``
        """
        ds = xr.open_dataset(filename, **kwargs)

        obj = cls()
        obj.params = ds

        return obj

    def to_netcdf(self, filename, **kwargs):
        """save params to a netCDF file

        Parameters
        ----------
        filename : str
            Name of the netCDF file to save.
        kwargs : Any
            Additional keyword arguments passed to ``xr.Dataset.to_netcf``
        """

        params = self.params()
        params.to_netcdf(filename, **kwargs)


def _fit_linear_regression_xr(
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

    if ("weights" in predictors) or ("intercept" in predictors):
        raise ValueError(
            "A predictor with the name 'weights' or 'intercept' is not allowed"
        )

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

    out = _fit_linear_regression_np(
        predictors_concat.transpose(dim, "predictor"),
        target.transpose(dim, target_dim),
        weights,
    )

    # split `out` into individual DataArrays
    keys = ["intercept"] + list(predictors)
    dataarrays = {key: (target_dim, out[:, i]) for i, key in enumerate(keys)}
    out = xr.Dataset(dataarrays, coords=target.coords)
    if dim in out.coords:
        out = out.drop_vars(dim)

    if weights is not None:
        out["weights"] = weights

    return out


def _fit_linear_regression_np(predictors, target, weights=None):
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

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(X=predictors, y=target, sample_weight=weights)

    intercepts = np.atleast_2d(reg.intercept_).T
    coefficients = np.atleast_2d(reg.coef_)

    return np.hstack([intercepts, coefficients])
