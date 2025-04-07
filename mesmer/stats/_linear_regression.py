import numpy as np
import xarray as xr

from mesmer.core._datatreecompat import map_over_datasets
from mesmer.core.datatree import (
    _extract_single_dataarray_from_dt,
    collapse_datatree_into_dataset,
)
from mesmer.core.utils import (
    _check_dataarray_form,
    _check_dataset_form,
    _to_set,
)


class LinearRegression:
    """Ordinary least squares Linear Regression for xr.DataArray objects."""

    def __init__(self):
        self._params = None

    def fit(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
        dim: str,
        weights: xr.DataArray | None = None,
        fit_intercept: bool = True,
    ):
        """
        Fit a linear model

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
            is a xr.Dataset, it must have each predictor as a DataArray.
        target : xr.DataArray
            Target DataArray. Must be 2D and contain `dim`.
        dim : str
            Dimension along which to fit the polynomials.
        weights : xr.DataArray, default: None.
            Individual weights for each sample. Must be 1D and contain `dim`.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model. If set to False, no
            intercept will be used in calculations (i.e. data is expected to be
            centered).
        """

        params = _fit_linear_regression_xr(
            predictors=predictors,
            target=target,
            dim=dim,
            weights=weights,
            fit_intercept=fit_intercept,
        )

        self._params = params

    def predict(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        exclude: str | set[str] | None = None,
    ) -> xr.DataArray:
        """
        Predict using the linear model.

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of ``DataArray`` objects used as predictors or a ``DataTree``, holding each
            predictor in a leaf. Each predictor must be 1D and contain ``dim``. If predictors
            is a ``xr.Dataset``, it must have each predictor as a single ``DataArray``.
        exclude : str or set of str, default: None
            Set of variables to exclude in the prediction. May include ``"intercept"``
            to initialize the prediction with 0.

        Returns
        -------
        prediction : xr.DataArray
            Returns predicted values.
        """

        params = self.params

        exclude = _to_set(exclude)

        non_predictor_vars = {"intercept", "weights", "fit_intercept"}
        required_predictors = set(params.data_vars) - non_predictor_vars - exclude
        available_predictors = set(predictors.keys()) - exclude

        if required_predictors - available_predictors:
            missing = sorted(required_predictors - available_predictors)
            missing = "', '".join(missing)
            raise ValueError(f"Missing predictors: '{missing}'")

        if available_predictors - required_predictors:
            superfluous = map(str, available_predictors - required_predictors)
            superfluous = sorted(superfluous)
            superfluous = "', '".join(superfluous)
            raise ValueError(
                f"Superfluous predictors: '{superfluous}', either params",
                "for this predictor are missing or you forgot to add it to 'exclude'.",
            )

        if "intercept" in exclude:
            prediction = xr.zeros_like(params.intercept)
        else:
            prediction = params.intercept

        # if predictors is a DataTree, rename all data variables to "pred" to avoid conflicts
        # not necessaey if predictors is empty DataTree or only data is in root, i.e. depth == 0
        if isinstance(predictors, xr.DataTree) and not predictors.depth == 0:
            predictors = map_over_datasets(
                lambda ds: ds.rename({var: "pred" for var in ds.data_vars}), predictors
            )

        for key in required_predictors:

            # TODO: fix once .transpose() is possible for DataTree
            signal = predictors[key] * params[key]

            if isinstance(signal, xr.DataTree):
                signal = map_over_datasets(xr.Dataset.transpose, signal)
            else:
                signal = signal.transpose()

            prediction = signal + prediction

        if isinstance(prediction, xr.DataTree):
            prediction = _extract_single_dataarray_from_dt(prediction)

        return prediction.rename("prediction")

    def residuals(
        self,
        predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
        target: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate the residuals of the fitted linear model

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree | xr.Dataset
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
            is a xr.Dataset, it must have each predictor as a DataArray.
        target : xr.DataArray
            Target DataArray. Must be 2D and contain `dim`.

        Returns
        -------
        residuals : xr.DataArray
            Returns residuals - the difference between the predicted values and target.

        """

        prediction = self.predict(predictors)

        residuals = target - prediction

        return residuals.rename("residuals")

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
            required_vars={"intercept", "fit_intercept"},
            optional_vars="weights",
            requires_other_vars=True,
        )

        self._params = params

    @classmethod
    def from_netcdf(cls, filename: str, **kwargs):
        """read params from a netCDF file

        Parameters
        ----------
        filename : str
            Name of the netCDF file to open.
        **kwargs : Any
            Additional keyword arguments passed to ``xr.open_dataset``
        """
        ds = xr.open_dataset(filename, **kwargs)

        obj = cls()
        obj.params = ds

        return obj

    def to_netcdf(self, filename: str, **kwargs):
        """save params to a netCDF file

        Parameters
        ----------
        filename : str
            Name of the netCDF file to save.
        **kwargs : Any
            Additional keyword arguments passed to ``xr.Dataset.to_netcf``
        """

        params = self.params
        params.to_netcdf(filename, **kwargs)


def _fit_linear_regression_xr(
    predictors: dict[str, xr.DataArray] | xr.DataTree | xr.Dataset,
    target: xr.DataArray,
    dim: str,
    weights: xr.DataArray | None = None,
    fit_intercept: bool = True,
) -> xr.Dataset:
    """
    Perform a linear regression

    Parameters
    ----------
    predictors : dict of xr.DataArray | DataTree | xr.Dataset
        A dict of DataArray objects used as predictors or a DataTree, holding each
        predictor in a leaf. Each predictor must be 1D and contain `dim`. If predictors
        is a xr.Dataset, it must have each predictor as a DataArray.
    target : xr.DataArray
        Target DataArray. Must be 2D and contain `dim`.
    dim : str
        Dimension along which to fit the polynomials.
    weights : xr.DataArray, default: None.
        Individual weights for each sample. Must be 1D and contain `dim`.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept
        will be used in calculations (i.e. data is expected to be centered).

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset of intercepts and coefficients. The intercepts and each predictor is an
        individual DataArray.
    """
    # if DataTree only has data in root, extract Dataset
    if isinstance(predictors, xr.DataTree) and predictors.depth == 0:
        predictors = predictors.to_dataset()

    if not isinstance(predictors, dict | xr.DataTree | xr.Dataset):
        raise TypeError(
            f"predictors should be a dict, DataTree or xr.Dataset, got {type(predictors)}."
        )

    if ("weights" in predictors) or ("intercept" in predictors):
        raise ValueError(
            "A predictor with the name 'weights' or 'intercept' is not allowed"
        )

    if dim == "predictor":
        raise ValueError("dim cannot currently be 'predictor'.")

    for key, pred in predictors.items():
        if isinstance(pred, xr.DataTree):
            pred = _extract_single_dataarray_from_dt(pred, name=f"predictor: {key}")

        _check_dataarray_form(pred, ndim=1, required_dims=dim, name=f"predictor: {key}")

    if isinstance(predictors, dict | xr.Dataset):
        predictors_concat = xr.concat(
            tuple(predictors.values()),
            dim="predictor",
            join="exact",
            coords="minimal",
        )
        predictors_concat = predictors_concat.assign_coords(
            {"predictor": list(predictors.keys())}
        )
    elif isinstance(predictors, xr.DataTree):
        # rename all data variables to "pred" to avoid conflicts when concatenating
        def _rename_vars(ds) -> xr.DataTree:
            (var,) = ds.data_vars
            return ds.rename({var: "pred"})

        predictors = map_over_datasets(_rename_vars, predictors)
        # TODO: reconsider collapse_datatree_into_dataset?
        predictors_concat = collapse_datatree_into_dataset(
            predictors, dim="predictor", join="exact", coords="minimal"  # type: ignore[arg-type]
        )
        predictors_concat = predictors_concat["pred"]

    _check_dataarray_form(target, required_dims=dim, name="target")

    if target.ndim == 1:
        # a 2D target array is required, extra dim is squeezed at the end
        extra_dim = f"__{dim}__"
        target = target.expand_dims(extra_dim)
    elif target.ndim != 2:
        raise ValueError(f"target should be 1D or 2D, but has {target.ndim}D")

    # ensure `dim` is equal
    xr.align(predictors_concat, target, join="exact")

    if weights is not None:
        _check_dataarray_form(weights, ndim=1, required_dims=dim, name="weights")
        xr.align(weights, target, join="exact")

    (target_dim,) = list(set(target.dims) - {dim})

    out = _fit_linear_regression_np(
        predictors_concat.transpose(dim, "predictor"),
        target.transpose(dim, target_dim),
        weights,
        fit_intercept,
    )

    # remove (non-dimension) coords from target (#332, #333)
    target = target.drop_vars(target[dim].coords)

    # split `out` into individual DataArrays
    keys = ["intercept"] + list(predictors_concat.coords["predictor"].values)
    data_vars = {key: (target_dim, out[:, i]) for i, key in enumerate(keys)}
    out = xr.Dataset(data_vars, coords=target.coords)

    out["fit_intercept"] = fit_intercept

    if weights is not None:
        out["weights"] = weights

    return out.squeeze()


def _fit_linear_regression_np(predictors, target, weights=None, fit_intercept=True):
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
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept
        will be used in calculations (i.e. data is expected to be centered).

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

    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X=predictors, y=target, sample_weight=weights)

    intercepts = np.atleast_2d(reg.intercept_).T
    coefficients = np.atleast_2d(reg.coef_)

    # necessary when fit_intercept = False
    if not fit_intercept:
        intercepts = np.zeros_like(coefficients[:, :1])

    return np.hstack([intercepts, coefficients])
