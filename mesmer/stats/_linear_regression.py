import warnings

import numpy as np
import xarray as xr
from datatree import DataTree

from mesmer.core.utils import (
    _check_dataarray_form,
    _check_dataset_form,
    _datatree_to_arraydict,
    _to_set,
    collapse_datatree_into_dataset,
)

# TODO: deprecate predictor dicts?


class LinearRegression:
    """Ordinary least squares Linear Regression for xr.DataArray objects."""

    def __init__(self):
        self._params = None

    def fit(
        self,
        predictors: dict[str, xr.DataArray] | DataTree,
        target: xr.DataArray,
        dim: str,
        weights: xr.DataArray | None = None,
        fit_intercept: bool = True,
    ):
        """
        Fit a linear model

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`.
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
        predictors: dict[str, xr.DataArray] | DataTree,
        exclude=None,
    ) -> xr.DataArray:
        """
        Predict using the linear model.

        Parameters
        ----------
        predictors : dict of xr.DataArray | DataTree
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`.
        exclude : str or set of str, default: None
            Set of variables to exclude in the prediction. May include ``"intercept"``
            to initialize the prediction with 0.

        Returns
        -------
        prediction : xr.DataArray
            Returns predicted values.
        """

        if not isinstance(predictors, dict | DataTree):
            raise TypeError(
                f"predictors should be a dict or DataTree, got {type(predictors)}."
            )

        params = self.params

        exclude = _to_set(exclude)

        non_predictor_vars = {"intercept", "weights", "fit_intercept"}
        required_predictors = set(params.data_vars) - non_predictor_vars - exclude
        available_predictors = set(predictors.keys())

        if required_predictors - available_predictors:
            missing = sorted(required_predictors - available_predictors)
            missing = "', '".join(missing)
            raise ValueError(f"Missing predictors: '{missing}'")

        # TODO: should we even warn?
        if available_predictors - required_predictors:
            superfluous = sorted(available_predictors - required_predictors)
            superfluous = "', '".join(superfluous)
            warnings.warn(f"Superfluous predictors: '{superfluous}', will be ignored.")

        if "intercept" in exclude:
            prediction = xr.zeros_like(params.intercept)
        else:
            prediction = params.intercept

        for key in required_predictors:
            predictor = (
                predictors[key]
                if isinstance(predictors, dict)
                # NOTE: once we can store DataArrays in DataTrees this should not be necessary
                else _datatree_to_arraydict(predictors)[key]
            )
            prediction = prediction + predictor * params[key]

        return prediction

    def residuals(
        self,
        predictors: dict[str, xr.DataArray] | DataTree,
        target: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate the residuals of the fitted linear model

        Parameters
        ----------
        predictors : dict of xr.DataArray
            A dict of DataArray objects used as predictors or a DataTree, holding each
            predictor in a leaf. Each predictor must be 1D and contain `dim`.
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
            required_vars={"intercept", "fit_intercept"},
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
        **kwargs : Any
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
        **kwargs : Any
            Additional keyword arguments passed to ``xr.Dataset.to_netcf``
        """

        params = self.params
        params.to_netcdf(filename, **kwargs)


def _fit_linear_regression_xr(
    predictors: dict[str, xr.DataArray] | DataTree,
    target: xr.DataArray,
    dim: str,
    weights: xr.DataArray | None = None,
    fit_intercept: bool = True,
) -> xr.Dataset:
    """
    Perform a linear regression

    Parameters
    ----------
    predictors : dict of xr.DataArray | DataTree
        A dict of DataArray objects used as predictors or a DataTree, holding each
        predictor in a leaf. Each predictor must be 1D and contain `dim`.
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

    if not isinstance(predictors, dict | DataTree):
        raise TypeError(
            f"predictors should be a dict or DataTree, got {type(predictors)}."
        )

    if isinstance(predictors, DataTree):
        if predictors.depth > 1:
            raise ValueError(
                f"Predictors' DataTree must have a depth of 1, not {predictors.depth}."
            )
            # TODO maybe add something about stacking the data?

        # extract dataarrays from the DataTree
        # NOTE: once we can store DataArrays in DataTrees this should not be necessary
        predictors = _datatree_to_arraydict(predictors)

    if ("weights" in predictors) or ("intercept" in predictors):
        raise ValueError(
            "A predictor with the name 'weights' or 'intercept' is not allowed"
        )

    if dim == "predictor":
        raise ValueError("dim cannot currently be 'predictor'.")

    for key, pred in predictors.items():
        _check_dataarray_form(pred, ndim=1, required_dims=dim, name=f"predictor: {key}")

    predictors_concat = xr.concat(
        tuple(predictors.values()),
        dim="predictor",
        join="exact",
        coords="minimal",
    )

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
    keys = ["intercept"] + list(predictors)
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


def prep_linear_regression_data(
    predictors: DataTree,
    target: DataTree,
    stacking_dims: list[str],
    weights: DataTree | None = None,
    collapse_dim: str = "scenario",
    stacked_dim: str = "sample",
) -> tuple[DataTree, xr.Dataset, xr.Dataset] | tuple[DataTree, xr.Dataset]:
    """
    prepares data for Linear Regression:
    1. Broadcasts predictors to target
    2. Collapses DataTrees into DataSets
    3. Stacks the DataSets along the stacking dimensions

    Parameters
    ----------
    predictors : DataTree
        A ``DataTree`` of ``xr.Dataset`` objects used as predictors. The ``DataTree``
        must have subtrees for each predictor each of which has to have at least one
        leaf, holding a ``xr.Dataset`` representing a scenario. The subtrees of
        different predictors must be isomorphic (i.e. have the save scenarios). The ``xr.Dataset``
        must at least contain `dim` and each ``xr.Dataset`` must only hold one data variable.
    target : DataTree
        A ``DataTree``holding the targets. Must be isomorphic to the predictor subtrees, i.e.
        have the same scenarios. Each leaf must hold a ``xr.Dataset`` which must be at least 2D
        and contain `dim`, but may also contain a dimension for ensemble members.
    stacking_dims : list[str]
        Dimension(s) to stack.
    weights : DataTree, default: None.
        Individual weights for each sample, must be isomorphic to target. Must at least contain
        `dim`, and must have the ensemble member dimesnion if target has it.
    collapse_dim : str, default: "scenario"
        Dimension along which to collapse the DataTrees, will automatically be added to the
        stacking dims.
    stacked_dim : str, default: "sample"
        Name of the stacked dimension.

    Returns
    -------
    tuple
        Tuple of the prepared predictors, target and weights, where the predictors and target are
        stacked along the stacking dimensions and the weights are stacked along the stacking dimensions
        and the ensemble member dimension.

    Notes
    -----
    Dimensions which exist along the target but are not in the stacking_dims will be excluded from the
    broadcasting of the predictors.
    """

    stacking_dims.append(collapse_dim)

    # exclude target dimensions from broadcasting which are not in the stacking_dims
    exclude_dim = set(target.leaves[0].ds.dims) - set(stacking_dims)

    # predictors need to be
    for key, subtree in predictors.items():
        # 1) broadcast to target
        broadcasted = subtree.broadcast_like(target, exclude=exclude_dim)
        # 2) collapsed into DataSets
        ds = collapse_datatree_into_dataset(broadcasted, dim=collapse_dim)
        # 3) stacked
        predictors[key] = DataTree(
            ds.stack({stacked_dim: stacking_dims}, create_index=False).dropna(
                dim=stacked_dim
            )
        )

    # target needs to be
    # 1) collapsed into DataSet
    target_ds = collapse_datatree_into_dataset(target, dim=collapse_dim)
    # 2) stacked
    target_stacked = target_ds.stack(
        {stacked_dim: stacking_dims}, create_index=False
    ).dropna(dim=stacked_dim)

    # weights need to be
    if weights is not None:
        # 1) collapsed into DataSet
        weights_ds = collapse_datatree_into_dataset(weights, dim=collapse_dim)
        # 2) stacked
        weights_stacked = weights_ds.stack(
            {stacked_dim: stacking_dims}, create_index=False
        ).dropna(dim=stacked_dim)
        return predictors, target_stacked, weights_stacked

    return predictors, target_stacked
