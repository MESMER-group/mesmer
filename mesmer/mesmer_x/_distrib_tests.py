# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr

from mesmer.core.datatree import collapse_datatree_into_dataset, map_over_datasets
from mesmer.mesmer_x._expression import Expression


def _test_coeffs_in_bounds(expression: Expression, values_coeffs):

    # checking set boundaries on coefficients
    for coeff in expression.boundaries_coeffs:
        bottom, top = expression.boundaries_coeffs[coeff]

        if coeff not in expression.coefficients_list:
            raise ValueError(
                f"Provided wrong boundaries on coefficient, {coeff}"
                " does not exist in Expression"
            )

        values = values_coeffs[expression.coefficients_list.index(coeff)]

        if np.any(values < bottom) or np.any(top < values):
            # out of boundaries
            return False

    return True


def _test_evol_params(expression: Expression, params):

    # checking set boundaries on parameters
    for param in expression.boundaries_params:
        bottom, top = expression.boundaries_params[param]

        param_values = params[param]

        # out of boundaries
        if np.any(param_values < bottom) or np.any(param_values > top):
            return False

    return True


def _test_support(expression: Expression, params, data):
    # test of the support of the distribution: is there any data out of the
    # corresponding support? dont try testing if there are issues on the parameters

    bottom, top = expression.distrib.support(**params)

    # out of support
    if (
        np.any(np.isnan(bottom))
        or np.any(np.isnan(top))
        or np.any(data < bottom)
        or np.any(data > top)
    ):
        return False

    return True


def _test_proba_value(expression: Expression, threshold_min_proba, params, data):
    """
    Test that all cdf(data) >= threshold_min_proba and 1 - cdf(data) >= threshold_min_proba
    Ensures that data lies within a confidence interval of threshold_min_proba for the tested
    distribution.
    """
    # NOTE: DONT write 'x=data', because 'x' may be called differently for some
    # distribution (eg 'k' for poisson).

    cdf = expression.distrib.cdf(data, **params)
    return np.all(1 - cdf >= threshold_min_proba) and np.all(cdf >= threshold_min_proba)


def validate_coefficients(
    expression: Expression, data_pred, data_targ, coefficients, threshold_min_proba
):
    """validate coefficients

    Parameters
    ----------
    coefficients : numpy array 1D
        Coefficients to validate.

    data_pred : numpy array 1D
        Predictors for the training sample.

    data_targ : numpy array 1D
        Target for the training sample.

    Returns
    -------
    test_coeff : boolean
        True if the coefficients are within conditional_distrib.expression.boundaries_coeffs. If
        False, all other tests will also be set to False and not tested.

    test_param : boolean
        True if parameters are within conditional_distrib.expression.boundaries_params and within the support of the distribution.
        False if not or if test_coeff is False. If False, test_proba will be set to False and not tested.

    test_proba : boolean
        Only tested if conditional_distrib.threshold_min_proba is not None.
        True if the probability of the target samples for the given coefficients
        is above conditional_distrib.threshold_min_proba.
        False if not or if test_coeff or test_param or test_coeff is False.

    distrib : distrib_cov
        The distribution that has been evaluated for the given coefficients.

    """

    test_coeff = _test_coeffs_in_bounds(expression, coefficients)

    # tests on coeffs show already that it won't work: fill in the rest with False
    if not test_coeff:
        return test_coeff, False, False, False, False

    # evaluate the distribution for the predictors and this iteration of coeffs
    params = expression.evaluate_params(coefficients, data_pred)
    # test for the validity of the parameters
    test_param = _test_evol_params(expression, params)

    # tests on params show already that it won't work: fill in the rest with False
    if not test_param:
        return test_coeff, test_param, False, False, False

    # test for the support of the distribution
    test_support = _test_support(expression, params, data_targ)

    # tests on params show already that it won't work: fill in the rest with False
    if not test_support:
        return test_coeff, test_param, test_support, False, False

    # test for the probability of the values
    if threshold_min_proba is None:
        return test_coeff, test_param, test_support, True, params

    else:
        test_proba = _test_proba_value(
            expression, threshold_min_proba, params, data_targ
        )

        # return values for each test and the evaluated distribution
        return test_coeff, test_param, test_support, test_proba, params


def validate_data(data_pred, data_targ, data_weights):
    """
    check data for nans or infs

    Parameters
    ----------
    data_pred
        Predictors for the training sample.

    data_targ
        Target for the training sample.

    data_weights
        Weights for the training sample.
    -------
    """

    def _check_data(data, name):
        # checking for NaN values
        if np.isnan(data).any():
            raise ValueError(f"nan values in {name}")

        # checking for infinite values
        if np.isinf(data).any():
            raise ValueError(f"infinite values in {name}")

    _check_data(data_targ, "target")
    _check_data(data_pred, "predictors")
    _check_data(data_weights, "weights")


def prepare_data(predictors, target, weights, first_guess=None):
    """
    shaping data into DataArrays for first guess, training or evaluation of scores.

    Parameters
    ----------
    predictors : dict of xr.DataArray or xr.Dataset | xr.Dataset | xr.DataTree
        Predictors for the first guess. Must either be a dictionary of xr.DataArray or
        xr.Dataset, each key/item being a predictor; a xr.Dataset with a coordinate
        being the list of predictors, and a variable that contains all predictors; or
        a xr.DataTree with one branch per predictor.
    target : xr.DataArray
        Target DataArray.
    weights : xr.DataArray
        Individual weights for each sample.
    first_guess : xr.Dataset | None default None
        First guess. If provided the function will return a DataArray, with the
        predictor variables stacked along a "predictor" dimension.

    Returns
    -------
    :data_pred:`xr.DataArray`
        shaped predictors for training (predictor, sample)
    :data_targ:`xr.DataArray`
        shaped sample for training (sample, gridpoint)
    :data_weights:`xr.DataArray`
        shaped weights for training (sample)
    :data_first_guess:`xr.DataArray` | None
        shaped first guess for training (coefficients, gridpoint)
    """
    # check formats
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

        predictors_concat_ds = collapse_datatree_into_dataset(
            predictors, dim="predictor", join="exact", coords="minimal"  # type: ignore[arg-type]
        )
        predictors_concat = predictors_concat_ds["pred"]

    else:
        raise Exception(
            "predictors is supposed to be a dict of xr.DataArray, a xr.Dataset or a xr.DataTree"
        )

    # check format of target
    if not (isinstance(target, xr.Dataset) or isinstance(target, xr.DataArray)):
        raise Exception("the target must be a xr.Dataset or xr.DataArray.")

    # check format of weights
    if not (isinstance(weights, xr.Dataset) or isinstance(weights, xr.DataArray)):
        raise Exception("the weights must be a xr.Dataset or xr.DataArray.")

    if isinstance(first_guess, xr.Dataset):
        first_guess = first_guess.to_dataarray(dim="coefficient")

    return predictors_concat, target, weights, first_guess
