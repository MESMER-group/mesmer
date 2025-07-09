# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np

from mesmer.mesmer_x._expression import Expression


def _coeffs_in_bounds(expression: Expression, values_coeffs):

    # checking set boundaries on coefficients
    for coeff, (bottom, top) in expression.boundaries_coeffs.items():

        values = values_coeffs[expression.coefficients_list.index(coeff)]

        if np.any(values < bottom) or np.any(top < values):
            # out of boundaries
            return False

    return True


def _params_in_bounds(expression: Expression, params):

    # checking set boundaries on parameters
    for param, (bot, top) in expression.boundaries_params.items():

        param_values = params[param]

        # out of boundaries
        if (
            # only check values if bot/ top are not -+inf
            (not np.isinf(bot) and np.min(param_values) < bot)
            or (not np.isinf(top) and np.max(param_values) > top)
        ):
            return False

    return True


def _param_in_bounds(expression: Expression, param_values, name):
    """as params_in_bounds but for a single param (faster)"""

    # short circuit if no boundaries are given (the boundaries are +-inf)
    if name in expression.boundaries_params:
        (bot, top) = expression.boundaries_params[name]

        # out of boundaries
        if (
            # only check values if bot/ top are not -+inf
            (not np.isinf(bot) and np.min(param_values) < bot)
            or (not np.isinf(top) and np.max(param_values) > top)
        ):
            return False

    return True


def _params_in_distr_support(expression: Expression, params, data):
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


def _validate_coefficients(
    expression: Expression, data_pred, data_targ, coefficients, threshold_min_proba
):
    """validate coefficients

    Parameters
    ----------
    expression: Expression
        Expression to validate the coefficients for.

    data_pred : numpy array 1D
        Predictors for the training sample.

    data_targ : numpy array 1D
        Target for the training sample.

    coefficients : numpy array 1D
        Coefficients to validate.

    threshold_min_proba : float | None
        Minimal probability of each data sample in the distribution.


    Returns
    -------

    coeffs_in_bounds : bool
        True if the coefficients are within conditional_distrib.expression.boundaries_coeffs. If
        False, all other tests will also be set to False and not tested.

    params_in_bounds : bool
        True if the params are within conditional_distrib.expression.boundaries_params

    params_in_support : bool
        True if parameters are within conditional_distrib.expression.boundaries_params and within the support of the distribution.
        False if not or if test_coeff is False. If False, test_proba will be set to False and not tested.

    test_proba : bool
        Only tested if conditional_distrib.threshold_min_proba is not None.
        True if the probability of the target samples for the given coefficients
        is above conditional_distrib.threshold_min_proba.
        False if not or if test_coeff or test_param or test_coeff is False.

    params : dict
        The evaluated params for the given coefficients, if any of the tests fail, empty dict.

    """
    params = {}

    coeffs_in_bounds = _coeffs_in_bounds(expression, coefficients)
    # tests on coeffs show already that it won't work: fill in the rest with False
    if not coeffs_in_bounds:
        return coeffs_in_bounds, False, False, False, params

    # evaluate the distribution for the predictors and this iteration of coeffs
    params = expression._evaluate_params_fast(coefficients, data_pred)

    # test for the validity of the parameters
    params_in_bounds = _params_in_bounds(expression, params)
    # tests on params show that it won't work: fill in the rest with False
    if not params_in_bounds:
        return coeffs_in_bounds, params_in_bounds, False, False, params

    # test for the support of the distribution
    params_in_support = _params_in_distr_support(expression, params, data_targ)
    # tests on params show that it won't work: fill in the rest with False
    if not params_in_support:
        return coeffs_in_bounds, params_in_bounds, params_in_support, False, params

    # test for the probability of the values
    if threshold_min_proba is None:
        return coeffs_in_bounds, params_in_bounds, params_in_support, True, params

    test_proba = _test_proba_value(expression, threshold_min_proba, params, data_targ)
    # return values for each test and the evaluated distribution
    return coeffs_in_bounds, params_in_bounds, params_in_support, test_proba, params


def _check_no_nan_no_inf(data, name):
    """
    check data for nans or infs
    """

    # checking for NaN values
    if np.isnan(data).any():
        raise ValueError(f"nan values in {name}")

    # checking for infinite values
    if np.isinf(data).any():
        raise ValueError(f"infinite values in {name}")
