# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr
from mesmer.core.datatree import collapse_datatree_into_dataset
from mesmer.mesmer_x._expression import Expression

class distrib_tests:
    def __init__(
        self,
        expr_fit: Expression,
        threshold_min_proba=1.0e-9,
        boundaries_params=None,
        boundaries_coeffs=None,
    ):
        """Class defining the tests to perform during first guess and training of distributions.

        Parameters
        ----------
        expr_fit : class 'expression'
            Expression to train. The string provided to the class can be found in
            'expr_fit.expression'.

        threshold_min_proba : float or None, default: 1e-9
            If numeric imposes a check during the fitting that every sample fulfills
            `cdf(sample) >= threshold_min_proba and 1 - cdf(sample) >= threshold_min_proba`,
            i.e. each sample lies within some confidence interval of the distribution.
            Note that it follows that threshold_min_proba math::\\in (0,0.5). Important to
            ensure that all points are feasible with the fitted distribution.
            If `None` this test is skipped.

        boundaries_params : dict, default: None
            Prescribed boundaries on the parameters of the expression. Some basic
            boundaries are already provided through 'expr_fit.boundaries_params'.

        boundaries_coeffs : dict, optional
            Prescribed boundaries on the coefficients of the expression. Default: None.

        """
        # initialization of expr_fit
        self.expr_fit = expr_fit

        # initialization and basic checks on threshold_min_proba
        self.threshold_min_proba = threshold_min_proba
        if threshold_min_proba is not None and (
            (threshold_min_proba <= 0) or (0.5 <= threshold_min_proba)
        ):
            raise ValueError("`threshold_min_proba` must be in (0, 0.5)")

        # initialization and basic checks on boundaries
        self.boundaries_params = self.expr_fit.boundaries_parameters
        if boundaries_params is not None:
            for param in boundaries_params:
                lower_bound = np.max(
                    [boundaries_params[param][0], self.boundaries_params[param][0]]
                )
                upper_bound = np.min(
                    [boundaries_params[param][1], self.boundaries_params[param][1]]
                )
                self.boundaries_params[param] = [lower_bound, upper_bound]
        self.boundaries_coeffs = {} if boundaries_coeffs is None else boundaries_coeffs

    def _test_coeffs_in_bounds(self, values_coeffs):

        # checking set boundaries on coefficients
        for coeff in self.boundaries_coeffs:
            bottom, top = self.boundaries_coeffs[coeff]

            # TODO: move this check to __init__ NOTE: also used in fg
            if coeff not in self.expr_fit.coefficients_list:
                raise ValueError(
                    f"Provided wrong boundaries on coefficient, {coeff}"
                    " does not exist in expr_fit"
                )

            values = values_coeffs[self.expr_fit.coefficients_list.index(coeff)]

            if np.any(values < bottom) or np.any(top < values):
                # out of boundaries
                return False

        return True

    def _test_evol_params(self, params):

        # checking set boundaries on parameters
        for param in self.boundaries_params:
            bottom, top = self.boundaries_params[param]

            param_values = params[param]

            # out of boundaries
            if np.any(param_values < bottom) or np.any(param_values > top):
                return False

        return True

    def _test_support(self, params, data):
        # test of the support of the distribution: is there any data out of the
        # corresponding support? dont try testing if there are issues on the parameters

        bottom, top = self.expr_fit.distrib.support(**params)

        # out of support
        if (
            np.any(np.isnan(bottom))
            or np.any(np.isnan(top))
            or np.any(data < bottom)
            or np.any(data > top)
        ):
            return False

        return True

    def _test_proba_value(self, params, data):
        """
        Test that all cdf(data) >= threshold_min_proba and 1 - cdf(data) >= threshold_min_proba
        Ensures that data lies within a confidence interval of threshold_min_proba for the tested
        distribution.
        """
        # NOTE: DONT write 'x=data', because 'x' may be called differently for some
        # distribution (eg 'k' for poisson).

        cdf = self.expr_fit.distrib.cdf(data, **params)
        thresh = self.threshold_min_proba
        return np.all(1 - cdf >= thresh) and np.all(cdf >= thresh)

    def validate_coefficients(self, data_pred, data_targ, coefficients):
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
            True if the coefficients are within self.boundaries_coeffs. If
            False, all other tests will also be set to False and not tested.

        test_param : boolean
            True if parameters are within self.boundaries_params and within the support of the distribution.
            False if not or if test_coeff is False. If False, test_proba will be set to False and not tested.

        test_proba : boolean
            Only tested if self.threshold_min_proba is not None.
            True if the probability of the target samples for the given coefficients
            is above self.threshold_min_proba.
            False if not or if test_coeff or test_param or test_coeff is False.

        distrib : distrib_cov
            The distribution that has been evaluated for the given coefficients.

        """

        test_coeff = self._test_coeffs_in_bounds(coefficients)

        # tests on coeffs show already that it won't work: fill in the rest with False
        if not test_coeff:
            return test_coeff, False, False, False, False

        # evaluate the distribution for the predictors and this iteration of coeffs
        params = self.expr_fit.evaluate_params(coefficients, data_pred)
        # test for the validity of the parameters
        test_param = self._test_evol_params(params)

        # tests on params show already that it won't work: fill in the rest with False
        if not test_param:
            return test_coeff, test_param, False, False, False

        # test for the support of the distribution
        test_support = self._test_support(params, data_targ)

        # tests on params show already that it won't work: fill in the rest with False
        if not test_support:
            return test_coeff, test_param, test_support, False, False

        # test for the probability of the values
        if self.threshold_min_proba is None:
            return test_coeff, test_param, test_support, True, params

        else:
            test_proba = self._test_proba_value(params, data_targ)

            # return values for each test and the evaluated distribution
            return test_coeff, test_param, test_support, test_proba, params

    def get_var_data(self, data):
        if isinstance(data, np.ndarray):
            return data

        elif isinstance(data, xr.DataArray):
            return data

        elif isinstance(data, xr.Dataset):
            var_name = [var for var in data.variables][0]
            return data[var_name]

        elif isinstance(data, xr.DataTree):
            # TODO: useless, datatree uses datasets anyway, so it will become a dataarray
            new_data = xr.DataTree()
            for pred in data:
                var_name = [var for var in data[pred].variables][0]
                _ = xr.DataTree(name=pred, parent=new_data, data=data[pred][var_name])
            return new_data

        else:
            raise ValueError("data must be a DataArray, Dataset or DataTree")

    def validate_data(self, data_pred, data_targ, data_weights):
        """validate data

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
        # basic checks on data_targ
        self.check_data(data_targ, "target")

        # basic checks on data_pred
        self.check_data(data_pred, "predictors")

        # basic checks on weights
        self.check_data(data_weights, "weights")

    def check_data(self, data, name):
        """
        basic check data
        """
        # getting variable. useful if calling _fit_np | _find_fg_np with wrong format
        data = self.get_var_data(data)

        # checking for NaN values
        if np.isnan(data).any():
            raise ValueError(f"nan values in {name}")

        # checking for infinite values
        if np.isinf(data).any():
            raise ValueError(f"infinite values in {name}")

    def prepare_data(self, predictors, target, weights):
        """
        shaping data for first guess, training or evaluation of scores.

        Parameters
        ----------
        predictors : dict of xr.DataArray or xr.Dataset | xr.Dataset | xr.DataTree
            Predictors for the first guess. Must either be a dictionary of xr.DataArray or
            xr.Dataset, each key/item being a predictor; a xr.Dataset with a coordinate
            being the list of predictors, and a variable that contains all predictors; or
            a xr.DataTree with one branch per predictor.
        target : xr.DataArray | xr.Dataset
            Target DataArray.
        weights : xr.DataArray | xr.Dataset
            Individual weights for each sample.

        Returns
        -------
        :data_pred:`xr.Dataset`
            shaped predictors for training (gridpoint, coefficient)
        :data_targ:`xr.Dataset`
            shaped sample for training (gridpoint, coefficient)
        :data_weights:`xr.Dataset`
            shaped weights for training (gridpoint, coefficient)
        """
        # check format of predictors
        if isinstance(predictors, dict):
            tmp = {
                key: self.class_tests.get_var_data(predictors[key])
                for key in predictors
            }
            ds_pred = xr.Dataset(tmp)

        elif isinstance(predictors, xr.Dataset):
            if "predictor" not in predictors.coords:
                raise Exception(
                    "If predictors are provided as xr.Dataset, it must contain a coordinate 'predictor'."
                )

        elif isinstance(predictors, xr.DataTree):
            # preparing predictors
            ds_pred = collapse_datatree_into_dataset(predictors, dim="predictor")

        else:
            raise Exception(
                "predictors is supposed to be a dict of xr.DataArray, xr.Dataset or xr.DataTree"
            )

        # check format of target
        if not (isinstance(target, xr.Dataset) or isinstance(target, xr.DataArray)):
            raise Exception("the target must be a xr.Dataset or xr.DataArray.")

        # check format of weights
        if not (isinstance(weights, xr.Dataset) or isinstance(weights, xr.DataArray)):
            raise Exception("the weights must be a xr.Dataset or xr.DataArray.")

        # getting just dataarray in the datasets
        data_pred = self.get_var_data(ds_pred)
        data_targ = self.get_var_data(target)
        data_weights = self.get_var_data(weights)

        return data_pred, data_targ, data_weights
