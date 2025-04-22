# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Refactored code for the training of distributions

"""

import math

import numpy as np
import scipy as sp
import xarray as xr

from mesmer.core.datatree import (
    collapse_datatree_into_dataset,
)

_DISCRETE_DISTRIBUTIONS = sp.stats._discrete_distns._distn_names
_CONTINUOUS_DISTRIBUTIONS = sp.stats._continuous_distns._distn_names

_ALL_DISTRIBUTIONS = _DISCRETE_DISTRIBUTIONS + _CONTINUOUS_DISTRIBUTIONS

_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class Expression:

    def __init__(self, expr, expr_name):
        """conditional distribution: definition, interpretation and evaluation

        When initialized, the class identifies the distribution, inputs, parameters and
        coefficients. Then, the next function would be evaluate(coefficients, inputs,
        forced_shape) to assess the distribution given the provided values of
        coefficients & inputs. NB language: the distribution depends on parameters
        (loc, scale, etc), that are functions of inputs and coefficients.

        Parameters
        ----------
        expr : str
            string describing the expression that will be used. Existing methods for
            flexible conditional distributions don't provide the level of flexibility
            that MESMER-X uses.
            Requirements to follow to have the expression being understood:

            - distribution: must be the name of a distribution in scipy stats (discrete
              or continuous): https://docs.scipy.org/doc/scipy/reference/stats.html

            - parameters: all names for the parameters of the distributions must be
              provided: loc, scale, shape, mu, a, b, c, etc.

            - coefficients: must be named "c#", with # being the number of the
              coefficient: e.g. c1, c2, c3.

            - inputs: any string that can be written as a variable in python, and
              surrounded with "__": e.g. __GMT__, __X1__, __gmt_tm1__, but NOT
              __gmt-1__, __#days__, __GMT, _GMT_ etc.

            - equations: the equations for the evolutions must be written as it would be
              normally. Names of packages should be included.
              spaces dont matter in the equations

        expr_name : str
            Name of the expression.

        Notes
        -----
        - Forcing values of certain parameters (eg scale) to be positive: to implement
          in class 'expression'?
        - Forcing values of certain parameters (eg mu for poisson) to be integers: to
          implement in class 'expression'?
        - Reasons for not using sympy:
          - sympy.stats does not have all required distributions (e.g. GEV) & all
            required functions (e.g. log-likelihood)
            -> so using scipy distributions with parameters as sympy expressions
          - but these expressions dont deal well with numpy array, or even less with
            xarray -> could, but tedious.
          - And the result would be significantly slower than with numpy/xarray based
            approaches like here.

        Examples
        --------
        - "genextreme(loc=c1 + c2 * __pred1__, scale=c3 + c4 * __pred2__**2, c=c5)"
        - "norm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __GMT_t__ + c4 * __GMT_tm1__ - c5) ), scale=c6)"
        - "exponpow(loc=c1, scale=c2+np.min([np.max(np.mean([__GMT_tm1__,__GMT_tp1__],axis=0)), math.gamma(__XYZ__)]), b=c3)"
        """

        # basic initialization
        self.expression = expr
        self.expression_name = expr_name

        # identify distribution
        self._interpret_distrib()

        # identify parameters
        self._find_parameters_list()
        self._find_expr_parameters()

        # identify coefficients
        self._find_coefficients()

        # identify inputs
        self._find_inputs()

        # correct expressions of parameters
        self._correct_expr_parameters()

        # compile expression for faster eval
        self._compile_expression()

    def _interpret_distrib(self):
        """interpreting the expression"""

        dist = str.split(self.expression, "(")[0]

        if dist not in _ALL_DISTRIBUTIONS:
            raise AttributeError(
                f"Could not find distribution '{dist}'."
                " Please provide a distribution written as in scipy.stats:"
                " https://docs.scipy.org/doc/scipy/reference/stats.html"
            )

        self.distrib = getattr(sp.stats, dist)
        self.is_distrib_discrete = dist in _DISCRETE_DISTRIBUTIONS

    def _find_expr_parameters(self):

        # removing spaces that would hinder the identification
        tmp_expression = self.expression.replace(" ", "")

        # removing distribution part
        tmp_expression = "(".join(str.split(tmp_expression, "(")[1:])
        tmp_expression = ")".join(str.split(tmp_expression, ")")[:-1])

        # identifying groups
        sub_expressions = str.split(tmp_expression, ",")

        self.parameters_expressions = {}
        for sub_exp in sub_expressions:
            param, sub = str.split(sub_exp, "=")
            if param in self.parameters_list:
                self.parameters_expressions[param] = sub
            else:
                raise ValueError(
                    f"The parameter '{param}' is not part of prepared expression in scipy.stats:"
                    " https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats."
                    f" of the distribution '{self.distrib.name}'"
                )

        # recommend not to try to fill in missing information on parameters with
        # constant parameters, but to raise a ValueError instead.
        for pot_param in self.parameters_list:
            if pot_param not in self.parameters_expressions.keys():
                raise ValueError(f"No information provided for `{pot_param}`")

    def _find_parameters_list(self):
        """
        List parameters for scipy.stats.distribution.
            distribution: a string or scipy.stats distribution object.

        Returns
            A list of distribution parameter strings.

        Based on the code available at: https://github.com/scipy/scipy/issues/9575
        """

        self.parameters_list = []

        if self.distrib.shapes:
            self.parameters_list += [
                name.strip() for name in self.distrib.shapes.split(",")
            ]

        if self.distrib.name in _DISCRETE_DISTRIBUTIONS:
            self.parameters_list += ["loc"]
        elif self.distrib.name in _CONTINUOUS_DISTRIBUTIONS:
            self.parameters_list += ["loc", "scale"]

        # prepary basic boundaries on parameters: incomplete, did not find a way to
        # evaluate automatically the limits on shape parameters

        self.boundaries_parameters = {
            p: [-np.inf, np.inf] for p in self.parameters_list
        }

        # scale must be positive
        if "scale" in self.boundaries_parameters:
            self.boundaries_parameters["scale"][0] = 0

    def _find_coefficients(self):
        """
        coefficients are supposed to be written as "c#", with "#" being a number.
        """

        self.coefficients_list, self.coefficients_dict = [], {}

        for param in self.parameters_expressions:
            self.coefficients_dict[param] = []
            # iniatilize detection
            cf = ""

            # adding one space at the end to append the last coefficient
            for pep in self.parameters_expressions[param] + " ":

                if pep == "c":
                    # starting expression for a coefficient
                    cf = "c"
                elif (cf == "c") and (pep in _DIGITS):
                    # continuing expression for a coefficient
                    cf += pep
                else:
                    if cf not in ["", "c"]:

                        # ending expression for a coefficient
                        if cf not in self.coefficients_list:
                            self.coefficients_list.append(cf)
                            self.coefficients_dict[param].append(cf)

                        cf = ""
                    else:
                        # not a coefficient, move to the next
                        pass

    def _find_inputs(self):

        self.inputs_list = []

        for param in self.parameters_expressions:
            terms = str.split(self.parameters_expressions[param], "__")
            for i in np.array(terms)[np.arange(1, len(terms), 2)]:
                if i not in self.inputs_list:
                    self.inputs_list.append(i)

        # require a specific order for use in correct_expr_parameters
        self.inputs_list.sort(key=len, reverse=True)

    def _correct_expr_parameters(self):
        # list of inputs and coefficients, sorted by descending length, to be sure that
        # when removing them, will remove the good ones and not those with their short
        # name contained in the long name of another

        tmp_list = self.inputs_list + self.coefficients_list + ["__"]
        tmp_list.sort(key=len, reverse=True)

        # making sure that the expression will be understood
        for param in self.parameters_expressions:

            # 1. checking the names of packages
            expr = self.parameters_expressions[param]

            # removing inputs and coefficients
            for x in tmp_list:
                expr = expr.replace(x, "")

            # reading this condensed expression to find terms to replace
            t = ""  # initialization

            # adding one space at the end to treat the last term
            for ex in expr + " ":

                # TODO: could do this using regular expressions
                if ex in ["(", ")", "[", "]", "+", "-", "*", "/", "%"] + _DIGITS:
                    # this is a readable character that will not be an issue to read
                    if t not in ["", " "]:
                        # a term is here, and needs to be confirmed as readable
                        if t.startswith("np."):
                            if t[len("np.") :] not in vars(np):
                                raise ValueError(
                                    f"Proposed a numpy function that does not exist: '{t}'"
                                )
                            else:
                                # nothing to replace, can go with that
                                pass
                        elif t.startswith("math."):
                            if t[len("math.") :] not in vars(math):
                                raise ValueError(
                                    f"Proposed a math function that does not exist: '{t}'"
                                )
                            else:
                                # nothing to replace, can go with that
                                pass
                        else:
                            raise ValueError(
                                f"Unknown function '{t}' in expression"
                                f" '{self.parameters_expressions[param]}' for '{param}'."
                                "Do you need to prepend it with 'np.' (or 'math.')?"
                                " Currently only numpy and math functions are supported."
                            )
                    else:
                        # was a readable character, is still a readable character,
                        # nothing to do
                        pass
                else:
                    t += ex

                # TODO: would make sense to invert the logic here - move building 't'
                # above the validity check

            # replacing names of inputs in expressions
            for i in self.inputs_list:
                self.parameters_expressions[param] = self.parameters_expressions[
                    param
                ].replace(f"__{i}__", i)

    def _compile_expression(self):
        """compile expression for faster eval"""

        self._compiled_param_expr = {
            param: compile(expr, param, "eval")
            for param, expr in self.parameters_expressions.items()
        }

    def evaluate_params(self, coefficients_values, inputs_values, forced_shape=None):
        """
        Evaluates the parameters for the provided inputs and coefficients

        Parameters
        ----------
        coefficients_values : dict | xr.Dataset(c_i) | list of values
            Coefficient arrays or scalars. Can have the following form
            - dict(c_i = values or np.array())
            - xr.Dataset(c_i)
            - list of values
        inputs_values : dict | xr.Dataset
            Input arrays or scalars. Can be passed as
            - dict(inp_i = values or np.array())
            - xr.Dataset(inp_i)
        forced_shape : None | tuple or list of dimensions
            coefficients_values and inputs_values for transposition of the shape.
            Can include additional axes like 'realization'.

        Returns
        -------
        params: dict
            Realized parameters for the given expression, coefficients and covariates;
            to pass ``self.distrib(**params)`` or its methods.

        Warnings
        --------
        with xarrays for coefficients_values and inputs_values, the outputs will have
        for shape first the one of the coefficient, then the one of the inputs
        --> trying to avoid this issue with 'forced_shape'
        """

        # TODO:
        # - use broadcasting
        # - can we avoid using exec & eval?
        # - only parse the values once? (to avoid doing it repeatedly)
        # - require list of coefficients_values (similar to minimize)?
        # - convert dataset to numpy arrays?

        # Check 1: are all the coefficients provided?
        if isinstance(coefficients_values, dict | xr.Dataset):
            # case where provide explicit information on coefficients_values
            for c in self.coefficients_list:
                if c not in coefficients_values:
                    raise ValueError(f"Missing information for the coefficient: '{c}'")
        else:
            # case where a vector is provided, used for the optimization performed
            # during the training
            if len(coefficients_values) != len(self.coefficients_list):
                raise ValueError("Inconsistent information for the coefficients_values")

            coefficients_values = {
                c: coefficients_values[i] for i, c in enumerate(self.coefficients_list)
            }

        # Check 2: are all the inputs provided?
        for i in self.inputs_list:
            if i not in inputs_values:
                raise ValueError(f"Missing information for the input: '{i}'")

        # Check 3: do the inputs have the same shape
        shapes = {inputs_values[i].shape for i in self.inputs_list}
        if len(shapes) > 1:
            raise ValueError("shapes of inputs must be equal")

        # gather coefficients and covariates (can't use d1 | d2, does not work for dataset)
        locals = {**coefficients_values, **inputs_values}

        # evaluate parameters
        parameters_values = {}
        for param, expr in self._compiled_param_expr.items():
            parameters_values[param] = eval(expr, None, locals)

            # if constant parameter but varying inputs, need to broadcast
            if (
                isinstance(coefficients_values, xr.Dataset)
                and len(self.inputs_list) > 0
                and parameters_values[param].ndim == 1
            ):
                parameters_values[param], _ = xr.broadcast(
                    parameters_values[param], inputs_values
                )

        # forcing the shape of the parameters if necessary
        if forced_shape is not None and len(self.inputs_list) > 0:
            for param in self.parameters_list:
                # Add missing dimensions in forced_shape (e.g., 'realization')
                for dim in forced_shape:
                    if dim not in parameters_values[param].dims:
                        parameters_values[param] = parameters_values[param].expand_dims(
                            dim=dim, axis=0
                        )

                # Transpose the parameters to match the forced shape
                dims_param = [
                    d for d in forced_shape if d in parameters_values[param].dims
                ]
                parameters_values[param] = parameters_values[param].transpose(
                    *dims_param
                )

        return parameters_values

    def evaluate(self, coefficients_values, inputs_values, forced_shape=None):
        """
        Evaluates the distribution with the provided inputs and coefficients

        Parameters
        ----------
        coefficients_values : dict | xr.Dataset(c_i) | list of values
            Coefficient arrays or scalars. Can have the following form
            - dict(c_i = values or np.array())
            - xr.Dataset(c_i)
            - list of values
        inputs_values : dict | xr.Dataset
            Input arrays or scalars. Can be passed as
            - dict(inp_i = values or np.array())
            - xr.Dataset(inp_i)
        forced_shape : None | tuple or list of dimensions
            coefficients_values and inputs_values for transposition of the shape

        Returns
        -------
        distr: scipy stats frozen distribution
            Frozen distribution with the realized parameters applied to.

        Warnings
        --------
        with xarrays for coefficients_values and inputs_values, the outputs with have
        for shape first the one of the coefficient, then the one of the inputs
        --> trying to avoid this issue with 'forced_shape'
        """

        params = self.evaluate_params(coefficients_values, inputs_values, forced_shape)
        return self.distrib(**params)


class probability_integral_transform:
    def __init__(
        self,
        expr_start,
        expr_end,
        coeffs_start=None,
        coeffs_end=None,
    ):
        """
        Probability integral transform of the data given parameters of a given distribution
        into their equivalent in a standard normal distribution.

        Parameters
        ----------
        expr_start : str
            string describing the starting expression
        expr_end : str
            string describing the starting expression
        coeffs_start : xarray dataset
            Coefficients of the starting expression. Default: None.
        coeffs_end : xarray dataset
            Coefficients of the ending expression. Default: None.
        """
        # preparation of distributions
        self.expression_start = Expression(expr_start, "start")
        self.expression_end = Expression(expr_end, "end")

        # preparation of coefficients
        self.coeffs_start = self.prepare_coefficients(coeffs_start)
        self.coeffs_end = self.prepare_coefficients(coeffs_end)

    def prepare_coefficients(self, coeffs):
        """
        Prepare coefficients for the expression
        """
        if coeffs is None:
            # creating an empty dataset for use
            return xr.Dataset()

        elif isinstance(coeffs, xr.Dataset):
            # no need to correct the format
            return coeffs

        else:
            raise Exception("coefficients must be a xarray Dataset or None")

    def transform(
        self,
        data,
        target_name,
        preds_start=None,
        preds_end=None,
        threshold_proba=1.0e-9,
    ):
        """
        Probability integral transform of the data given parameters of a given distribution
        into their equivalent in a standard normal distribution.

        Parameters
        ----------
        data : stacked Datatree
            Data to transform. xarray Dataset with coordinate 'gridpoint'.
        target_name : str
            name of the variable to train
        preds_start : stacked Datatree | None
            Covariants of the starting expression. Default: None.
        preds_end : stacked Datatree | None
            Covariants of the ending expression. Default: None.
        threshold_proba : float, default: 1.e-9.
            Threshold for the probability of the sample on the starting distribution.
            Applied both to the lower and upper bounds of the distribution.
        Returns
        -------
        transf_inputs : not sure yet what data format will be used at the end.
            Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint' and one
            2D variable with both coordinates
        """
        if isinstance(data, xr.DataTree):
            # check on similar format
            if isinstance(preds_start, xr.Dataset) or isinstance(preds_end, xr.Dataset):
                raise Exception("predictors must have the same format as data")

            # looping over scenarios of data
            out = dict()
            for scen, data_scen in data.items():
                # preparing data to transform
                data_scen = data_scen.to_dataset()

                # preparing predictors
                ds_pred_start = self.prepare_predictors(preds_start, scen=scen)
                ds_preds_end = self.prepare_predictors(preds_end, scen=scen)

                # transforming data
                tmp = self._transform(
                    data_scen, target_name, ds_pred_start, ds_preds_end, threshold_proba
                )

                # creating transf_data as a dataset with tmp as a variable
                out[scen] = xr.Dataset(
                    {target_name: (data_scen[target_name].dims, tmp)},
                    coords=data_scen[target_name].coords,
                )

            # creating datatree
            return xr.DataTree.from_dict(out)

        elif isinstance(data, xr.Dataset):
            # check on similar format
            if isinstance(preds_start, xr.DataTree) or isinstance(
                preds_end, xr.DataTree
            ):
                raise Exception("predictors must have the same format as data")

            # preparing predictors
            ds_pred_start = self.prepare_predictors(preds_start)
            ds_preds_end = self.prepare_predictors(preds_end)

            # transforming data
            tmp = self._transform(
                data, target_name, ds_pred_start, ds_preds_end, threshold_proba
            )

            # creating transf_data as a dataset with tmp as a variable
            transf_data = xr.Dataset(
                {target_name: (data[target_name].dims, tmp)},
                coords=data[target_name].coords,
            )
            return transf_data

        else:
            raise Exception("data must be a xarray Dataset or Datatree")

    def prepare_predictors(self, preds, scen=None):
        # preparation of predictors
        if preds is None:
            ds_preds = xr.Dataset()

        elif isinstance(preds, xr.DataTree):
            if scen is not None:
                # taking only the correct scenario, while keeping the same format
                preds = xr.DataTree.from_dict({p: preds[p][scen] for p in preds})

            # correcting format: must be dict(str, DataArray or array) for Expression
            tmp = collapse_datatree_into_dataset(preds, dim="predictor")
            var_name = [var for var in tmp.variables][0]
            ds_preds = xr.Dataset()
            for pp in tmp["predictor"].values:
                ds_preds[pp] = tmp[var_name].sel(predictor=pp)

        elif isinstance(preds, xr.Dataset):
            # no need to correct format
            ds_preds = preds

        return ds_preds

    def _transform(
        self, data, target_name, ds_pred_start, ds_preds_end, threshold_proba
    ):
        # parameters of starting distribution
        params_start = self.expression_start.evaluate_params(
            self.coeffs_start, ds_pred_start, forced_shape=data[target_name].dims
        )

        # probabilities of the sample on the starting distribution
        cdf_data = self.expression_start.distrib.cdf(data[target_name], **params_start)

        # avoiding very unlikely values
        cdf_data[np.where(cdf_data < threshold_proba)] = threshold_proba
        cdf_data[np.where(cdf_data > 1 - threshold_proba)] = 1 - threshold_proba

        # parameters of ending distribution
        params_end = self.expression_end.evaluate_params(
            self.coeffs_end, ds_preds_end, forced_shape=data[target_name].dims
        )

        # values corresponding to probabilities of sample on the ending distribution
        return self.expression_end.distrib.ppf(cdf_data, **params_end)


def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights

    Source:
      https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
      @author Jack Peterson (jack@tinybike.net)
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)

    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median
