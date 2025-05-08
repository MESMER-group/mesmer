# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import math
import warnings

import numpy as np
import scipy as sp
import xarray as xr

_DISCRETE_DISTRIBUTIONS = sp.stats._discrete_distns._distn_names
_CONTINUOUS_DISTRIBUTIONS = sp.stats._continuous_distns._distn_names

_ALL_DISTRIBUTIONS = _DISCRETE_DISTRIBUTIONS + _CONTINUOUS_DISTRIBUTIONS

_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class Expression:

    def __init__(
        self,
        expr: str,
        expr_name: str,
        boundaries_params: dict = {},
        boundaries_coeffs: dict = {},
    ):
        """Symbolic expression of a conditional distribution.

        When initialized, the class identifies the distribution, predictors, parameters and
        coefficients. The expression is then compiled so that it can be evaluated.

        Parameters
        ----------
        expr : str
            Mathematical expression of the conditional distribution as a string.

        expr_name : str
            Name for the expression.

        boundaries_params : dict
            Boundaries for the parameters. The keys are the names of the parameters, and
            the values are lists of two elements, the lower and upper bounds. The default
            is an empty dict, which will set the boundaries to [-inf, inf] for all parameters
            found in the expression, except `scale` which must be positive, so the lower boundary
            will be set to 0. These boundaries will later be enforced when fitting the conditional
            distribution.

        boundaries_coeffs : dict
            Boundaries for the coefficients. The keys are the names of the coefficients as
            they are written in the expression, and the values are lists of two elements,
            the lower and upper bounds. The default is an empty dict. These boundaries will later
            be enforced when fitting the conditional distribution.

        Notes
        -----
        A valid expression of a distribution contains the following elements:

            - a distribution: must be the name of a distribution in `scipy.stats` (discrete
              or continuous): https://docs.scipy.org/doc/scipy/reference/stats.html

            - the parameters: all names for the parameters of the distributions must be
              provided (as named in the `scipy.stats` distribution): loc, scale, shape,
              mu, a, b, c, etc.

            - predictors: predictors that the distribution will be conditional to, must
              be written like a variable in python and surrounded by "__":
              e.g. __GMT__, __X1__, __gmt_tm1__, but NOT __gmt-1__, __#days__, __GMT, _GMT_ etc.

            - coefficients: coefficients of the predictors, must be named "c#", with # being
              the number of the coefficient: e.g. c1, c2, c3.

            - mathematical terms: mathematical terms for the evolutions are be written as they
              would be normally in python. Names of packages should be included. Spaces do not
              matter in the equations

        .. warning::
            Currently, the expression can only contain integers as numbers, no floats!


        Examples
        --------
        `my_expr = Expression("genextreme(loc=c1 + c2 * __pred1__, scale=c3 + c4 * __pred2__**2, c=c5", "my_expr1")`
        `my_expr = Expression("norm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __GMT_t__ + c4 * __GMT_tm1__ - c5) ), scale=c6)", "my_expr2")`
        `my_expr = Expression("exponpow(loc=c1, scale=c2+np.min([np.max(np.mean([__GMT_tm1__,__GMT_tp1__],axis=0)), math.gamma(__XYZ__)]), b=c3), "my_expr3")`
        """
        # TODO: Forcing values of certain parameters (eg mu for poisson) to be integers?

        # basic initialization
        self.expression = expr
        self.expression_name = expr_name
        self.boundaries_params = boundaries_params
        self.boundaries_coeffs = boundaries_coeffs

        # identify distribution
        self._interpret_distrib()

        # identify parameters
        self._find_parameters_list()
        self._find_expr_parameters()

        # identify coefficients
        self._find_coefficients()

        # identify predictors
        self._find_predictors()

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

        # prepare boundaries on parameters
        # default is [-inf, inf] except for scale which must be positive
        for param in self.parameters_list:
            if param not in self.boundaries_params:
                if param == "scale":
                    self.boundaries_params[param] = [0, np.inf]
                else:
                    self.boundaries_params[param] = [-np.inf, np.inf]

            if (
                "scale" in self.boundaries_params
                and self.boundaries_params["scale"][0] < 0
            ):
                warnings.warn(
                    "Found lower boundary on scale parameter that is negative, setting to 0."
                )
                self.boundaries_params["scale"][0] = 0

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

        self.n_coeffs = len(self.coefficients_list)

        # save coefficient indices
        loc_coeffs = self.coefficients_dict.get("loc", [])
        self.ind_loc_coeffs = np.array(
            [self.coefficients_list.index(c) for c in loc_coeffs]
        )

        scale_coeffs = self.coefficients_dict.get("scale", [])
        self.ind_sca_coeffs = np.array(
            [self.coefficients_list.index(c) for c in scale_coeffs]
        )

        other_params = [p for p in self.parameters_list if p not in ["loc", "scale"]]
        if len(other_params) > 0:
            fg_ind_others = []
            for param in other_params:
                for c in self.coefficients_dict[param]:
                    fg_ind_others.append(self.coefficients_list.index(c))

            self.ind_others = np.array(fg_ind_others)
        else:
            self.ind_others = np.array([])

    def _find_predictors(self):

        self.predictors_list = []

        for param in self.parameters_expressions:
            terms = str.split(self.parameters_expressions[param], "__")
            for i in np.array(terms)[np.arange(1, len(terms), 2)]:
                if i not in self.predictors_list:
                    self.predictors_list.append(i)

        # require a specific order for use in correct_expr_parameters
        self.predictors_list.sort(key=len, reverse=True)

    def _correct_expr_parameters(self):
        # list of predictors and coefficients, sorted by descending length, to be sure that
        # when removing them, will remove the good ones and not those with their short
        # name contained in the long name of another

        tmp_list = self.predictors_list + self.coefficients_list + ["__"]
        tmp_list.sort(key=len, reverse=True)

        # making sure that the expression will be understood
        for param in self.parameters_expressions:

            # 1. checking the names of packages
            expr = self.parameters_expressions[param]

            # removing predictors and coefficients
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

            # replacing names of predictors in expressions
            for i in self.predictors_list:
                self.parameters_expressions[param] = self.parameters_expressions[
                    param
                ].replace(f"__{i}__", i)

    def _compile_expression(self):
        """compile expression for faster eval"""

        self._compiled_param_expr = {
            param: compile(expr, param, "eval")
            for param, expr in self.parameters_expressions.items()
        }

    def evaluate_params(
        self, coefficients_values, predictors_values, forced_shape=None
    ):
        """
        Evaluates the parameters for the provided predictors and coefficients

        Parameters
        ----------
        coefficients_values : dict | xr.Dataset(c_i) | list of values
            Coefficient arrays or scalars. Can have the following form
            - dict(c_i = values or np.array())
            - xr.Dataset(c_i)
            - list of values
        predictors_values : dict | xr.Dataset
            Input arrays or scalars. Can be passed as
            - dict(pred_i = values or np.array())
            - xr.Dataset(pred_i)
        forced_shape : None | tuple or list of dimensions
            coefficients_values and predictors_values for transposition of the shape.
            Can include additional axes like 'realization'.

        Returns
        -------
        params: dict
            Realized parameters for the given expression, coefficients and covariates;
            to pass ``self.distrib(**params)`` or its methods.

        Warnings
        --------
        with xarrays for coefficients_values and predictors_values, the outputs will have
        for shape first the one of the coefficient, then the one of the predictors
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

        # Check 2: are all the predictors provided?
        for pred in self.predictors_list:
            if pred not in predictors_values:
                raise ValueError(f"Missing information for the predictor: '{pred}'")

        # Check 3: do the predictors have the same shape
        shapes = {predictors_values[i].shape for i in self.predictors_list}
        if len(shapes) > 1:
            raise ValueError("shapes of predictors must be equal")

        # gather coefficients and covariates (can't use d1 | d2, does not work for dataset)
        locals = {**coefficients_values, **predictors_values}

        # evaluate parameters
        parameters_values = {}
        for param, expr in self._compiled_param_expr.items():
            parameters_values[param] = eval(expr, None, locals)

            # if constant parameter but varying predictors, need to broadcast
            if (
                isinstance(coefficients_values, xr.Dataset)
                and len(self.predictors_list) > 0
                and parameters_values[param].ndim == 1
            ):
                parameters_values[param], _ = xr.broadcast(
                    parameters_values[param], predictors_values
                )

        # forcing the shape of the parameters if necessary
        if forced_shape is not None and len(self.predictors_list) > 0:
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

    def evaluate(self, coefficients_values, predictors_values, forced_shape=None):
        """
        Evaluates the distribution with the provided predictors and coefficients

        Parameters
        ----------
        coefficients_values : dict | xr.Dataset(c_i) | list of values
            Coefficient arrays or scalars. Can have the following form
            - dict(c_i = values or np.array())
            - xr.Dataset(c_i)
            - list of values
        predictors_values : dict | xr.Dataset
            Input arrays or scalars. Can be passed as
            - dict(pred_i = values or np.array())
            - xr.Dataset(pred_i)
        forced_shape : None | tuple or list of dimensions
            coefficients_values and predictors_values for transposition of the shape

        Returns
        -------
        distr: scipy stats frozen distribution
            Frozen distribution with the realized parameters applied to.

        Warnings
        --------
        with xarrays for coefficients_values and predictors_values, the outputs will have
        the dimensions of the coefficients first, then the ones of the predictors
        """

        params = self.evaluate_params(
            coefficients_values, predictors_values, forced_shape
        )
        return self.distrib(**params)
