# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Refactored code for the training of distributions

"""

import math

import numpy as np
import scipy.stats as ss

# import sympy as sp
import xarray as xr


class Expression:
    """
        Class to interpret string of conditional distribution, interpret them, and evaluate them.
        When initialized, the class identifies the distribution, inputs, parameters and coefficients.
        Then, the next function would be evaluate(coefficients, inputs, forced_shape) to assess the distribution given the provided values of coefficients & inputs.
        NB language: the distribution depends on parameters (loc, scale, etc), that are functions of inputs and coefficients.

    Parameters:
    -------
        expr: str
            string describing the expression that will be used. Existing methods for flexible conditional distributions dont provide the level of flexibility that MESMER-X actually uses.
            Requirements to follow to have the expression being understood:
                - distribution: must be the name of a distribution in scipy stats (continuous or discrete): https://docs.scipy.org/doc/scipy/reference/stats.html
                - parameters: all names for the parameters of the distributions must be provided: loc, scale, shape, mu, a, b, c, etc.
                - coefficients: must be named "c#", with # being the number of the coefficient: e.g. c1, c2, c3.
                - inputs: any string that can be written as a variable in python, and surrounded with "__": e.g. __GMT__, __X1__, __gmt_tm1__, but NOT __gmt-1__, __#days__, __GMT, _GMT_ etc.
                - equations: the equations for the evolutions must be written as it would be normally. Names of packages should be included.
                spaces dont matter in the equiations
            Examples:
                - "genextreme(loc=c1 + c2 * __pred1__, scale=c3 + c4 * __pred2__**2, c=c5)"
                - "norm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __GMT_t__ + c4 * __GMT_tm1__ - c5) ), scale=c6)"
                - "exponpow(loc=c1, scale=c2+np.min([np.max(np.mean([__GMT_tm1__,__GMT_tp1__],axis=0)), math.gamma(__XYZ__)]), b=c3)"

        expr_name: str
            Name of the expression.

    Notes:
    -------
        - Forcing values of certain parameters (eg scale) to be positive: to implement in class 'expression'?
        - Forcing values of certain parameters (eg mu for poisson) to be integers: to implement in class 'expression'?
        - Reasons for not using sympy:
            - sympy.stats does not have all required distributions (e.g. GEV) & all required functions (e.g. log-likelihood)
            -> so using scipy distributions with parameters as sympy expressions
            - but these expressions dont deal well with numpy array, or even less with xarray
            -> could, but tedious.
            - And the result would be significantly slower than with numpy/xarray based approaches like here.
    """

    # --------------------
    # INITIALIZATION
    def __init__(self, expr, expr_name):
        # basic initalization
        self.expression = expr
        self.expression_name = expr_name

        # identify distribution
        self.interpret_distrib()

        # identify parameters
        self.find_parameters_list()
        self.find_expr_parameters()

        # identify coefficients
        self.find_coefficients()

        # identify inputs
        self.find_inputs()

        # correct expressions of parameters
        self.correct_expr_parameters()

    # --------------------

    # --------------------
    # INTERPRETING THE EXPRESSION
    def interpret_distrib(self):
        dist = str.split(self.expression, "(")[0]

        if (
            dist
            in ss._discrete_distns._distn_names + ss._continuous_distns._distn_names
        ):
            self.distrib = vars(ss.distributions)[dist]
            self.is_distrib_discrete = self.distrib in ss._discrete_distns._distn_names

        else:
            raise Exception(
                "Please provide a distribution written as in scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html"
            )

    def find_expr_parameters(self):
        # removing spaces that would hinder the identification
        tmp_expression = self.expression.replace(" ", "")

        # removing distribution part
        tmp_expression = "(".join(str.split(tmp_expression, "(")[1:])
        tmp_expression = ")".join(str.split(tmp_expression, ")")[:-1])

        # identifying groups
        sub_expressions = str.split(tmp_expression, ",")

        # creating dictionary
        self.parameters_expressions = {}
        for sub_exp in sub_expressions:
            param, sub = str.split(sub_exp, "=")
            if param in self.parameters_list:
                self.parameters_expressions[param] = sub
            else:
                raise Exception(
                    "The parameter "
                    + param
                    + " is not part of prepared expression in scipy.stats: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats."
                    + self.distrib.name
                )

        # recommend not to try to fill in missing information on parameters with constant parameters, but to raise an Exception instead.
        for pot_param in self.parameters_list:
            if pot_param not in self.parameters_expressions.keys():
                raise Exception("No information provided for " + pot_param)

    def find_parameters_list(self):
        """
        List parameters for scipy.stats.distribution.
            distribution: a string or scipy.stats distribution object.

        Returns
            A list of distribution parameter strings.

        Based on the code available at: https://github.com/scipy/scipy/issues/9575
        """
        # identifies parameters of the distribution
        if isinstance(self.distrib, str):
            self.distrib = getattr(ss, self.distrib)
        if self.distrib.shapes:
            self.parameters_list = [
                name.strip() for name in self.distrib.shapes.split(",")
            ]
        else:
            self.parameters_list = []
        if self.distrib.name in ss._discrete_distns._distn_names:
            self.parameters_list += ["loc"]
        elif self.distrib.name in ss._continuous_distns._distn_names:
            self.parameters_list += ["loc", "scale"]
        else:
            raise Exception(
                "Distribution name not found in discrete or continuous lists."
            )

        # prepary basic boundaries on parameters: incomplete, did not find a way to evaluate automatically the limits on shape parameters
        self.boundaries_parameters = {
            p: [-np.inf, np.inf] for p in self.parameters_list
        }
        if "scale" in self.boundaries_parameters:
            self.boundaries_parameters["scale"][0] = 0

    def find_coefficients(self):
        """
        coefficients are supposed to be written as "c#", with "#" being a number.
        """
        self.coefficients_list, self.coefficients_dict = [], {}
        for param in self.parameters_expressions:
            self.coefficients_dict[param] = []
            # iniatilize detection
            cf = ""
            for pep in (
                self.parameters_expressions[param] + " "
            ):  # adding one space at the end to append the last coefficient
                if pep == "c":
                    # starting expression for a coefficient
                    cf = "c"
                elif (cf == "c") and (
                    pep in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                ):
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

    def find_inputs(self):
        self.inputs_list = []
        for param in self.parameters_expressions:
            terms = str.split(self.parameters_expressions[param], "__")
            for l in np.array(terms)[np.arange(1, len(terms), 2)]:
                if l not in self.inputs_list:
                    self.inputs_list.append(l)

        # require a specific order for use in correct_expr_parameters
        self.inputs_list.sort(key=len, reverse=True)

    def correct_expr_parameters(self):
        # list of inputs and coefficients, sorted by descending length, to be sure that when removing them, will remove the good ones and not those with their short name contained in the long name of another
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
            dico_replace, t = {}, ""  # initialization
            for l in expr + " ":  # adding one space at the end to treat the last term
                if l in [
                    "(",
                    ")",
                    "[",
                    "]",
                    "+",
                    "-",
                    "*",
                    "/",
                    "%",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ]:
                    # this is a readable character that will not be an issue to read
                    if t not in ["", " "]:
                        # a term is here, and needs to be confirmed as readable
                        if "np." == t[: len("np.")]:
                            if t[len("np.") :] not in vars(np):
                                raise Exception(
                                    "Proposed a numpy function that doesnt exist: " + t
                                )
                            else:
                                # nothing to replace, can go with that
                                pass
                        elif "math." == t[: len("math.")]:
                            if t[len("math.") :] not in vars(math):
                                raise Exception(
                                    "Proposed a math function that doesnt exist: " + t
                                )
                            else:
                                # nothing to replace, can go with that
                                pass
                        elif t in vars(np):
                            dico_replace[t] = "np." + t
                        elif t in vars(math):
                            dico_replace[t] = "math." + t
                        else:
                            raise Exception(
                                "The term "
                                + t
                                + " appears in "
                                + self.parameters_expressions[param]
                                + ", but couldnt find an equivalent in numpy or math."
                            )
                    else:
                        # was a readable character, is still a readable character, nothing to do
                        pass
                else:
                    t += l

            # list of replacements in correct order
            tmp = list(dico_replace.keys())
            tmp.sort(key=len, reverse=True)

            # replacing
            for t in tmp:
                self.parameters_expressions[param] = self.parameters_expressions[
                    param
                ].replace(t, dico_replace[t])

            # 2. replacing names of inputs in expressions
            for i in self.inputs_list:
                self.parameters_expressions[param] = self.parameters_expressions[
                    param
                ].replace("__" + i + "__", i)

    # --------------------

    # --------------------
    # EVALUATE THE EXPRESSION
    def evaluate(self, coefficients_values, inputs_values, forced_shape=None):
        """
        Evaluates the distribution with the provided inputs and coefficients

        Parameters:
        -------
            coefficients_values: dict(c_i = values or np.array())  or  xr.dataset(c_i)  or  list of values
            inputs_values: dict(inp_i = values or np.array())  or  xr.dataset(inp_i)
            forced_shape: tuple or list of dimensions of coefficients_values and inputs_values for transposition of the shape

        /!\ Warning: with xarrays for coefficients_values and inputs_values, the outputs with have for shape first the one of the coefficient, then the one of the inputs --> trying to avoid this issue with 'forced_shape'
        """
        # Check 1: are all the coefficients provided?
        if type(coefficients_values) in [dict, xr.Dataset]:
            # case where provide explicit information on coefficients_values
            for c in self.coefficients_list:
                if c not in coefficients_values:
                    raise Exception("Missing information for the coefficient " + c)
        else:
            # case where a vector is provided, used for the optimization performed during the training
            if len(coefficients_values) != len(self.coefficients_list):
                raise Exception("Inconsistent information for the coefficients_values")
            else:
                coefficients_values = {
                    c: coefficients_values[i]
                    for i, c in enumerate(self.coefficients_list)
                }

        # Check 2: are all the inputs provided?
        for i in self.inputs_list:
            if i not in inputs_values:
                raise Exception("Missing information for the input " + i)

        # Check 3: do the inputs have the same shape
        shapes = [inputs_values[i].shape for i in self.inputs_list]
        if len(set(shapes)) > 1:
            raise Exception("Different shapes of inputs detected.")

        # Evaluation 1: coefficients
        for c in coefficients_values:
            exec(c + " = coefficients_values[c]")  # using the evil exec...

        # Evaluation 2: inputs
        for i in inputs_values:
            exec(i + " = inputs_values[i]")  # the evil exec strikes back...

        # Evaluation 3: parameters
        self.parameters_values = {}
        for param in self.parameters_list:
            # may need to silence warnings here, to avoid spamming
            self.parameters_values[param] = eval(
                self.parameters_expressions[param]
            )  # the evil exec evolves as evil eval!

        # Correcting shapes 1: constant parameters must have the shape of the inputs
        if len(self.inputs_list) > 0:
            for param in self.parameters_list:
                if (type(self.parameters_values[param]) in [int, float]) or (
                    self.parameters_values[param].ndim == 0
                ):
                    if (type(coefficients_values) == xr.Dataset) and (
                        type(inputs_values) == xr.Dataset
                    ):
                        self.parameters_values[param] = self.parameters_values[
                            param
                        ] * xr.ones_like(inputs_values[self.inputs_list[0]])
                    else:
                        self.parameters_values[param] = self.parameters_values[
                            param
                        ] * np.ones(inputs_values[self.inputs_list[0]].shape)

        # Correcting shapes 2: eventually forcing shape
        if len(self.inputs_list) > 0:
            if forced_shape is not None:
                for param in self.parameters_list:
                    dims_param = [
                        d
                        for d in forced_shape
                        if d in self.parameters_values[param].dims
                    ]
                    self.parameters_values[param] = self.parameters_values[
                        param
                    ].transpose(*dims_param)

        # evaluation of the distribution
        return self.distrib(**self.parameters_values)

    # --------------------


def probability_integral_transform(
    data,
    target_name,
    expr_start,
    expr_end,
    coeffs_start=None,
    preds_start=None,
    coeffs_end=None,
    preds_end=None,
):
    """Probability integral transform of the data given parameters of a given distribution into their equivalent in a standard normal distribution.

    Parameters
    ----------
    data: not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint' and one 2D variable with both coordinates

    target_name: str
        name of the variable to train

    expr_start : str
        string describing the starting expression

    expr_end : str
        string describing the starting expression

    preds_start: not sure yet what data format will be used at the end.
        Covariants of the starting expression. Default: empty Dataset.

    coeffs_start: xarray dataset
        Coefficients of the starting expression. Default: empty Dataset.

    preds_end: not sure yet what data format will be used at the end.
        Covariants of the ending expression. Default: empty Dataset.

    coeffs_end: xarray dataset
        Coefficients of the ending expression. Default: empty Dataset.

    Returns
    -------
    transf_inputs : not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint' and one 2D variable with both coordinates

    Notes
    -----
    - Assumptions:
        Context: The transformation may fail if the values are very unlikely, leading to a CDF equal or too close from 0 or 1, which will raise issues.
        Current solution: During training, this problem is avoided thanks to the option 'threshold_min_proba', meaning that all points in sample would have a minimum probability.
        Limit to solution: However, if transforming a sample not used during sample, and in a domain not represented in the training sample, it may cause issues.
        Additional fix: If this situation is encountered, I suggest to block the CDF values 'cdf_item' within a domain: values with a probability of 1.e-99 could be forced to 1.e-9. Unless someone has a better idea! :D
    - Disclaimer:
    - TODO:

    """
    # preparation of distributions
    expression_start = Expression(expr_start, "start")
    expression_end = Expression(expr_end, "end")
    if coeffs_start is None:
        coeffs_start = xr.Dataset()
    if coeffs_end is None:
        coeffs_end = xr.Dataset()

    # transformation
    OUT = []
    # loop to change with new data structure of MESMER
    for i, item in enumerate(data):
        data_item, scen = item
        if preds_start is None:
            preds_start_item = xr.Dataset()
        else:
            preds_start_item = preds_start[i][0]
        if preds_end is None:
            preds_end_item = xr.Dataset()
        else:
            preds_end_item = preds_end[i][0]
        print("Transforming " + target_name + ": " + scen, end="\r")

        # calculation distributions for this scenario
        distrib_start = expression_start.evaluate(
            coeffs_start, preds_start_item, forced_shape=data_item[target_name].dims
        )
        distrib_end = expression_end.evaluate(
            coeffs_end, preds_end_item, forced_shape=data_item[target_name].dims
        )

        # probabilities of the sample on the starting distribution
        cdf_item = distrib_start.cdf(data_item[target_name])

        # corresponding values on the ending distribution
        transf_item = distrib_end.ppf(cdf_item)

        # archiving
        target_name
        OUT.append((transf_item, scen))
    return OUT


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


def listxrds_to_np(listds, name_var, forcescen, coords=None):
    """
    **Temporary** function to prepare the format of inputs for training. This is meant to be used ONLY before the format of MESMERv1 data is confirmed.

    listds: list of xr Dataset
        predictors or target.
    name_var: str
        name of the variable to select.
    forcescen: list of str
        names of the scenarios to select, in this specific order. used to ensure the consistency between predictors & target.
    coords: dict
        default None (no selection). Otherwise, will loop over keys & values of dictionary to select in listds.
    """
    # looping over scenarios
    TMP = []
    for scen in forcescen:
        for item in listds:  # could be replaced with a while, but still a quick loop
            if item[1] == scen:
                # for each scenario, creating one unique serie: consistency of members & scenarios have to be ensured while loading data
                if coords is not None:
                    TMP.append(item[0][name_var].loc[coords].values.flatten())
                else:
                    TMP.append(item[0][name_var].values.flatten())
    return np.hstack(TMP)
