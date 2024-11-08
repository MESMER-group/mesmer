# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Refactored code for the training of distributions

"""

import functools
import warnings

import numpy as np
import properscoring as ps
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import basinhopping, minimize, shgo

from mesmer.core.geospatial import geodist_exact
from mesmer.mesmer_x.train_utils_mesmerx import (
    Expression,
    listxrds_to_np,
    weighted_median,
)
from mesmer.stats import gaspari_cohn


def ignore_warnings(func):
    # adapted from https://stackoverflow.com/a/70292317
    # TODO: don't suppress all warnings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return func(*args, **kwargs)

    return wrapper


# TODO: enable distrib class and training func for xarray objs
def xr_train_distrib(
    predictors,
    target,
    target_name,
    expr,  # TODO: replace by instance of Expression (instead of building it here)
    expr_name,
    option_2ndfit=True,
    r_gasparicohn_2ndfit=500,
    scores_fit=["func_optim", "NLL", "BIC"],
):
    """
    train in each grid point the target a conditional distribution as described by expr
    with the provided predictors

    Parameters
    ----------
    predictors : not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with a coordinate 'time' and 1D variable with
        this coordinate

    target : not sure yet what data format will be used at the end.
        Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint' and one
        2D variable with both coordinates

    target_name : str
        name of the variable to train

    expr : str
        See docstring of mesmer.mesmer_x.train_utils_mesmerx.Expression

    expr_name : str
        Name of the expression.

    option_2ndfit : boolean, default: True
        If True, will do a first fit in each gridpoint, AND THEN, will do another fit
        that uses as first guess the results of the first fit around each gridpoint.
        It helps in reducing the risk of spurious fits. Default: False.
        This is an experimental feature, that has not been extensively tested.

    r_gasparicohn_2ndfit : float
        Distance used in calculation of the correlation in Gaspari-Cohn matrix for the
        2nd fit.

    scores_fit : list, default: 'func_optim', 'NLL', 'BIC'
        After the fit, several scores can be calculated to assess the quality of the fit

        - func_optim: function optimized, as described in
          options_optim['type_fun_optim']: negative log likelihood or full conditional
          negative log likelihood
        - NLL: Negative Log Likelihood

          - BIC: Bayesian Information Criteria
          - CRPS: Continuous Ranked Probability Score (warning, takes a long time to
            compute)
    """

    # PREPARATION OF DATA: temporary because working on temporary format.

    # - not implementing checks on the format of predictors & target, because their
    #   current format is temporary and will be updated in MESMER v1
    # - must make 100% sure that each point of the predictors is associated with its
    #   corresponding point of the target: same ESM, same scenario, same member, same
    #   timestep, same gridpoint
    # - compiling list of scenarios, for similar order in listxrds_to_np, ensuring
    #   consistency of series of predictors & target

    list_scens_pred = [item[1] for item in predictors]
    list_scens_targ = [item[1] for item in target]
    list_scens = [scen for scen in list_scens_pred if scen in list_scens_targ]
    list_scens.sort()
    gridpoints = target[0][0].gridpoint.values

    # PREPARATION OF FIT

    # getting list of inputs from predictors
    expression_fit = Expression(expr, expr_name)

    # shaping predictors in current temporary format for use in np_train_distrib
    predictors_np = {
        inp: listxrds_to_np(listds=predictors, name_var=inp, forcescen=list_scens)
        for inp in expression_fit.inputs_list
    }

    # preparing the Datasets of the fit:
    # here, had the choice between two options: one variable for all coefficients
    # (gridpoint, coefficient) or one variable for each coefficient (gridpoint).
    # prefers 2nd option, because more similar to MESMER, and also because will
    # facilitate future developments of MESMER-X on expressions depending on the
    # gridpoint.

    coefficients_xr = xr.Dataset()
    for coef in expression_fit.coefficients_list:
        coefficients_xr[coef] = xr.DataArray(
            np.nan, coords={"gridpoint": gridpoints}, dims=("gridpoint",)
        )

    quality_xr = xr.Dataset()
    for score in scores_fit:
        quality_xr[score] = xr.DataArray(
            np.nan, coords={"gridpoint": gridpoints}, dims=("gridpoint",)
        )

    # FIT

    print(f"Fitting the variable {target_name} with the expression {expr}:")

    # looping over grid points (to replace with a map function)
    for igp, gp in enumerate(gridpoints):
        fraction = (igp + 1) / gridpoints.size
        print(f"{fraction:0.1%}", end="\r")

        # shaping target for this gridpoint
        target_np = listxrds_to_np(
            listds=target,
            name_var=target_name,
            forcescen=list_scens,
            coords={"gridpoint": gp},
        )

        # training
        coefficients_np, quality_np = np_train_distrib(
            targ_np=target_np,
            pred_np=predictors_np,
            expr_fit=expression_fit,
            scores_fit=scores_fit,
        )

        # TODO: assign to .values which is much faster (in the inner loop)
        for ic, coef in enumerate(expression_fit.coefficients_list):
            coefficients_xr[coef].loc[{"gridpoint": gp}] = coefficients_np[ic]

        for score in scores_fit:
            quality_xr[score].loc[{"gridpoint": gp}] = quality_np[score]

    # SECOND FIT if required
    if option_2ndfit:
        # preparing the Datasets of the fit
        coefficients_xr2 = coefficients_xr.copy()
        quality_xr2 = quality_xr.copy()

        # remnants of MESMERv0, because stuck with its format...
        lon_l_vec = target[0][0][target_name].lon
        lat_l_vec = target[0][0][target_name].lat

        geodist = geodist_exact(lon_l_vec, lat_l_vec)

        corr_gc = gaspari_cohn(geodist / r_gasparicohn_2ndfit)

        sel_nonan = ~np.isnan(coefficients_xr[expression_fit.coefficients_list[0]])

        print(
            f"Fitting the variable {target_name} with the expression {expr}: (2nd round)"
        )

        print(
            f"Fitting the variable {target_name} with the expression {expr}: (2nd round)"
        )

        for igp, gp in enumerate(gridpoints):
            fraction = (igp + 1) / gridpoints.size
            print(f"{fraction:0.1%}", end="\r")

            # calculate first guess, with a weighted median based on Gaspari-Cohn
            # matrix, while avoiding NaN values. Warning, weighted mean does not work
            # well, because some gridpoints may have spurious values different by orders
            # of magnitude.

            fg = np.zeros(len(expression_fit.coefficients_list))
            for ic, coef in enumerate(expression_fit.coefficients_list):
                fg[ic] = weighted_median(
                    data=coefficients_xr[coef].values[sel_nonan],
                    weights=corr_gc[igp, sel_nonan.values],
                )

            # shaping target for this gridpoint
            target_np = listxrds_to_np(
                listds=target,
                name_var=target_name,
                forcescen=list_scens,
                coords={"gridpoint": gp},
            )

            # training
            coefficients_np, quality_np = np_train_distrib(
                targ_np=target_np,
                pred_np=predictors_np,
                expr_fit=expression_fit,
                scores_fit=scores_fit,
                first_guess=fg,
            )

            # saving results
            for ic, coef in enumerate(expression_fit.coefficients_list):
                coefficients_xr2[coef].loc[{"gridpoint": gp}] = coefficients_np[ic]

            for score in scores_fit:
                quality_xr2[score].loc[{"gridpoint": gp}] = quality_np[score]

    return coefficients_xr, quality_xr


def np_train_distrib(
    targ_np,
    pred_np,
    expr_fit,
    scores_fit=["func_optim", "NLL", "BIC"],
    first_guess=None,
):
    dfit = distrib_cov(
        data_targ=targ_np,
        data_pred=pred_np,
        expr_fit=expr_fit,
        scores_fit=scores_fit,
        first_guess=first_guess,
    )
    dfit.fit()
    return dfit.coefficients_fit, dfit.quality_fit


class distrib_cov:

    def __init__(
        self,
        data_targ,
        data_pred,
        expr_fit: Expression,
        data_targ_addtest=None,  # TODO: rename to data_targ_verification?
        data_preds_addtest=None,  # TODO: rename to data_preds_verification?
        threshold_min_proba=1.0e-9,
        boundaries_params=None,
        boundaries_coeffs=None,
        first_guess=None,
        func_first_guess=None,
        scores_fit=["func_optim", "NLL", "BIC"],
        options_optim=None,  # TODO: replace by options class?
        options_solver=None,  # TODO: dito?
    ):
        """fit a conditional distribution.

        This is meant to provide flexibility in the provided expressions and robustness
        in the training. The components included in this class are:

        - (evaluation of weights for the sample)
        - tests on coefficients & parameters of the expression, in & out of sample
        - 1st optimizations for the first guess
        - 2nd optimization with minimization of the negative log likehood or full
          conditioning negative loglikelihood

        Once this class has been initialized, the go-to function is fit(). It returns
        the vector of solutions for the problem proposed.

        Parameters
        ----------
        data_targ : numpy array 1D
            Sample of the target for fit of a conditional distribution
            Normally the timeseries of the target at one gridpoint.

        data_pred : dict of 1D vectors
            Covariates for the conditional distribution. Each key must be the exact name
            of the inputs used in 'expr_fit', and the values must be aligned with the
            values in 'data_targ'.
            Normally the timeseries of the global mean predictor.

        expr_fit : class 'expression'
            Expression to train. The string provided to the class can be found in
            'expr_fit.expression'.

        data_targ_addtest : numpy array 1D, default: None
            Additional sample of the target. The fit will not be optimized on this data,
            but will test that this data remains valid. Important to avoid points out of
            support of the distribution.

        data_preds_addtest : numpy array 1D, default: None
            Additional sample of the covariates. The fit will not be optimized on this
            data, but will test that this data remains valid. Important to avoid points
            out of support of the distribution.

        threshold_min_proba : float, default: 1e-9
            Will test that each point of the sample and added sample have their
            probability higher than this value. Important to ensure that all points are
            feasible with the fitted distribution.

        boundaries_params : dict, default: None
            Prescribed boundaries on the parameters of the expression. Some basic
            boundaries are already provided through 'expr_fit.boundaries_params'.

        boundaries_coeffs : dict, optional
            Prescribed boundaries on the coefficients of the expression. Default: None.

        first_guess : numpy array, default: None
            If provided, will use these values as first guess for the first guess.

        func_first_guess : callable, default: None
            If provided, and that 'first_guess' is not provided, will be called to
            provide a first guess for the fit. This is an experimental feature, thus not
            tested.
            !! BE AWARE THAT THE ESTIMATION OF A FIRST GUESS BY YOUR MEANS COMES AT YOUR
            OWN RISKS.

        scores_fit : list of str, default: ['func_optim', 'NLL', 'BIC']
            After the fit, several scores can be calculated to assess the quality of the
            fit:

            - func_optim: function optimized, as described in
              options_optim['type_fun_optim']: negative log likelihood or full
              conditional negative log likelihood
            - NLL: Negative Log Likelihood
            - BIC: Bayesian Information Criteria
            - CRPS: Continuous Ranked Probability Score (warning, takes a long time to
              compute)

        options_optim : dict, default: None
            A dictionary with options for the function to optimize:

            * type_fun_optim: string, default: "NLL"
                If 'NLL', will optimize using the negative log likelihood. If 'fcNLL',
                will use the full conditional negative log likelihood based on the
                stopping rule. The arguments `threshold_stopping_rule`, `ind_year_thres`
                and `exclude_trigger` only apply to 'fcNLL'.

            * weighted_NLL: boolean, default: False
                If True, the optimization function will based on the weighted sum of the
                NLL, instead of the sum of the NLL. The weights are calculated as the
                inverse density in the predictors.

            * threshold_stopping_rule: float > 1, default: None
                Maximum return period, used to define the threshold of the stopping
                rule.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

            * ind_year_thres: np.array, default: None
                Positions in the predictors where the thresholds have to be tested.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

            * exclude_trigger: boolean, default: None
                Whether the threshold will be included or not in the stopping rule.
                threshold_stopping_rule, ind_year_thres and exclude_trigger must be used
                together.

        options_solver : dict, optional
            A dictionary with options for the solvers, used to determine an adequate
            first guess and for the final optimization.

            * method_fit: string, default: "Powell"
                Type of algorithm used during the optimization, using the function
                'minimize'. Prepared options: BFGS, L-BFGS-B, Nelder-Mead, Powell, TNC,
                trust-constr. The default 'Powell', is HIGHLY RECOMMENDED for its
                stability & speed.

            * xtol_req: float, default: 1e-3
                Accuracy of the fit in coefficients. Interpreted differently depending
                on 'method_fit'.

            * ftol_req: float, default: 1e-6
                Accuracy of the fit in objective.

            * maxiter: int, default: 10000
                Maximum number of iteration of the optimization.

            * maxfev: int, default: 10000
                Maximum number of evaluation of the function during the optimization.

            * error_failedfit : boolean, default: True.
                If True, will raise an issue if the fit failed.

            * fg_with_global_opti : boolean, default: False
                If True, will add a step where the first guess is improved using a
                global optimization. In theory, better, but in practice, no improvement
                in performances, but makes the fit much slower (+ ~10s).

        Notes
        -----
        This code is entirely based on the code used in MESMER-X ([1]_  and [2]_).
        However, there are some minor differences. The reasons are mostly due to
        streamlining of the code, removing deprecated features & implementing new ones.

        - Streamlining:

          - Instead of prescribing distribution & intricated inputs, using the class
            'Expression'. Shortens a lot the code and keeps the same principle.
          - The first guess has been extended to any expression. Also much shorter code.

        - Deprecated:

          - Fit of the logit of sample instead of the sample. It was used for
            expressions with sigmoids, but did not improve the fit
          - Test in the coefficients of a sigmoid to avoid simultaneous drifts. Appeared
            mostly with logits, and not feasible with class 'expression'

        - Removed
          - In tests, was checking not only whether the coefficients or parameters were
            exceeding boundaries, but also were close. Removed because of modifications
            in first guess didn't work with situations with scale=0.
          - Forcing values of certain parameters (eg mu for poisson) to be integers: to
            implement in class 'expression'?

        - Implemented:

          - Additional sample used to test the validity of the fit through tests on
            coefficients & parameters.
          - Minimum probability for the points in the sample. Useful to avoid having
            points becoming extremely unlikely, despite being in the sample.
          - Optimization may be now account as well on the stopping rule (inspired b
            https://github.com/OpheliaMiralles/pykelihood)
          - Weighting points of the sample with inverse of their density. Useful to give
            equivalent weights to the whole domain fitted.

        - Reasons for choosing that over available alternatives:
          - scipy.stats.fit: nothing about conditional distribution, all tests on
            coefficients, parameters, values AND nothing about first guess
          - pykelihood: not enough for the first guess
            (https://github.com/OpheliaMiralles/pykelihood/blob/master/pykelihood/kernels.py)
          - symfit: not enough for the first guess
            (https://symfit.readthedocs.io/en/stable/fitting_types.html#likelihood)

        .. [1] https://doi.org/10.1029%2F2022GL099012

        .. [2] https://doi.org/10.5194/esd-14-1333-2023

        """

        # preparing basic information
        self.data_targ = data_targ

        # can be different from length of predictors IF no predictors.
        self.n_sample = len(self.data_targ)

        if np.isnan(self.data_targ).any():
            raise ValueError("nan values in target")

        if np.isinf(self.data_targ).any():
            raise ValueError("infinite values in target")

        self.data_pred = data_pred

        # TODO: raise error if data_pred is not a dict

        if any(np.isnan(self.data_pred[pred]).any() for pred in self.data_pred):
            raise ValueError("nan values in predictors")

        if any(np.isinf(self.data_pred[pred]).any() for pred in self.data_pred):
            raise ValueError("infinite values in predictors")

        self.expr_fit = expr_fit

        # preparing additional data
        add_test = (data_targ_addtest is not None) and (data_preds_addtest is not None)
        self.add_test = add_test

        # TODO: this case can not happen - can it?
        if (self.add_test is False) and (
            (data_targ_addtest is not None) or (data_preds_addtest is not None)
        ):
            raise ValueError(
                "Only one of `data_targ_addtest` & `data_preds_addtest` have been"
                " provided, not both of them."
            )

        self.data_targ_addtest = data_targ_addtest
        self.data_preds_addtest = data_preds_addtest

        if (threshold_min_proba <= 0) or (1 < threshold_min_proba):
            raise ValueError("`threshold_min_proba` must be in [0;1[")

        self.threshold_min_proba = threshold_min_proba

        # preparing information on boundaries
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

        # preparing additional information
        self.first_guess = first_guess
        self.func_first_guess = func_first_guess
        self.n_coeffs = len(self.expr_fit.coefficients_list)

        if (self.first_guess is not None) and (len(self.first_guess) != self.n_coeffs):
            raise ValueError(
                f"The provided first guess does not have the correct shape: {self.n_coeffs}"
            )

        self.scores_fit = scores_fit

        # preparing information on solver
        default_options_solver = {
            "method_fit": "Powell",
            "xtol_req": 1e-6,
            "ftol_req": 1.0e-6,
            "maxiter": 1000 * self.n_coeffs * np.log(self.n_coeffs),
            "maxfev": 1000 * self.n_coeffs * np.log(self.n_coeffs),
            "error_failedfit": False,
            "fg_with_global_opti": False,
        }

        options_solver = options_solver or {}
        if not isinstance(options_solver, dict):
            raise ValueError("`options_solver` must be a dictionary")

        # TODO: use get? (e.g. self.method_fit = options_solver.get("method_fit", "Powell")
        options_solver = default_options_solver | options_solver

        self.xtol_req = options_solver["xtol_req"]
        self.ftol_req = options_solver["ftol_req"]
        self.maxiter = options_solver["maxiter"]
        self.maxfev = options_solver["maxfev"]
        self.method_fit = options_solver["method_fit"]

        if self.method_fit not in (
            "BFGS",
            "L-BFGS-B",
            "Nelder-Mead",
            "Powell",
            "TNC",
            "trust-constr",
        ):
            raise ValueError("method for this fit not prepared, to avoid")

        xtol = {
            "BFGS": "xrtol",
            "L-BFGS-B": "gtol",
            "Nelder-Mead": "xatol",
            "Powell": "xtol",
            "TNC": "xtol",
            "trust-constr": "xtol",
        }
        ftol = {
            "BFGS": "gtol",
            "L-BFGS-B": "ftol",
            "Nelder-Mead": "fatol",
            "Powell": "ftol",
            "TNC": "ftol",
            "trust-constr": "gtol",
        }

        self.name_xtol = xtol[self.method_fit]
        self.name_ftol = ftol[self.method_fit]
        self.error_failedfit = options_solver["error_failedfit"]
        self.fg_with_global_opti = options_solver["fg_with_global_opti"]

        # preparing information on functions to optimize
        default_options_optim = dict(
            weighted_NLL=False,
            type_fun_optim="NLL",
            threshold_stopping_rule=None,
            exclude_trigger=None,
            ind_year_thres=None,
        )

        options_optim = options_optim or {}

        if not isinstance(options_optim, dict):
            raise ValueError("`options_optim` must be a dictionary")

        options_optim = default_options_optim | options_optim

        # preparing weights
        # TODO: move this out of init or think of more flexible bins
        self.weighted_NLL = options_optim["weighted_NLL"]
        self.weights_driver = self.get_weights()

        # preparing information for the stopping rule
        self.type_fun_optim = options_optim["type_fun_optim"]
        self.threshold_stopping_rule = options_optim["threshold_stopping_rule"]
        self.ind_year_thres = options_optim["ind_year_thres"]
        self.exclude_trigger = options_optim["exclude_trigger"]

        if self.type_fun_optim == "NLL" and (
            self.threshold_stopping_rule is not None or self.ind_year_thres is not None
        ):
            raise ValueError(
                "`threshold_stopping_rule` and `ind_year_thres` not used for"
                " `type_fun_optim='NLL'`"
            )

        if self.type_fun_optim == "fcNLL" and (
            self.threshold_stopping_rule is None or self.ind_year_thres is None
        ):
            raise ValueError(
                "`type_fun_optim='fcNLL'` needs both, `threshold_stopping_rule`"
                "  and `ind_year_thres`."
            )

    # TODO: don't do this in init. Give the user the option to either use this function
    # or give their own weigths as soon as we switch the xarray wrapper into here and
    # the user actually initialized this class themselves
    def get_weights(self, n_bins_density=40):

        if self.weighted_NLL:
            weights_driver = self._get_weights_nll(n_bins_density=n_bins_density)
        else:
            weights_driver = np.ones(self.data_targ.shape)
        # TODO: move the normalization into the function
        return weights_driver / np.sum(weights_driver)

    def _get_weights_nll(self, n_bins_density=40):
        """
        Generate weights for the sample, based on the inverse of the density of the
        predictors. More precisely, the density of the predictors is measured by a
        multidimensional histogram where each dimension is one of the predictors. The
        histogram is then smoothed by a regular grid interpolator to give the density
        of the predictors in this "predictor space". Subsequently, the weights are
        the inverse of this density of the predictors. Consequently, Samples in regions
        of this space with low densitiy will have higher weights, this is, "unusual" samples
        will have more weight.

        Parameters
        ----------
        n_bins_density : int, default: 40
            Number of bins used to calculate the density of the predictors.

        Returns
        -------
        weights_driver : numpy array 1D
            Weights for the sample, based on the inverse of the density of the
            predictors.

        Example
        -------
        TODO

        """

        # if no predictors, straightforward
        if len(self.data_pred) == 0:
            # TODO: isn't data_pred a dict and does therefore not have a shape? Yes. Also it is empty.
            # TODO: Do we want to allow no predictor?
            return np.ones(self.data_targ.shape)

        # explode data_pred dictionary into a single array for all predictors
        tmp = np.array(list(self.data_pred.values())).T

        # assessing limits on each axis
        # TODO *nan*min/max should not be necessary bc we already checked for nan values in the data?
        mn, mx = np.nanmin(tmp, axis=0), np.nanmax(tmp, axis=0)

        # TODO: at the moment bins == edges, either change bins to edges and do n_bins_density + 1
        # or change bins = n_bins_density in histogramdd
        bins = np.linspace(
            (mn - 0.05 * (mx - mn)),
            (mx + 0.05 * (mx - mn)),
            n_bins_density,
        )

        # interpolating over whole region
        gmt_hist, edges = np.histogramdd(sample=tmp, bins=bins.T)

        gmt_bins_center = [0.5 * (edge[1:] + edge[:-1]) for edge in edges]

        # TODO: add bounds_error=False, fill_value=None (extrapolates the values outside the grid)
        interp = RegularGridInterpolator(
            points=gmt_bins_center,
            values=gmt_hist,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        # evaluate interpolated density at datapoints
        density = interp(tmp)

        return 1 / density  # inverse of density

    def _test_coeffs_in_bounds(self, values_coeffs):

        # checking set boundaries on coefficients
        for coeff in self.boundaries_coeffs:
            bottom, top = self.boundaries_coeffs[coeff]

            # TODO: move this check to __init__
            if coeff not in self.expr_fit.coefficients_list:
                raise ValueError(
                    f"Provided wrong boundaries on coefficient, {coeff}"
                    " does not exist in expr_fit"
                )

            values = values_coeffs[self.expr_fit.list_coefficients.index(coeff)]

            if np.any(values < bottom) or np.any(top < values):
                # out of boundaries
                return False

        return True

    def _test_evol_params(self, distrib, data):

        # checking set boundaries on parameters
        for param in self.boundaries_params:
            bottom, top = self.boundaries_params[param]

            # TODO: avoid using implementation detail of frozen distr of sp.stats
            param_values = distrib.kwds[param]

            # out of boundaries
            # TODO: why >= (and not >) or < (and not <=)?
            if np.any(param_values < bottom) or np.any(param_values >= top):
                return False

        # test of the support of the distribution: is there any data out of the
        # corresponding support? dont try testing if there are issues on the parameters

        bottom, top = distrib.support()

        # out of support
        if (
            np.any(np.isnan(bottom))
            or np.any(np.isnan(top))
            or np.any(data < bottom)
            or np.any(data > top)
        ):
            return False

        return True

    def _test_proba_value(self, distrib, data):
        # tested values must have a minimum probability of occurring, i.e. be in a
        # confidence interval
        # NOTE: DONT write 'x=data', because 'x' may be called differently for some
        # distribution (eg 'k' for poisson).

        cdf = distrib.cdf(data)
        thres = self.threshold_min_proba
        # TODO (mathause): why does this use cdf and not pdf?
        return np.all(1 - cdf >= thres) and np.all(cdf >= thres)

    def validate_coefficients(self, coefficients):
        """validate coefficients

        Validate estimated coefficients
        1. using the target data and predictors and
        2. potentially the cross-validaten data
        """

        test_coeff = self._test_coeffs_in_bounds(coefficients)

        # tests on coeffs show already that it wont work: fill in the rest with False
        if not test_coeff:
            return test_coeff, False, False, False

        # evaluate the distribution for the predictors and this iteration of coeffs
        distrib = self.expr_fit.evaluate(coefficients, self.data_pred)

        if self.add_test:
            distrib_add = self.expr_fit.evaluate(coefficients, self.data_preds_addtest)

        # test for the validity of the parameters
        test_param = self._test_evol_params(distrib, self.data_targ)

        if self.add_test:
            test_param &= self._test_evol_params(distrib_add, self.data_targ_addtest)

        # tests on params show already that it wont work: fill in the rest with False
        if not test_param:
            return test_coeff, test_param, False, False

        # test for the probability of the values
        if self.threshold_min_proba is None:
            return test_coeff, test_param, True, distrib

        test_proba = self._test_proba_value(distrib, self.data_targ)

        if self.add_test:
            test_proba &= self._test_proba_value(distrib_add, self.data_targ_addtest)

        # return values for each test and the distribution that has already been
        # evaluated
        return test_coeff, test_param, test_proba, distrib

    # suppress nan & inf warnings
    @ignore_warnings
    def find_fg(self):
        """
        compute first guess of the coefficients, to ensure convergence of the incoming
        fit.

        Motivation:
            In many situations, the fit may be complex because of complex expressions
            for the conditional distributions & because large domains in the set of
            coefficients lead to invalid fits (e.g. sample out of support).

        Criteria:
            The method must return a first guess that is ROBUST (close to the global
            minimum) & VALID (respect all conditions implemented in the tests), must be
            FLEXIBLE (any sample, any distribution & any expression).

        Method:
            1. Global fit of the coefficients of the location using derivatives, to
               improve the very first guess for the location
            2. Fit of the coefficients of the location, assuming that the center of the
               distribution should be close from its location.
            3. Fit of the coefficients of the scale, assuming that the deviation of the
               distribution should be close from its scale.
            4. Fit of remaining coefficients, assuming that the sample must be within
               the support of the distribution, with some margin.
            5. Improvement of all coefficients: better coefficients on location & scale,
               and especially estimating those on shape. Based on the Negative Log
               Likelihoog, albeit without the validity of the coefficients.
            6. Improvement of coefficients: ensuring that all points are within a likely
               support of the distribution. Two possibilities tried:
               (For 4, tried 2 approaches: based on CDF or based on NLL^n. The idea is
               to penalize very unlikely values, both works, but NLL^n works as well for
               extremely unlikely values, that lead to division by 0 with CDF)
               (step 5 still not always working, trying without?)

        Risks for the method:
            The only risk that I identify is if the user sets boundaries on coefficients
            or parameters that would reject this optimal first guess, and the domain of
            the next local minimum is far away.

        Justification for the method:
            This is a surprisingly complex problem to satisfy the criteria of
            robustness, validity & flexibility.

            a. In theory, this problem could be solved with a global optimization. Among
               the global optimizers available in scipy, they come with two types of
               requirements:
               - basinhopping: requires a first guess. Tried with first guess close from
                 optimum, good performances, but lacks in reproductibility and
                 stability: not reliable enough here. Ok if runs much longer.
               - brute, differential_evolution, shgo, dual_annealing, direct: requires
                 bounds
                 - brute, dual_annealing, direct: performances too low & too slow
                 - differential_evolution: lacks in reproductibility & stability
                 - shgo: good performances with the right sampling method, relatively
                   fast, but still adds ~10s. Highly dependent on the bounds, must not
                   be too large.
               The best global optimizer, shgo, would then require bounds that are not
               too large.

            b. The set of coefficients have only sparse valid domains. The distance
               between valid domains is often bigger than the adequate width of bounds
               for shgo.
               It implies that the bounds used for shgo must correspond to the valid
               domain that already contains the global minimum, and no other domain.
               It implies that the region of the global minimum must have already been
               identified...
               This is the tricky part, here are the different elements that I used to
               tackle this problem:
               - all combinations of local & global optimizers of scipy
               - including or not the tests for validity
               - assessment of bounds for global optimizers based on ratio of NLL or
                 domain of data_targ
               - first optimizations based on mean & spread
               - brute force approach on logspace valid for parameters (not scalable
                 with # of parameters)
             c. The only solution that was working is inspired by what the
                semi-analytical solution used in the original code of MESMER-X. Its
                principle is to use fits first for location coefficients, then scale,
                then improve.
                The steps are described in the section Method, steps 1-5. At step 5 that
                these coefficients are very close from the global minimum. shgo usually
                does not bring much at the expense of speed.
                Thus skipping global minimum unless asked.

        Warnings
        --------
        To anyone trying to improve this part:
        If you attempt to modify the calculation of the first guess, it is *absolutely
        mandatory* to test the new code on all criteria: ROBUSTNESS, VALIDITY,
        FLEXIBILITY. In particular, it is mandatory to test it for different
        situations: variables, grid points, distributions & expressions.
        """

        # preparing derivatives to estimate derivatives of data along predictors,
        # and infer a very first guess for the coefficients facilitates the
        # representation of the trends
        self.smooth_data_targ = self.smooth_data(self.data_targ)

        m, s = np.mean(self.smooth_data_targ), np.std(self.smooth_data_targ)

        ind_targ_low = np.where(self.smooth_data_targ < m - s)[0]
        ind_targ_high = np.where(self.smooth_data_targ > m + s)[0]

        pred_low = {p: np.mean(self.data_pred[p][ind_targ_low]) for p in self.data_pred}
        pred_high = {
            p: np.mean(self.data_pred[p][ind_targ_high]) for p in self.data_pred
        }

        deriv_targ = {
            p: (
                np.mean(self.smooth_data_targ[ind_targ_high])
                - np.mean(self.smooth_data_targ[ind_targ_low])
            )
            / (pred_high[p] - pred_low[p])
            for p in self.data_pred
        }

        self.fg_info_derivatives = {
            "pred_low": pred_low,
            "pred_high": pred_high,
            "deriv_targ": deriv_targ,
            "m": m,
        }

        # Initialize first guess
        if self.first_guess is None:
            self.fg_coeffs = np.zeros(self.n_coeffs)

            # Step 1: fit coefficients of location (objective: generate an adequate
            # first guess for the coefficients of location. proven to be necessary
            # in many situations, & accelerate step 2)
            globalfit_d01 = basinhopping(
                func=self.fg_fun_deriv01, x0=self.fg_coeffs, niter=10
            )
            # warning, basinhopping tends to introduce non-reproductibility in fits,
            # reduced when using 2nd round of fits

            self.fg_coeffs = globalfit_d01.x

        else:
            # Using provided first guess, eg from 1st round of fits
            self.fg_coeffs = np.copy(self.first_guess)

        self.mem = np.copy(self.fg_coeffs)

        # Step 2: fit coefficients of location (objective: improving the subset of
        # location coefficients)
        self.fg_ind_loc = np.array(
            [
                self.expr_fit.coefficients_list.index(c)
                for c in self.expr_fit.coefficients_dict["loc"]
            ]
        )
        localfit_loc = self.minimize(
            func=self.fg_fun_loc,
            x0=self.fg_coeffs[self.fg_ind_loc],
            fact_maxfev_iter=len(self.fg_ind_loc) / self.n_coeffs,
            option_NelderMead="best_run",
        )
        self.fg_coeffs[self.fg_ind_loc] = localfit_loc.x

        # Step 3: fit coefficients of scale (objective: improving the subset of
        # scale coefficients)
        scale = self.expr_fit.coefficients_dict["scale"]
        self.fg_ind_sca = np.array(
            [self.expr_fit.coefficients_list.index(c) for c in scale]
        )
        if self.first_guess is None:
            # compared to all 0, better for ref level but worse for trend
            x0 = np.full(len(scale), fill_value=np.std(self.data_targ))

        else:
            x0 = self.fg_coeffs[self.fg_ind_sca]

        localfit_sca = self.minimize(
            func=self.fg_fun_sca,
            x0=x0,
            fact_maxfev_iter=len(self.fg_ind_sca) / self.n_coeffs,
            option_NelderMead="best_run",
        )
        self.fg_coeffs[self.fg_ind_sca] = localfit_sca.x

        # Step 4: fit other coefficients (objective: improving the subset of
        # other coefficients. May use multiple coefficients, eg beta distribution)
        other_params = [
            p for p in self.expr_fit.parameters_list if p not in ["loc", "scale"]
        ]
        if len(other_params) > 0:
            self.fg_ind_others = []
            for param in other_params:
                for c in self.expr_fit.coefficients_dict[param]:
                    self.fg_ind_others.append(self.expr_fit.coefficients_list.index(c))

            self.fg_ind_others = np.array(self.fg_ind_others)

            localfit_others = self.minimize(
                func=self.fg_fun_others,
                x0=self.fg_coeffs[self.fg_ind_others],
                fact_maxfev_iter=len(self.fg_ind_others) / self.n_coeffs,
                option_NelderMead="best_run",
            )
            self.fg_coeffs[self.fg_ind_others] = localfit_others.x

        # Step 5: fit coefficients using NLL (objective: improving all coefficients,
        # necessary to get good estimates for shape parameters, and avoid some local minima)
        localfit_nll = self.minimize(
            func=self.fg_fun_NLL_notests,
            x0=self.fg_coeffs,
            fact_maxfev_iter=1,
            option_NelderMead="best_run",
        )
        self.fg_coeffs = localfit_nll.x

        test_coeff, test_param, test_proba, _ = self.validate_coefficients(
            self.fg_coeffs
        )

        if not (test_coeff and test_param and test_proba):
            # Step 6: fit on CDF or LL^n (objective: improving all coefficients, necessary
            # to have all points within support. NB: NLL doesnt behave well enough here)
            # two potential functions:
            if False:
                # TODO: unreachable?
                # fit coefficients on CDFs
                fun_opti_prob = self.fg_fun_cdfs
            else:
                # fit coefficients on log-likelihood to the power n
                fun_opti_prob = self.fg_fun_LL_n

            localfit_opti = self.minimize(
                func=fun_opti_prob,
                x0=self.fg_coeffs,
                fact_maxfev_iter=1,
                option_NelderMead="best_run",
            )
            if ~np.any(np.isnan(localfit_opti.x)):
                self.fg_coeffs = localfit_opti.x

        # Step 7: if required, global fit within boundaries
        if self.fg_with_global_opti:

            # find boundaries on each coefficient
            bounds = []

            # TODO: does this assume the coeffs are ordered?
            for i_c in np.arange(self.n_coeffs):
                a = self.find_bound(i_c=i_c, x0=self.fg_coeffs, fact_coeff=-0.05)
                b = self.find_bound(i_c=i_c, x0=self.fg_coeffs, fact_coeff=0.05)
                vals_bounds = (a, b)

                bounds.append([np.min(vals_bounds), np.max(vals_bounds)])

            # global minimization, using the one with the best performances in this
            # situation. sobol or halton, observed lower performances with
            # implicial. n=1000, options={'maxiter':10000, 'maxev':10000})
            globalfit_all = shgo(self.func_optim, bounds, sampling_method="sobol")
            self.fg_coeffs = globalfit_all.x

    def minimize(self, func, x0, fact_maxfev_iter=1, option_NelderMead="dont_run"):
        """
        options_NelderMead: str
            * dont_run: would minimize only the chosen solver in method_fit
            * fail_run: would minimize using Nelder-Mead only if the chosen solver in
              method_fit fails
            * best_run: will minimize using Nelder-Mead and the chosen solver in
              method_fit, then select the best results
        """
        fit = minimize(
            func,
            x0=x0,
            method=self.method_fit,
            options={
                "maxfev": self.maxfev * fact_maxfev_iter,
                "maxiter": self.maxfev * fact_maxfev_iter,
                self.name_xtol: self.xtol_req,
                self.name_ftol: self.ftol_req,
            },
        )

        # observed that Powell solver is much faster, but less robust. May rarely create
        # directly NaN coefficients or wrong local optimum => Nelder-Mead can be used at
        # critical steps or when Powell fails.

        if (option_NelderMead == "fail_run" and fit.success is False) or (
            option_NelderMead == "best_run"
        ):
            fit_NM = minimize(
                func,
                x0=x0,
                method="Nelder-Mead",
                options={
                    "maxfev": self.maxfev * fact_maxfev_iter,
                    "maxiter": self.maxiter * fact_maxfev_iter,
                    "xatol": self.xtol_req,
                    "fatol": self.ftol_req,
                },
            )
            if (option_NelderMead == "fail_run") or (
                option_NelderMead == "best_run" and fit_NM.fun < fit.fun
            ):
                fit = fit_NM
        return fit

    @staticmethod
    def smooth_data(data, nn=10):
        return np.convolve(data, np.ones(nn) / nn, mode="same")

    def fg_fun_deriv01(self, x):
        params = self.expr_fit.evaluate_params(x, self.fg_info_derivatives["pred_low"])
        loc_low = params["loc"]
        params = self.expr_fit.evaluate_params(x, self.fg_info_derivatives["pred_high"])
        loc_high = params["loc"]

        deriv = {
            p: (loc_high - loc_low)
            / (
                self.fg_info_derivatives["pred_high"][p]
                - self.fg_info_derivatives["pred_low"][p]
            )
            for p in self.data_pred
        }

        return (
            np.sum(
                [
                    (deriv[p] - self.fg_info_derivatives["deriv_targ"][p]) ** 2
                    for p in self.data_pred
                ]
            )
            + (0.5 * (loc_low + loc_high) - self.fg_info_derivatives["m"]) ** 2
        )

    def fg_fun_loc(self, x_loc):
        x = np.copy(self.fg_coeffs)
        x[self.fg_ind_loc] = x_loc
        params = self.expr_fit.evaluate_params(x, self.data_pred)
        loc = params["loc"]
        return np.sum((loc - self.smooth_data_targ) ** 2)

    def fg_fun_sca(self, x_sca):
        x = np.copy(self.fg_coeffs)
        x[self.fg_ind_sca] = x_sca
        params = self.expr_fit.evaluate_params(x, self.data_pred)
        loc, sca = params["loc"], params["scale"]
        # ^ better to use that one instead of deviation, which is affected by the scale
        dev = np.abs(self.data_targ - loc)
        return np.sum((dev - sca) ** 2)

    def fg_fun_others(self, x_others, margin0=0.05):
        # preparing support
        x = np.copy(self.fg_coeffs)
        x[self.fg_ind_others] = x_others

        distrib = self.expr_fit.evaluate(x, self.data_pred)
        bot, top = distrib.support()
        val_bot = np.min(self.data_targ - bot)
        val_top = np.min(top - self.data_targ)
        # preparing margin on support
        m = np.mean(self.data_targ)
        s = np.std(self.data_targ - m)
        # optimization
        if val_bot < 0:
            # limit of val_bottom --> 0- = 1/margin0*s
            return np.exp(-val_bot) * 1 / (margin0 * s)
        elif val_top < 0:
            # limit of val_top --> 0+ = 1/margin0*s
            return np.exp(-val_top) * 1 / (margin0 * s)
        else:
            return 1 / (val_bot + margin0 * s) + 1 / (val_top + margin0 * s)

    def fg_fun_NLL_notests(self, coefficients):
        distrib = self.expr_fit.evaluate(coefficients, self.data_pred)
        self.ind_data_ok = np.arange(self.data_targ.size)
        return self.neg_loglike(distrib)

    def fg_fun_cdfs(self, x):
        distrib = self.expr_fit.evaluate(x, self.data_pred)
        cdf = distrib.cdf(self.data_targ)

        if self.threshold_min_proba is None:
            thres = 10 * 1.0e-9
        else:
            thres = np.min([0.1, 10 * self.threshold_min_proba])

        if np.any(np.isnan(cdf)):
            return np.inf

        # DO NOT CHANGE THESE EXPRESSIONS!!
        term_low = (thres - np.min(cdf)) ** 2 / np.min(cdf) ** 2
        term_high = (thres - np.min(1 - cdf)) ** 2 / np.min(1 - cdf) ** 2
        return np.max([term_low, term_high])

    def fg_fun_LL_n(self, x, n=4):
        distrib = self.expr_fit.evaluate(x, self.data_pred)
        LL = np.sum(distrib.logpdf(self.data_targ) ** n)
        return LL

    def find_bound(self, i_c, x0, fact_coeff):
        # could be accelerated using dichotomy, but 100 iterations max are fast enough
        # not to require to make this part more complex.
        x, iter, itermax, test = np.copy(x0), 0, 100, True
        while test and (iter < itermax):
            test_c, test_p, test_v, _ = self.validate_coefficients(x)
            test = test_c and test_p and test_v
            x[i_c] += fact_coeff * x[i_c]
            iter += 1
        return x[i_c]

    # OPTIMIZATION FUNCTIONS & SCORES
    def func_optim(self, coefficients):
        # check whether these coefficients respect all conditions: if so, can compute a
        # value for the optimization
        test_coeff, test_param, test_proba, distrib = self.validate_coefficients(
            coefficients
        )

        if test_coeff and test_param and test_proba:

            # check for the stopping rule
            if self.type_fun_optim == "fcNLL":
                # will apply the stopping rule: splitting data_fit into two sets of data
                # using the given threshold
                self.ind_data_ok, self.ind_data_stopped = self.stopping_rule(distrib)
            else:
                self.ind_data_ok = slice(None)

            # compute negative loglikelihood
            NLL = self.neg_loglike(distrib)

            # eventually compute full conditioning
            if self.type_fun_optim == "fcNLL":
                FC = self.fullcond_thres(distrib)
                optim = NLL + FC
            else:
                optim = NLL

        # returns value for optimization
        if test_coeff and test_param and test_proba:
            return optim
        else:
            # something wrong: returns a blocking value
            return np.inf

    def neg_loglike(self, distrib):
        return -self.loglike(distrib)

    def loglike(self, distrib):
        # compute loglikelihood
        if self.expr_fit.is_distrib_discrete:
            LL = distrib.logpmf(self.data_targ)
        else:
            LL = distrib.logpdf(self.data_targ)

        # weighted sum of the loglikelihood
        value = np.sum((self.weights_driver * LL)[self.ind_data_ok])

        if np.isnan(value):
            return -np.inf
        else:
            return value

    def stopping_rule(self, distrib):
        # evaluating threshold over time
        thres_t = distrib.isf(q=1 / self.threshold_stopping_rule)

        # selecting the minimum over the years to check
        thres = np.min(thres_t[self.ind_year_threshold])

        # identifying where exceedances occur
        if self.exclude_trigger:
            ind_data_stopped = self.data_targ > thres
        else:
            ind_data_stopped = self.data_targ >= thres

        # identifying remaining positions
        ind_data_ok = ~ind_data_stopped
        return ind_data_ok, ind_data_stopped

    def fullcond_thres(self, distrib):
        # calculating 2nd term for full conditional of the NLL
        # fc1 = distrib.logcdf(self.data_targ)
        fc2 = distrib.sf(self.data_targ)

        # return np.sum( (self.weights_driver * fc1)[self.ind_stopped_data] )
        # TODO: not 100% sure here, to double-check

        return np.log(np.sum((self.weights_driver * fc2)[self.ind_stopped_data]))

    def bic(self, distrib):
        return self.n_coeffs * np.log(self.n_sample) / self.n_sample - 2 * self.loglike(
            distrib
        )

    # TODO: remove /self.n_sample? bc weights are already normalized

    def crps(self, coeffs):
        # ps.crps_quadrature cannot be applied on conditional distributions, thu
        # calculating in each point of the sample, then averaging
        # NOTE: WARNING, TAKES A VERY LONG TIME TO COMPUTE

        tmp_cprs = []
        for i in np.arange(self.n_sample):
            distrib = self.expr_fit.evaluate(
                coeffs, {p: self.data_pred[p][i] for p in self.data_pred}
            )
            tmp_cprs.append(
                ps.crps_quadrature(
                    x=self.data_targ[i],
                    cdf_or_dist=distrib,
                    xmin=-10 * np.abs(self.data_targ[i]),
                    xmax=10 * np.abs(self.data_targ[i]),
                    tol=1.0e-4,
                )
            )

        # averaging
        return np.sum(self.weights_driver * np.array(tmp_cprs))

    @ignore_warnings  # suppress nan & inf warnings
    def fit(self):

        # Before fitting, need a good first guess, using 'find_fg'.
        if self.func_first_guess is not None:
            self.func_first_guess()
        else:
            self.find_fg()

        m = self.minimize(
            func=self.func_optim,
            x0=self.fg_coeffs,
            fact_maxfev_iter=1,
            option_NelderMead="best_run",
        )

        # checking if the fit has failed
        if self.error_failedfit and (m.success is False):
            raise ValueError("Failed fit.")
        else:
            self.coefficients_fit = m.x
            self.eval_quality_fit()

    def eval_quality_fit(self):
        # initialize
        self.quality_fit = {}

        # basic result: optimized value
        if "func_optim" in self.scores_fit:
            self.quality_fit["func_optim"] = self.func_optim(self.coefficients_fit)

        # distribution obtained
        distrib = self.expr_fit.evaluate(self.coefficients_fit, self.data_pred)

        # NLL averaged over sample
        if "NLL" in self.scores_fit:
            self.quality_fit["NLL"] = self.neg_loglike(distrib)

        # BIC averaged over sample
        if "BIC" in self.scores_fit:
            self.quality_fit["BIC"] = self.bic(distrib)

        # CRPS
        if "CRPS" in self.scores_fit:
            self.quality_fit["CRPS"] = self.crps(self.coefficients_fit)
