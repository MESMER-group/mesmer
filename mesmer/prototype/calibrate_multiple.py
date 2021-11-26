import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

from .calibrate import AutoRegression1DOrderSelection, AutoRegression1D
from .utils import calculate_gaspari_cohn_correlation_matrices


def _get_predictor_dims(predictors):
    predictors_dims = {k: v.dims for k, v in predictors.items()}
    predictors_dims_unique = set(predictors_dims.values())
    if len(predictors_dims_unique) > 1:
        raise AssertionError(
            f"Dimensions of predictors are not all the same, we have: {predictors_dims}"
        )

    return list(predictors_dims_unique)[0]


def _get_stack_coord_name(inp_array):
    stack_coord_name = "stacked_coord"
    if stack_coord_name in inp_array.dims:
        stack_coord_name = "memser_stacked_coord"

    if stack_coord_name in inp_array.dims:
        raise NotImplementedError("You have dimensions we can't safely unstack yet")

    return stack_coord_name


def _check_coords_match(obj, obj_other, check_coord):
    coords_match = obj.coords[check_coord].equals(obj_other.coords[check_coord])
    if not coords_match:
        raise AssertionError(f"{check_coord} is not the same on {obj} and {obj_other}")


def _flatten(inp, dims_to_flatten):
    stack_coord_name = _get_stack_coord_name(inp)
    inp_flat = inp.stack({stack_coord_name: dims_to_flatten}).dropna(
        stack_coord_name
    )

    return inp_flat, stack_coord_name


def _flatten_predictors(predictors, dims_to_flatten, stack_coord_name):
    predictors_flat = []
    for v, vals in predictors.items():
        if stack_coord_name in vals.dims:
            raise AssertionError(f"{stack_coord_name} is already in {vals.dims}")

        vals_flat = vals.stack({stack_coord_name: dims_to_flatten}).dropna(
            stack_coord_name
        )
        vals_flat.name = v
        predictors_flat.append(vals_flat)

    out = xr.merge(predictors_flat).to_stacked_array(
        "predictor", sample_dims=[stack_coord_name]
    )

    return out


def flatten_predictors_and_target(predictors, target):
    dims_to_flatten = _get_predictor_dims(predictors)
    stack_coord_name = _get_stack_coord_name(target)

    target_flattened = target.stack({stack_coord_name: dims_to_flatten}).dropna(
        stack_coord_name
    )
    predictors_flattened = _flatten_predictors(
        predictors, dims_to_flatten, stack_coord_name
    )
    _check_coords_match(target_flattened, predictors_flattened, stack_coord_name)

    return predictors_flattened, target_flattened, stack_coord_name


def _loop_levels(inp, levels):
    # annoyingly, there doesn't seem to be an inbuilt solution for this
    # https://github.com/pydata/xarray/issues/2438
    def _yield_level(inph, levels_left, out_names):
        for name, values in inph.groupby(levels_left[0]):
            out_names_here = out_names + [name]
            if len(levels_left) == 1:
                yield tuple(out_names_here), values
            else:
                yield from _yield_level(values, levels_left[1:], out_names_here)

    for names, values in _yield_level(inp, levels, []):
        yield names, values


def _select_auto_regressive_process_order(
    target,
    maxlag,
    ic,
    scenario_level="scenario",
    ensemble_member_level="ensemble_member",
    q=50,
    interpolation="nearest",
):
    """

    interpolation : str
        Passed to :func:`numpy.percentile`. Interpolation is not a good way to
        go here because it could lead to an AR order that wasn't actually chosen by any run. We recommend using the default value i.e. "nearest".
    """
    store = []

    for (scenario, ensemble_member), values in _loop_levels(
        target, (scenario_level, ensemble_member_level)
    ):
        orders = AutoRegression1DOrderSelection().calibrate(
            values, maxlag=maxlag, ic=ic
        )
        keep_order = 0 if orders is None else orders[-1]
        store.append(
            {
                "scenario": scenario,
                "ensemble_member": ensemble_member,
                "order": keep_order,
            }
        )

    store = pd.DataFrame(store).set_index(["scenario", "ensemble_member"])
    res = (
        store.groupby("scenario")["order"]
        # first operation gives result by scenario (i.e. over ensemble members)
        .quantile(q=q / 100, interpolation=interpolation)
        # second one gives result over all scenarios
        .quantile(q=q / 100, interpolation=interpolation)
    )

    return res


def _derive_auto_regressive_process_parameters(
    target, order, scenario_level="scenario", ensemble_member_level="ensemble_member"
):
    store = []
    for (scenario, ensemble_member), values in _loop_levels(
        target, (scenario_level, ensemble_member_level)
    ):
        parameters = AutoRegression1D().calibrate(values, order=order)
        parameters["scenario"] = scenario
        parameters["ensemble_member"] = ensemble_member
        store.append(parameters)

    store = pd.DataFrame(store).set_index(["scenario", "ensemble_member"])

    def _axis_mean(inp):
        return inp.apply(np.mean, axis=0)

    res = (
        store.groupby("scenario")
        # first operation gives result by scenario (i.e. over ensemble members)
        .apply(_axis_mean)
        # second one gives result over all scenarios
        .apply(np.mean, axis=0).to_dict()
    )

    return res


def calibrate_auto_regressive_process_multiple_scenarios_and_ensemble_members(
    target,
    maxlag=12,
    ic="bic",
):
    ar_order = _select_auto_regressive_process_order(target, maxlag, ic)
    ar_params = _derive_auto_regressive_process_parameters(target, ar_order)

    return ar_params


def calibrate_auto_regressive_process_with_spatially_correlated_errors_multiple_scenarios_and_ensemble_members(
    target,
    localisation_radii,
    max_cross_validation_iterations=30,
    gridpoint_dim_name="gridpoint",
):
    gridpoint_autoregression_parameters = {
        gridpoint: _derive_auto_regressive_process_parameters(gridpoint_vals, order=1)
        for gridpoint, gridpoint_vals in target.groupby("gridpoint")
    }

    gaspari_cohn_correlation_matrices = calculate_gaspari_cohn_correlation_matrices(
        target.lat,
        target.lon,
        localisation_radii,
    )

    localised_empirical_covariance_matrix = _calculate_localised_empirical_covariance_matrix(
        target,
        localisation_radii,
        gaspari_cohn_correlation_matrices,
        max_cross_validation_iterations,
        gridpoint_dim_name=gridpoint_dim_name,
    )

    gridpoint_autoregression_coeffcients = np.hstack([v["lag_coefficients"] for v in gridpoint_autoregression_parameters.values()])

    localised_empirical_covariance_matrix_with_ar1_errors = (
        (1 - gridpoint_autoregression_coeffcients ** 2)
        * localised_empirical_covariance_matrix
    )

    return localised_empirical_covariance_matrix_with_ar1_errors


def _calculate_localised_empirical_covariance_matrix(
    target,
    localisation_radii,
    gaspari_cohn_correlation_matrices,
    max_cross_validation_iterations,
    gridpoint_dim_name="gridpoint",
):
    dims_to_flatten = [d for d in target.dims if d != gridpoint_dim_name]
    target_flattened, stack_coord_name = _flatten(target, dims_to_flatten)
    target_flattened = target_flattened.transpose(stack_coord_name, gridpoint_dim_name)

    number_samples = target_flattened[stack_coord_name].shape[0]
    number_iterations = min([number_samples, max_cross_validation_iterations])

    # setup cross-validation stuff
    index_cross_validation_out = np.zeros([number_iterations, number_samples], dtype=bool)

    for i in range(number_iterations):
        index_cross_validation_out[i, i::max_cross_validation_iterations] = True

    # No idea what these are either
    log_likelihood_cross_validation_sum_max = -10000

    for lr in localisation_radii:
        log_likelihood_cross_validation_sum = 0

        for i in range(number_iterations):
            # extract folds (no idea why these are called folds)
            target_estimator = target_flattened.isel(**{stack_coord_name: ~index_cross_validation_out[i]}).values
            target_cross_validation = target_flattened.isel(**{stack_coord_name: index_cross_validation_out[i]}).values
            # selecting relevant weights goes in here

            empirical_covariance = np.cov(target_estimator, rowvar=False)
            # must be a better way to handle ensuring that the dimensions line up correctly (rather than
            # just cheating by using `.values`)
            empirical_covariance_localised = empirical_covariance * gaspari_cohn_correlation_matrices[lr].values

            # calculate likelihood of cross validation samples
            log_likelihood_cross_validation_samples = scipy.stats.multivariate_normal.logpdf(
                target_cross_validation,
                mean=np.zeros(gaspari_cohn_correlation_matrices[lr].shape[0]),
                cov=empirical_covariance_localised,
                allow_singular=True,
            )
            log_likelihood_cross_validation_samples_weighted_sum = np.average(
                log_likelihood_cross_validation_samples,
                # weights=wgt_scen_eq_cv # TODO: weights handling
            ) * log_likelihood_cross_validation_samples.shape[0]

            # add to full sum over all folds
            log_likelihood_cross_validation_sum += log_likelihood_cross_validation_samples_weighted_sum

        if log_likelihood_cross_validation_sum > log_likelihood_cross_validation_sum_max:
            log_likelihood_cross_validation_sum_max = log_likelihood_cross_validation_sum
        else:
            # experience tells us that once we start selecting large localisation radii, performance
            # will not improve ==> break (reduces computational effort and number of singular matrices
            # encountered)
            break

    # TODO: replace print with logging
    print(f"Selected localisation radius: {lr}")

    empirical_covariance = np.cov(target_flattened.values, rowvar=False)
    empirical_covariance_localised = empirical_covariance * gaspari_cohn_correlation_matrices[lr].values

    return empirical_covariance_localised
