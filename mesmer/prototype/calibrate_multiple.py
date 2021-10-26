import numpy as np
import xarray as xr

from .calibrate import AutoRegression1DOrderSelection, AutoRegression1D


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
    order = []
    for _, scenario_vals in target.groupby(scenario_level):
        scenario_orders = []
        for _, em_vals in scenario_vals.groupby(ensemble_member_level):
            em_orders = AutoRegression1DOrderSelection().calibrate(
                em_vals, maxlag=maxlag, ic=ic
            )

            if em_orders is not None:
                scenario_orders.append(em_orders[-1])
            else:
                scenario_orders.append(0)

        if scenario_orders:
            order.append(
                np.percentile(scenario_orders, q=q, interpolation=interpolation)
            )
        else:
            order.append(0)

    res = int(np.percentile(order, q=q, interpolation=interpolation))

    return res


def _derive_auto_regressive_process_parameters(
    target, order, scenario_level="scenario", ensemble_member_level="ensemble_member"
):
    # I don't like the fact that I'm duplicating these loops, surely there is a better way
    parameters = {
        "intercept": [],
        "lag_coefficients": [],
        "standard_innovations": [],
    }
    for _, scenario_vals in target.groupby(scenario_level):
        scenario_parameters = {k: [] for k in parameters}
        for _, em_vals in scenario_vals.groupby(ensemble_member_level):
            em_parameters = AutoRegression1D().calibrate(em_vals, order=order)
            for k, v in em_parameters.items():
                scenario_parameters[k].append(v)

        scenario_parameters_average = {
            k: np.vstack(v).mean(axis=0) for k, v in scenario_parameters.items()
        }
        for k, v in scenario_parameters_average.items():
            parameters[k].append(v)

    parameters_average = {k: np.vstack(v).mean(axis=0) for k, v in parameters.items()}

    return parameters_average


def calibrate_auto_regressive_process_multiple_scenarios_and_ensemble_members(
    target,
    maxlag=12,
    ic="bic",
):
    ar_order = _select_auto_regressive_process_order(target, maxlag, ic)
    ar_params = _derive_auto_regressive_process_parameters(target, ar_order)

    return ar_params


def calibrate_multiple_scenarios_and_ensemble_members(
    targets, predictors, calibration_class, calibration_kwargs, weighting_style
):
    """
    Calibrate based on multiple scenarios and ensemble members per scenario

    Parameters
    ----------
    targets : xarray.DataArray
        Target variables to calibrate to. This should have a scenario and an ensemble-member dimension (you'd need a different function to prepare things such that the dimensions all worked)

    predictors : xarray.DataArray
        Predictor variables, as above re comments about prepping data

    calibration_class : mesmer.calibration.MesmerCalibrateBase
        Class to use for calibration

    calibration_kwargs : dict
        Keyword arguments to pass to the calibration class

    weighting_style : ["equal", "equal_scenario", "equal_ensemble_member"]
        How to weight combinations of scenarios and ensemble members. "equal" --> all scenario - ensemble member combination are treated equally. "equal_scenario" --> all scenarios get equal weight (so if a scenario has more ensemble members then each ensemble member gets less weight). "equal_ensemble_member" --> all ensemble members get the same weight (so if an ensemble member is reported for more than one scenario then each scenario gets less weight for that ensemble member)
    """
    # this function (or class) would handle all of the looping over scenarios
    # and ensemble members. It would call the calibration class on each
    # scenario and ensemble member and know how much weight to give each
    # before combining them
