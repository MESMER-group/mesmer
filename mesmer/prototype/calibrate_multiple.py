import xarray as xr


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
