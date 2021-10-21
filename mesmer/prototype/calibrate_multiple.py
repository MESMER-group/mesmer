def calibrate_multiple_scenarios_and_ensemble_members(targets, predictors, calibration_class, calibration_kwargs, weighting_style):
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
