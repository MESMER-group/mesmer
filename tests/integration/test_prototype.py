import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from statsmodels.tsa.arima_process import ArmaProcess

from mesmer.calibrate_mesmer.train_lt import train_lt
from mesmer.calibrate_mesmer.train_lv import train_lv
from mesmer.calibrate_mesmer.train_gv import train_gv
from mesmer.prototype.calibrate import LinearRegression
from mesmer.prototype.calibrate_multiple import (
    calibrate_auto_regressive_process_multiple_scenarios_and_ensemble_members,
    flatten_predictors_and_target,
)
from mesmer.prototype.utils import calculate_gaspari_cohn_correlation_matrices


class _MockConfig:
    def __init__(
        self,
        method_lt="OLS",
        method_lv="OLS_AR1_sci",
        method_gv="AR",
        separate_gridpoints=True,
        weight_scenarios_equally=True,
        target_variable="tas",
        cross_validation_max_iterations=30,
    ):
        self.methods = {}
        self.methods[target_variable] = {}
        self.methods[target_variable]["lt"] = method_lt
        self.methods[target_variable]["lv"] = method_lv
        self.methods[target_variable]["gv"] = method_gv

        # this has to be set but isn't actually used
        self.preds = {}
        self.preds[target_variable] = {}
        self.preds[target_variable]["gv"] = None

        self.method_lt_each_gp_sep = separate_gridpoints
        self.wgt_scen_tr_eq = weight_scenarios_equally

        self.max_iter_cv = cross_validation_max_iterations


def _do_legacy_run_train_lt(
    emulator_tas,
    emulator_tas_squared,
    emulator_hfds,
    global_variability,
    esm_tas,
    cfg,
):
    preds_legacy = {}
    for name, vals in (
        ("gttas", emulator_tas),
        ("gttas2", emulator_tas_squared),
        ("gthfds", emulator_hfds),
        ("gvtas", global_variability),
    ):
        preds_legacy[name] = {}
        for scenario, vals_scen in vals.groupby("scenario"):
            # we have to force this to have an extra dimension, run, for legacy
            # to work although this really isn't how it should be because it
            # means you have some weird reshaping to do at a really low level
            preds_legacy[name][scenario] = vals_scen.dropna(dim="time").values[
                np.newaxis, :
            ]

    targs_legacy = {}
    for name, vals in (("tas", esm_tas),):
        targs_legacy[name] = {}
        for scenario, vals_scen in vals.groupby("scenario"):
            # we have to force this to have an extra dimension, run, for legacy
            # to work although this really isn't how it should be
            # order of dimensions is very important for legacy too
            targs_legacy[name][scenario] = vals_scen.T.dropna(dim="time").values[
                np.newaxis, :, :
            ]

    res_legacy = train_lt(
        preds_legacy,
        targs_legacy,
        esm="esm_name",
        cfg=cfg,
        save_params=False,
    )

    return res_legacy


def test_prototype_train_lt():
    time = [1850, 1950, 2014, 2015, 2050, 2100, 2300]
    scenarios = ["hist", "ssp126"]

    pred_dims = ["scenario", "time"]
    pred_coords = dict(
        time=time,
        scenario=scenarios,
    )

    emulator_tas = xr.DataArray(
        np.array(
            [
                [0, 0.5, 1, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 1.1, 1.5, 1.4, 1.2],
            ]
        ),
        dims=pred_dims,
        coords=pred_coords,
    )
    emulator_tas_squared = emulator_tas ** 2
    global_variability = xr.DataArray(
        np.array(
            [
                [-0.1, 0.1, 0.03, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 0.04, 0.2, -0.03, 0.0],
            ]
        ),
        dims=pred_dims,
        coords=pred_coords,
    )
    emulator_hfds = xr.DataArray(
        np.array(
            [
                [0.5, 1.5, 2.0, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 2.1, 2.0, 1.5, 0.4],
            ]
        ),
        dims=pred_dims,
        coords=pred_coords,
    )

    # we wouldn't actually start like this, but we'd write a utils function
    # to simply go from lat-lon to gridpoint and back
    targ_dims = ["scenario", "gridpoint", "time"]
    targ_coords = dict(
        time=time,
        scenario=scenarios,
        gridpoint=[0, 1],
        lat=(["gridpoint"], [-60, 60]),
        lon=(["gridpoint"], [120, 240]),
    )
    esm_tas = xr.DataArray(
        np.array(
            [
                [
                    [0.6, 1.6, 2.6, np.nan, np.nan, np.nan, np.nan],
                    [0.43, 1.13, 2.21, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, 2.11, 2.01, 1.54, 1.22],
                    [np.nan, np.nan, np.nan, 2.19, 2.04, 1.53, 1.21],
                ],
            ]
        ),
        dims=targ_dims,
        coords=targ_coords,
    )

    res_legacy = _do_legacy_run_train_lt(
        emulator_tas,
        emulator_tas_squared,
        emulator_hfds,
        global_variability,
        esm_tas,
        cfg=_MockConfig(),
    )

    (
        predictors_flattened,
        target_flattened,
        stack_coord_name,
    ) = flatten_predictors_and_target(
        predictors={
            "emulator_tas": emulator_tas,
            "emulator_tas_squared": emulator_tas_squared,
            "emulator_hfds": emulator_hfds,
            "global_variability": global_variability,
        },
        target=esm_tas,
    )

    res_updated = LinearRegression().calibrate(
        target_flattened,
        predictors_flattened,
        stack_coord_name,
    )

    # check that calibrated parameters match for each predictor variable
    for updated_name, legacy_vals in (
        ("emulator_tas", res_legacy[0]["coef_gttas"]["tas"]),
        ("emulator_tas_squared", res_legacy[0]["coef_gttas2"]["tas"]),
        ("emulator_hfds", res_legacy[0]["coef_gthfds"]["tas"]),
        ("global_variability", res_legacy[1]["coef_gvtas"]["tas"]),
        ("intercept", res_legacy[0]["intercept"]["tas"]),
    ):
        npt.assert_allclose(res_updated.sel(predictor=updated_name), legacy_vals)


def _do_legacy_run_train_gv(
    esm_tas_global_variability,
    cfg,
):
    targs_legacy = {}
    var_name = "tas"

    targs_legacy = {}
    for scenario, vals_scen in esm_tas_global_variability.groupby("scenario"):
        targs_legacy[scenario] = (
            vals_scen.T.dropna(dim="time").transpose("ensemble_member", "time").values
        )

    res_legacy = train_gv(
        targs_legacy,
        targ=var_name,
        esm="esm_name",
        cfg=cfg,
        save_params=False,
        max_lag=2,
    )

    return res_legacy


@pytest.mark.parametrize(
    "ar",
    (
        [1, 0.5, 0.3],
        [1, 0.5, 0.3, 0.3, 0.7],
        [0.9, 1, 0.2, -0.1],
    ),
)
def test_prototype_train_gv(ar):
    time_history = range(1850, 2014 + 1)
    time_scenario = range(2015, 2100 + 1)
    time = list(time_history) + list(time_scenario)

    magnitude = np.array([0.1])

    scenarios = ["hist", "ssp126"]

    targ_dims = ["scenario", "ensemble_member", "time"]
    targ_coords = dict(
        time=time,
        scenario=scenarios,
        ensemble_member=["r1i1p1f1", "r2i1p1f1"],
    )
    esm_tas_global_variability = xr.DataArray(
        np.array(
            [
                [
                    np.concatenate(
                        [
                            ArmaProcess(ar, magnitude).generate_sample(
                                nsample=len(time_history)
                            ),
                            np.nan * np.zeros(len(time_scenario)),
                        ]
                    ),
                    np.concatenate(
                        [
                            ArmaProcess(ar, magnitude).generate_sample(
                                nsample=len(time_history)
                            ),
                            np.nan * np.zeros(len(time_scenario)),
                        ]
                    ),
                ],
                [
                    np.concatenate(
                        [
                            np.nan * np.zeros(len(time_history)),
                            ArmaProcess(ar, magnitude).generate_sample(
                                nsample=len(time_scenario)
                            ),
                        ]
                    ),
                    np.concatenate(
                        [
                            np.nan * np.zeros(len(time_history)),
                            ArmaProcess(ar, magnitude).generate_sample(
                                nsample=len(time_scenario)
                            ),
                        ]
                    ),
                ],
            ]
        ),
        dims=targ_dims,
        coords=targ_coords,
    )

    res_legacy = _do_legacy_run_train_gv(
        esm_tas_global_variability,
        cfg=_MockConfig(),
    )

    res_updated = (
        calibrate_auto_regressive_process_multiple_scenarios_and_ensemble_members(
            esm_tas_global_variability,
            maxlag=2,
        )
    )

    for key, comparison in (
        ("intercept", res_legacy["AR_int"]),
        ("lag_coefficients", res_legacy["AR_coefs"]),
        ("standard_innovations", res_legacy["AR_std_innovs"]),
    ):
        npt.assert_allclose(res_updated[key], comparison)


def _do_legacy_run_train_lv(
    esm_tas_residual_local_variability,
    localisation_radii,
    cfg,
):
    targs_legacy = {"tas": {}}
    for scenario, vals_scen in esm_tas_residual_local_variability.groupby("scenario"):
        targs_legacy["tas"][scenario] = (
            vals_scen.T.dropna(dim="time").transpose("ensemble_member", "time", "gridpoint").values
        )

    gaspari_cohn_correlation_matrices = calculate_gaspari_cohn_correlation_matrices(
        latitudes=esm_tas_residual_local_variability.lat,
        longitudes=esm_tas_residual_local_variability.lon,
        localisation_radii=localisation_radii,
    )
    gaspari_cohn_correlation_matrices = {
        k: v.values for k, v in gaspari_cohn_correlation_matrices.items()
    }
    aux = {"phi_gc": gaspari_cohn_correlation_matrices}

    res_legacy = train_lv(
        preds={},
        targs=targs_legacy,
        esm="test",
        cfg=cfg,
        save_params=False,
        aux=aux,
        params_lv={},  # unclear why this is passed in
    )

    return res_legacy


# how train_lv works:
# 1. AR1 process for each individual gridpoint (use calibrate_auto_regressive_process_multiple_scenarios_and_ensemble_members)
# 2. find localised empirical covariance matrix
# 3. combine AR1 and localised empirical covariance matrix to get localised empirical covariance matrix
#    of innovations (i.e. errors) which can be used for later draws (again, with a bit of custom code
#    that feels like it should really be done using an existing library)


def test_prototype_train_lv():
    # input is residual local variability we want to reproduce (for Leah, residual
    # means after removing local trend and local variability due to global variability
    # but it could be whatever in reality)

    # also input the phi gc stuff (that's part of the calibration but doesn't go in the train
    # lv function for some reason --> put it in calibrate_local_variability prototype function
    # although split that function to allow for pre-calculating distance between points in future
    # as a performance boost)

    localisation_radii = np.arange(700, 2000, 1000)

    # see how much code I can reuse for the AR1 calibration
    time_history = range(1850, 2014 + 1)
    time_scenario = range(2015, 2100 + 1)
    time = list(time_history) + list(time_scenario)
    scenarios = ["hist", "ssp126"]

    # we wouldn't actually start like this, but we'd write a utils function
    # to simply go from lat-lon to gridpoint and back
    targ_dims = ["scenario", "ensemble_member", "gridpoint", "time"]
    targ_coords = dict(
        time=time,
        scenario=scenarios,
        ensemble_member=["r1i1p1f1", "r2i1p1f1"],
        gridpoint=[0, 1],
        lat=(["gridpoint"], [-60, 60]),
        lon=(["gridpoint"], [120, 240]),
    )

    ar = np.array([1])
    magnitude = np.array([0.05])

    def _get_history_sample():
        return np.concatenate(
            [
                ArmaProcess(ar, magnitude).generate_sample(
                    nsample=len(time_history)
                ),
                np.nan * np.zeros(len(time_scenario)),
            ]
        )

    def _get_scenario_sample():
        return np.concatenate(
            [
                np.nan * np.zeros(len(time_history)),
                ArmaProcess(ar, magnitude).generate_sample(
                    nsample=len(time_scenario)
                ),
            ]
        )

    esm_tas_residual_local_variability = xr.DataArray(
        np.array(
            [
                [
                    [
                        _get_history_sample(),
                        _get_history_sample(),
                    ],
                    [
                        _get_history_sample(),
                        _get_history_sample(),
                    ]
                ],
                [
                    [
                        _get_scenario_sample(),
                        _get_scenario_sample(),
                    ],
                    [
                        _get_scenario_sample(),
                        _get_scenario_sample(),
                    ],
                ],
            ]
        ),
        dims=targ_dims,
        coords=targ_coords,
    )

    res_legacy = _do_legacy_run_train_lv(
        esm_tas_residual_local_variability,
        localisation_radii,
        cfg=_MockConfig(),
    )

    res_updated = (
        calibrate_auto_regressive_process_multiple_scenarios_and_ensemble_members(
            esm_tas_residual_local_variability,
            localisation_radii,
        )
    )

    for key, comparison in (
        ("intercept", res_legacy["AR_int"]),
        ("lag_coefficients", res_legacy["AR_coefs"]),
        ("standard_innovations", res_legacy["AR_std_innovs"]),
    ):
        npt.assert_allclose(res_updated[key], comparison)
# things that aren't tested well:
# - what happens if ensemble member and scenario don't actually make a coherent set
# - units (should probably be using dataset rather than dataarray for inputs and outputs?)
# - weights
