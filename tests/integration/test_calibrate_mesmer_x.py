import pathlib

import numpy as np
import pytest
import xarray as xr

import mesmer
from mesmer.mesmer_x import (
    ConditionalDistribution,
    ConditionalDistributionOptions,
    Expression,
    ProbabilityIntegralTransform,
    find_first_guess,
    get_weights_density,
)

# TODO: extend to more scenarios and members
# TODO: extend predictors


@pytest.mark.parametrize(
    (
        "scenario",
        "target_name",
        "expr",
        "expr_name",
        "option_2ndfit",
        "update_expected_files",
    ),
    [
        pytest.param(
            "ssp585",
            "tasmax",
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr1",
            False,
            False,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "ssp585",
            "tasmax",
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr1_2ndfit",
            True,
            False,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_calibrate_mesmer_x(
    scenario,
    target_name,
    expr,
    expr_name,
    option_2ndfit,
    test_data_root_dir,
    update_expected_files,
):
    # set some configuration parameters
    THRESHOLD_LAND = 1 / 3
    esm = "IPSL-CM6A-LR"

    # TODO: replace with filefinder later
    # load data
    TEST_DATA_PATH = pathlib.Path(test_data_root_dir)
    TEST_PATH = (
        TEST_DATA_PATH / "output" / target_name / "one_scen_one_ens" / "test-params"
    )
    cmip6_data_path = mesmer.example_data.cmip6_ng_path()

    # load predictor data
    path_tas = cmip6_data_path / "tas" / "ann" / "g025"

    fN_hist = path_tas / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_tas / f"tas_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    tas_hist = xr.open_dataset(fN_hist, decode_times=time_coder).drop_vars(
        ["height", "file_qf", "time_bnds"]
    )
    tas_ssp585 = xr.open_dataset(fN_ssp585, decode_times=time_coder).drop_vars(
        ["height", "file_qf", "time_bnds"]
    )

    tas = xr.DataTree.from_dict(
        {
            "historical": tas_hist,
            "ssp585": tas_ssp585,
        }
    )

    # make global mean
    # global_mean_dt = map_over_subtree(mesmer.weighted.global_mean)
    tas_glob_mean = mesmer.weighted.global_mean(tas)

    # load target data
    path_target = cmip6_data_path / target_name / "ann" / "g025"

    fN_hist = path_target / f"{target_name}_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_target / f"{target_name}_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    targ_hist = xr.open_dataset(fN_hist, decode_times=time_coder)
    targ_ssp585 = xr.open_dataset(fN_ssp585, decode_times=time_coder)
    # make sure times align
    targ_hist["time"] = tas_hist["time"]
    targ_ssp585["time"] = tas_ssp585["time"]

    target_data = xr.DataTree.from_dict(
        {
            "historical": targ_hist,
            "ssp585": targ_ssp585,
        }
    )

    # stack target data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds, stack_dim="gridpoint")
        return ds

    # mask_and_stack_dt = map_over_subtree(mask_and_stack)
    target_data = mask_and_stack(target_data, threshold_land=THRESHOLD_LAND)
    pred_data = xr.DataTree.from_dict({"tas": tas_glob_mean})

    # stack datasets
    # weights
    weights = get_weights_density(
        pred_data=pred_data,
        predictor="tas",
        targ_data=target_data,
        target=target_name,
        dims=("member", "time"),
    )

    # stacking
    stacked_pred, stacked_targ, stacked_weights = (
        mesmer.core.datatree.broadcast_and_stack_scenarios(
            predictors=pred_data,
            target=target_data,
            weights=weights,
            member_dim=None,
        )
    )

    # declaring analytical form of the conditional distribution
    expression = Expression(expr, expr_name)
    distrib = ConditionalDistribution(
        expression, ConditionalDistributionOptions(expression)
    )

    # preparing first guess
    coeffs_fg = find_first_guess(
        distrib,
        predictors=stacked_pred,
        target=stacked_targ.tasmax,
        weights=stacked_weights.weight,
        first_guess=None,
    )

    # training the conditional distribution
    distrib.fit(
        predictors=stacked_pred,
        target=stacked_targ.tasmax,
        first_guess=coeffs_fg,
        weights=stacked_weights.weight,
    )
    transform_coeffs = distrib.coefficients

    # second round if necessary
    if option_2ndfit:
        distrib.fit(
            predictors=stacked_pred,
            target=stacked_targ.tasmax,
            first_guess=transform_coeffs,
            weights=stacked_weights.weight,
            smooth_coeffs=True,
            r_gasparicohn=500,
        )
        transform_coeffs = distrib.coefficients

    # probability integral transform on non-stacked data for AR(1) process
    target_expr = Expression("norm(loc=0, scale=1)", "standard_normal")
    target_distrib = ConditionalDistribution(
        target_expr,
        ConditionalDistributionOptions(target_expr),
    )

    pit = ProbabilityIntegralTransform(distrib, target_distrib)
    transf_target = pit.transform(
        data=target_data,
        target_name=target_name,
        preds_orig=pred_data,
    )

    # training of auto-regression with spatially correlated innovations
    local_ar_params = mesmer.stats.fit_auto_regression_scen_ens(
        transf_target,
        ens_dim="member",
        dim="time",
        lags=1,
    )

    # estimate covariance matrix
    # prep distance matrix
    geodist = mesmer.core.geospatial.geodist_exact(
        lon=target_data["historical"].lon, lat=target_data["historical"].lat
    )
    # prep localizer
    LOCALISATION_RADII = range(1750, 2001, 250)
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist=geodist, localisation_radii=LOCALISATION_RADII
    )

    # TODO: should we both for MESMER and for MESMER-X remove the
    # residuals from the AR(1) process before calculating the covariance?
    # is that we is happening in 'adjust_covariance_ar1'?
    # TODO: using here weights from MESMER-X. I noticed that it affects the
    # calculation in find_localized_empirical_covariance. Need to solve that.
    localized_ecov = mesmer.stats.find_localized_empirical_covariance(
        data=stacked_targ[target_name],
        weights=stacked_weights.weight,
        localizer=phi_gc_localizer,
        dim="sample",
        k_folds=30,
    )

    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    file_end = f"{target_name}_{expr_name}_{esm}_{scenario}"
    distrib_file = TEST_PATH / "distrib" / f"params_transform_distrib_{file_end}.nc"
    local_ar_file = TEST_PATH / "local_variability" / f"params_local_AR_{file_end}.nc"
    localized_ecov_file = (
        TEST_PATH / "local_variability" / f"params_localized_ecov_{file_end}.nc"
    )

    if update_expected_files:
        # save the parameters
        distrib.to_netcdf(distrib_file)
        local_ar_params.to_netcdf(local_ar_file)
        localized_ecov.to_netcdf(localized_ecov_file)
        pytest.skip("Updated param files.")

    else:
        # load the parameters
        expected_transform_params = xr.open_dataset(distrib_file)

        xr.testing.assert_allclose(
            transform_coeffs, expected_transform_params, rtol=1.5e-5
        )

        expected_local_ar_params = xr.open_dataset(local_ar_file)

        np.testing.assert_allclose(
            local_ar_params["intercept"].values,
            expected_local_ar_params["intercept"].values,
            atol=1e-7,
            rtol=1e-05,
        )

        xr.testing.assert_allclose(
            local_ar_params["intercept"],
            expected_local_ar_params["intercept"],
            atol=1e-7,
        )
        xr.testing.assert_allclose(
            local_ar_params["coeffs"], expected_local_ar_params["coeffs"]
        )
        xr.testing.assert_allclose(
            local_ar_params["variance"], expected_local_ar_params["variance"]
        )
        xr.testing.assert_equal(
            local_ar_params["nobs"], expected_local_ar_params["nobs"]
        )

        expected_localized_ecov = xr.open_dataset(localized_ecov_file)
        xr.testing.assert_allclose(localized_ecov, expected_localized_ecov)
