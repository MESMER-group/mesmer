import pathlib

import pytest
import xarray as xr

import mesmer
import mesmer.mesmer_x

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
            #marks=pytest.mark.slow,
        ),
        pytest.param(
            "ssp585",
            "tasmax",
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr1_2ndfit",
            True,
            False,
            #marks=pytest.mark.slow,
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

    # make global mean
    # global_mean_dt = map_over_subtree(mesmer.weighted.global_mean)
    tas_glob_mean_hist = mesmer.weighted.global_mean(tas_hist)
    tas_glob_mean_ssp585 = mesmer.weighted.global_mean(tas_ssp585)

    # load target data
    path_target = cmip6_data_path / target_name / "ann" / "g025"

    fN_hist = path_target / f"{target_name}_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_target / f"{target_name}_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    targ_hist = xr.open_dataset(fN_hist, decode_times=time_coder)
    targ_ssp585 = xr.open_dataset(fN_ssp585, decode_times=time_coder)

    # stack target data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds, stack_dim="gridpoint")
        return ds

    # mask_and_stack_dt = map_over_subtree(mask_and_stack)
    targ_stacked_hist = mask_and_stack(targ_hist, threshold_land=THRESHOLD_LAND)
    targ_stacked_ssp585 = mask_and_stack(targ_ssp585, threshold_land=THRESHOLD_LAND)

    # switching to stacked datasets as used for linear regression
    tmp = xr.DataTree.from_dict(
        {
            "historical": tas_glob_mean_hist.expand_dims({"member": ["r1i1p1f1"]}),
            "ssp585": tas_glob_mean_ssp585.expand_dims({"member": ["r1i1p1f1"]}),
        }
    )
    dt_pred = xr.DataTree.from_dict({"tas": tmp}, name="predictors")
    dt_targ = xr.DataTree.from_dict(
        {
            "historical": targ_stacked_hist.expand_dims({"member": ["r1i1p1f1"]}),
            "ssp585": targ_stacked_ssp585.expand_dims({"member": ["r1i1p1f1"]}),
        },
        name=target_name,
    )

    # weights
    weights = mesmer.mesmer_x.get_weights_density(
        pred_data=dt_pred,
        predictor="tas",
        targ_data=dt_targ,
        target=target_name,
        dims=("member", "time"),
    )

    # stacking
    stacked_pred, stacked_targ, stacked_weights = (
        mesmer.core.datatree.broadcast_and_stack_scenarios(
            predictors=dt_pred,
            target=dt_targ,
            weights=weights,
            )
    )

    # declaring analytical form of the conditional distribution
    expression_fit = mesmer.mesmer_x.Expression(expr, expr_name)

    # preparing tests that will be used for first guess and training
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expr_fit=expression_fit,
        threshold_min_proba=1.0e-9,
        boundaries_params=None,
        boundaries_coeffs=None,
    )

    # preparing optimizers that will be used for first guess and training
    optim_mx = mesmer.mesmer_x.distrib_optimizer(
        expr_fit=expression_fit,
        class_tests=tests_mx,
        options_optim=None,
        options_solver=None,
    )

    # preparing first guess
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expr_fit=expression_fit,
        class_tests=tests_mx,
        class_optim=optim_mx,
        first_guess=None,
        func_first_guess=None,
    )
    coeffs_fg = fg_mx.find_fg(
        predictors=stacked_pred,
        target=stacked_targ,
        weights=stacked_weights,
        dim="sample",
    )

    # training the conditional distribution
    train_mx = mesmer.mesmer_x.ConditionalDistribution(
        expr_fit=expression_fit, class_tests=tests_mx, class_optim=optim_mx
    )
    # first round
    transform_params = train_mx.fit(
        predictors=stacked_pred,
        target=stacked_targ,
        first_guess=coeffs_fg,
        weights=stacked_weights,
        dim="sample",
    )

    # second round if necessary
    if option_2ndfit:
        transform_params = train_mx.fit(
            predictors=stacked_pred,
            target=stacked_targ,
            first_guess=transform_params,
            weights=stacked_weights,
            dim="sample",
            option_smooth_coeffs=True,
            r_gasparicohn=500,
        )

    # probability integral transform on non-stacked data for AR(1) process
    pit = mesmer.mesmer_x.probability_integral_transform(
        expr_start=expr,
        coeffs_start=transform_params,
        expr_end="norm(loc=0, scale=1)",
        coeffs_end=None,
    )
    transf_target = pit.transform(
        data=dt_targ, target_name=target_name, preds_start=dt_pred, preds_end=None
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
        lon=stacked_targ.lon, lat=stacked_targ.lat
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
        transform_params.to_netcdf(distrib_file)
        local_ar_params.to_netcdf(local_ar_file)
        localized_ecov.to_netcdf(localized_ecov_file)
        pytest.skip("Updated param files.")

    else:
        # load the parameters
        expected_transform_params = xr.open_dataset(distrib_file)
        xr.testing.assert_allclose(transform_params, expected_transform_params)

        expected_local_ar_params = xr.open_dataset(local_ar_file)
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
