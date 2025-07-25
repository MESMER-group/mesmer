import filefisher
import numpy as np
import pytest
import xarray as xr

import mesmer


def find_hist_proj_simulations(
    variable,
    model,
    scenario,
    time_res="ann",
    member="*",
    resolution="g025",
):

    cmip6_data_path = mesmer.example_data.cmip6_ng_path(relative=True)

    cmip_filefinder = filefisher.FileFinder(
        path_pattern=cmip6_data_path / "{variable}/{time_res}/{resolution}",
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member}_{resolution}.nc",
    )

    fc_scens = cmip_filefinder.find_files(
        variable=variable,
        scenario=scenario,
        model=model,
        resolution=resolution,
        time_res=time_res,
        member=member,
    )

    # get the historical members that are also in the future scenarios, but only once
    unique_scen_members_y = fc_scens.df.member.unique()

    fc_hist = cmip_filefinder.find_files(
        variable=variable,
        scenario="historical",
        model=model,
        resolution=resolution,
        time_res=time_res,
        member=unique_scen_members_y,
    )

    fc_all = fc_hist.concat(fc_scens)

    return fc_all


def load_data(filecontainer):

    out = xr.DataTree()

    scenarios = filecontainer.df.scenario.unique().tolist()

    # load data for each scenario
    for scen in scenarios:
        files = filecontainer.search(scenario=scen)

        # load all members for a scenario
        members = []
        for fN, meta in files.items():
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds = xr.open_dataset(fN, decode_times=time_coder)
            # drop unnecessary variables
            ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
            # assign member-ID as coordinate
            ds = ds.assign_coords({"member": meta["member"]})
            members.append(ds)

        # create a Dataset that holds each member along the member dimension
        scen_data = xr.concat(members, dim="member")
        # put the scenario dataset into the DataTree
        out[scen] = xr.DataTree(scen_data)

    return out


@pytest.mark.slow
def test_calibrate_mesmer_m(test_data_root_dir, update_expected_files):
    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    # LOCALISATION_RADII = list(range(1250, 6251, 250)) + list(range(6500, 8501, 500))
    # restrict radii for faster tests
    LOCALISATION_RADII = list(range(5750, 6251, 250)) + list(range(6500, 8001, 500))

    esm = "IPSL-CM6A-LR"
    scenario = "ssp585"
    member = "r1i1p1f1"

    # define paths and load data
    test_path = test_data_root_dir / "output" / "tas" / "mon" / "test-params"

    # load annual data
    fc_all_y = find_hist_proj_simulations(
        variable="tas", scenario=scenario, model=esm, time_res="ann", member=member
    )
    tas_y_orig = load_data(fc_all_y)

    # load monthly data
    fc_all_m = find_hist_proj_simulations(
        variable="tas", scenario=scenario, model=esm, time_res="mon", member=member
    )
    tas_m_orig = load_data(fc_all_m)

    # data preprocessing
    tas_anoms_y = mesmer.anomaly.calc_anomaly(
        tas_y_orig, reference_period=REFERENCE_PERIOD
    )
    tas_anoms_m = mesmer.anomaly.calc_anomaly(
        tas_m_orig, reference_period=REFERENCE_PERIOD
    )

    # create local gridded tas data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    tas_stacked_y = mask_and_stack(tas_anoms_y, threshold_land=THRESHOLD_LAND)
    tas_stacked_m = mask_and_stack(tas_anoms_m, threshold_land=THRESHOLD_LAND)

    tas_stacked_y = xr.combine_by_coords(
        [dt.ds for dt in tas_stacked_y.values()],
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
    ).squeeze(drop=True)

    tas_stacked_m = xr.combine_by_coords(
        [dt.ds for dt in tas_stacked_m.values()],
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
    ).squeeze(drop=True)

    # fit harmonic model
    harmonic_model_fit = mesmer.stats.fit_harmonic_model(
        tas_stacked_y.tas, tas_stacked_m.tas
    )

    # train power transformer
    yj_transformer = mesmer.stats.YeoJohnsonTransformer("logistic")
    pt_coefficients = yj_transformer.fit(
        tas_stacked_y.tas,
        harmonic_model_fit.residuals,
    )
    transformed_hm_resids = yj_transformer.transform(
        tas_stacked_y.tas, harmonic_model_fit.residuals, pt_coefficients
    )

    # fit cyclo-stationary AR(1) process
    ar1_fit = mesmer.stats.fit_auto_regression_monthly(
        transformed_hm_resids.transformed
    )

    # find localized empirical covariance
    geodist = mesmer.geospatial.geodist_exact(tas_stacked_y.lon, tas_stacked_y.lat)

    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist, localisation_radii=LOCALISATION_RADII
    )

    weights = xr.ones_like(ar1_fit.residuals.isel(gridcell=0))
    weights.name = "weights"

    localized_ecov = mesmer.stats.find_localized_empirical_covariance_monthly(
        ar1_fit.residuals, weights, phi_gc_localizer, "time", k_folds=30
    )

    # we need to get the original time coordinate to be able to validate our results
    m_time = tas_stacked_m.time.rename("monthly_time")

    PARAM_FILEFINDER = filefisher.FileFinder(
        path_pattern=test_path / "{module}/",
        file_pattern="params_{module}_{variable}_{esm}_{scen}.nc",
    )

    scen_str = scenario

    keys = {"esm": esm, "scen": scen_str, "variable": "tas"}

    local_hm_file = PARAM_FILEFINDER.create_full_name(keys, module="harmonic-model")
    local_pt_file = PARAM_FILEFINDER.create_full_name(keys, module="power-transformer")
    local_ar_file = PARAM_FILEFINDER.create_full_name(keys, module="local-variability")
    localized_ecov_file = PARAM_FILEFINDER.create_full_name(keys, module="covariance")
    time_file = PARAM_FILEFINDER.create_full_name(keys, module="monthly-time")

    # save params
    if update_expected_files:
        # drop unnecessary variables
        harmonic_model_fit = harmonic_model_fit.drop_vars(["residuals", "time"])
        ar1_fit = ar1_fit.drop_vars(["residuals", "time"])

        # save
        harmonic_model_fit.to_netcdf(local_hm_file)
        pt_coefficients.to_netcdf(local_pt_file)
        ar1_fit.to_netcdf(local_ar_file)
        localized_ecov.to_netcdf(localized_ecov_file)
        m_time.to_netcdf(time_file)

        pytest.skip("Updated param files.")

    # testing
    else:
        assert_params_allclose(
            harmonic_model_fit,
            pt_coefficients,
            ar1_fit,
            localized_ecov,
            m_time,
            local_hm_file,
            local_pt_file,
            local_ar_file,
            localized_ecov_file,
            time_file,
        )


def assert_params_allclose(
    harmonic_model_fit,
    pt_coefficients,
    ar1_fit,
    localized_ecov,
    m_time,
    local_hm_file,
    local_pt_file,
    local_ar_file,
    localized_ecov_file,
    time_file,
):
    # test params

    # load expected values
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    expected_hm_params = xr.open_dataset(local_hm_file, decode_times=time_coder)
    expected_pt_params = xr.open_dataset(local_pt_file, decode_times=time_coder)
    expected_AR1_params = xr.open_dataset(local_ar_file, decode_times=time_coder)
    expected_localized_ecov = xr.open_dataset(
        localized_ecov_file, decode_times=time_coder
    )
    expected_m_time = xr.open_dataset(time_file, decode_times=time_coder)

    # the following parameters should be exactly the same
    exact_exp_params = xr.merge(
        [
            expected_hm_params.selected_order,
            expected_localized_ecov.localization_radius,
            expected_m_time.monthly_time,
        ]
    )
    exact_cal_params = xr.merge(
        [
            harmonic_model_fit.selected_order,
            localized_ecov.localization_radius,
            m_time,
        ]
    )

    xr.testing.assert_equal(exact_exp_params, exact_cal_params)

    # compare the rest
    # using numpy because it outputs the differences and how many values are off

    # the tols are set to the best we can do
    # NOTE: it is always rather few values that are off
    np.testing.assert_allclose(
        expected_hm_params.coeffs,
        harmonic_model_fit.coeffs,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        expected_pt_params.lambda_coeffs, pt_coefficients, atol=1.0e-4, rtol=1.5e-4
    )
    np.testing.assert_allclose(
        expected_AR1_params.slope,
        ar1_fit.slope,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        expected_AR1_params.intercept,
        ar1_fit.intercept,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        localized_ecov.localized_covariance,
        localized_ecov.localized_covariance,
    )
