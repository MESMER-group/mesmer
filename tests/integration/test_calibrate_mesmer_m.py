import pathlib

import pytest
import xarray as xr

import mesmer


def _load_data(*filenames):
    # TODO: extract to a 'official' helper function

    # NOTE: open_mfdataset is considerably slower...
    # ds = xr.open_mfdataset(
    #     filenames,
    #     combine="by_coords",
    #     use_cftime=True,
    #     combine_attrs="override",
    #     data_vars="minimal",
    #     compat="override",
    #     coords="minimal",
    #     drop_variables=["height", "file_qf"],
    # ).load()

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    load_opt = {"drop_variables": ["height", "file_qf"], "decode_times": time_coder}
    datasets = [xr.open_dataset(fN, **load_opt) for fN in filenames]

    ds = xr.combine_by_coords(
        datasets,
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
    )

    return ds


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

    # define paths and load data
    TEST_DATA_PATH = pathlib.Path(test_data_root_dir)
    TEST_PATH = TEST_DATA_PATH / "output" / "tas" / "mon" / "test-params"
    cmip6_data_path = mesmer.example_data.cmip6_ng_path()

    # load annual data
    path_tas_ann = cmip6_data_path / "tas" / "ann" / "g025"
    fN_hist_ann = path_tas_ann / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_proj_ann = path_tas_ann / f"tas_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"
    tas_y = _load_data(fN_hist_ann, fN_proj_ann)

    # load monthly data
    path_tas_mon = cmip6_data_path / "tas" / "mon" / "g025"
    fN_hist_mon = path_tas_mon / f"tas_mon_{esm}_historical_r1i1p1f1_g025.nc"
    fN_proj_mon = path_tas_mon / f"tas_mon_{esm}_{scenario}_r1i1p1f1_g025.nc"
    tas_m = _load_data(fN_hist_mon, fN_proj_mon)

    # data preprocessing
    ref_y = tas_y.sel(time=REFERENCE_PERIOD).mean("time", keep_attrs=True)
    ref_m = tas_m.sel(time=REFERENCE_PERIOD).mean("time", keep_attrs=True)

    tas_y = tas_y - ref_y
    tas_m = tas_m - ref_m

    # create local gridded tas data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    tas_stacked_y = mask_and_stack(tas_y, threshold_land=THRESHOLD_LAND)
    tas_stacked_m = mask_and_stack(tas_m, threshold_land=THRESHOLD_LAND)

    # fit harmonic model
    harmonic_model_fit = mesmer.stats.fit_harmonic_model(
        tas_stacked_y.tas, tas_stacked_m.tas
    )

    # train power transformer
    pt_coefficients = mesmer.stats.fit_yeo_johnson_transform(
        tas_stacked_y.tas,
        harmonic_model_fit.residuals,
    )
    transformed_hm_resids = mesmer.stats.yeo_johnson_transform(
        tas_stacked_y.tas, harmonic_model_fit.residuals, pt_coefficients
    )

    # fit cyclo-stationary AR(1) process
    AR1_fit = mesmer.stats.fit_auto_regression_monthly(
        transformed_hm_resids.transformed, time_dim="time"
    )

    # work out covariance matrix
    geodist = mesmer.geospatial.geodist_exact(tas_stacked_y.lon, tas_stacked_y.lat)

    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist, localisation_radii=LOCALISATION_RADII
    )

    weights = xr.ones_like(AR1_fit.residuals.isel(gridcell=0))
    weights.name = "weights"

    localized_ecov = mesmer.stats.find_localized_empirical_covariance_monthly(
        AR1_fit.residuals, weights, phi_gc_localizer, "time", 30
    )

    # we need to get the original time coordinate to be able to validate our results
    m_time = tas_stacked_m.time.rename("monthly_time")

    # save params
    if update_expected_files:
        # drop unnecessary variables
        harmonic_model_fit = harmonic_model_fit.drop_vars(["residuals", "time"])
        AR1_fit = AR1_fit.drop_vars(["residuals", "time"])

        # save
        harmonic_model_fit.to_netcdf(
            TEST_PATH
            / "harmonic_model"
            / f"params_harmonic_model_tas_{esm}_{scenario}.nc"
        )
        pt_coefficients.to_netcdf(
            TEST_PATH
            / "power_transformer"
            / f"params_power_transformer_tas_{esm}_{scenario}.nc"
        )
        AR1_fit.to_netcdf(
            TEST_PATH / "local_variability" / f"params_AR1_tas_{esm}_{scenario}.nc"
        )
        localized_ecov.to_netcdf(
            TEST_PATH
            / "local_variability"
            / f"params_localized_ecov_tas_{esm}_{scenario}.nc"
        )
        m_time.to_netcdf(
            TEST_PATH / "time" / f"params_monthly_time_tas_{esm}_{scenario}.nc"
        )
        pytest.skip("Updated param files.")

    # testing
    else:
        # load expected values
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        expected_hm_params = xr.open_dataset(
            TEST_PATH
            / "harmonic_model"
            / f"params_harmonic_model_tas_{esm}_{scenario}.nc",
            decode_times=time_coder,
        )
        expected_pt_params = xr.open_dataset(
            TEST_PATH
            / "power_transformer"
            / f"params_power_transformer_tas_{esm}_{scenario}.nc",
            decode_times=time_coder,
        )
        expected_AR1_params = xr.open_dataset(
            TEST_PATH / "local_variability" / f"params_AR1_tas_{esm}_{scenario}.nc",
            decode_times=time_coder,
        )
        expected_localized_ecov = xr.open_dataset(
            TEST_PATH
            / "local_variability"
            / f"params_localized_ecov_tas_{esm}_{scenario}.nc",
            decode_times=time_coder,
        )
        expected_m_time = xr.open_dataset(
            TEST_PATH / "time" / f"params_monthly_time_tas_{esm}_{scenario}.nc",
            decode_times=time_coder,
        )

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
        import numpy as np

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
            AR1_fit.slope,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            expected_AR1_params.intercept,
            AR1_fit.intercept,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            localized_ecov.localized_covariance,
            localized_ecov.localized_covariance,
        )
