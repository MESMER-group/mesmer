import importlib

import pytest
import xarray as xr

import mesmer


@pytest.mark.slow
def test_calibrate_mesmer_m(update_expected_files=False):
    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    LOCALISATION_RADII = list(range(1250, 6251, 250)) + list(range(6500, 8501, 500))

    esm = "IPSL-CM6A-LR"

    # define paths and load data
    TEST_DATA_PATH = importlib.resources.files("mesmer").parent / "tests" / "test-data"
    TEST_PATH = TEST_DATA_PATH / "output" / "tas" / "mon"
    cmip6_data_path = TEST_DATA_PATH / "calibrate-coarse-grid" / "cmip6-ng"

    path_tas_ann = cmip6_data_path / "tas" / "ann" / "g025"
    fN_hist_ann = path_tas_ann / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_proj_ann = path_tas_ann / f"tas_ann_{esm}_ssp585_r1i1p1f1_g025.nc"

    tas_y = xr.open_mfdataset(
        [fN_hist_ann, fN_proj_ann],
        combine="by_coords",
        use_cftime=True,
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        drop_variables=["height", "file_qf"],
    ).load()

    path_tas_mon = cmip6_data_path / "tas" / "mon" / "g025"
    fN_hist_mon = path_tas_mon / f"tas_mon_{esm}_historical_r1i1p1f1_g025.nc"
    fN_proj_mon = path_tas_mon / f"tas_mon_{esm}_ssp585_r1i1p1f1_g025.nc"
    tas_m = xr.open_mfdataset(
        [fN_hist_mon, fN_proj_mon],
        combine="by_coords",
        use_cftime=True,
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        drop_variables=["height", "file_qf"],
    ).load()

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

    # we need to get the original time coordinate to be able to validate our results
    m_time = tas_stacked_m.time

    # fit harmonic model
    harmonic_model_fit = mesmer.stats.fit_harmonic_model(
        tas_stacked_y.tas, tas_stacked_m.tas
    )

    # train power transformer
    resids_after_hm = tas_stacked_m - harmonic_model_fit.hm_predictions
    pt_coefficients = mesmer.stats.fit_yeo_johnson_transform(
        resids_after_hm.tas, tas_stacked_y.tas
    )
    transformed_hm_resids = mesmer.stats.yeo_johnson_transform(
        resids_after_hm.tas, pt_coefficients.lambda_coeffs, tas_stacked_y.tas
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

    # merge into one dataset
    harmonic_model_fit = harmonic_model_fit.drop_vars("hm_predictions")
    AR1_fit = AR1_fit.drop_vars("residuals")
    localized_ecov = localized_ecov.drop_vars("covariance")
    m_time = m_time.rename("monthly_time")
    calibrated_params = xr.merge(
        [harmonic_model_fit, pt_coefficients, AR1_fit, localized_ecov, m_time]
    )

    # save params
    if update_expected_files:
        calibrated_params.to_netcdf(TEST_PATH / "test-mesmer_m-params.nc")
        pytest.skip("Updated test-mesmer_m-params.nc")

    # testing
    else:
        # load expected values
        expected_params = xr.open_dataset(
            TEST_PATH / "test-mesmer_m-params.nc", use_cftime=True
        )

        # the following parameters should be exactly the same
        exact_exp_params = xr.merge(
            [
                expected_params.hm_selected_order,
                expected_params.localization_radius,
                expected_params.monthly_time,
            ]
        )
        exact_cal_params = xr.merge(
            [
                calibrated_params.hm_selected_order,
                calibrated_params.localization_radius,
                calibrated_params.monthly_time,
            ]
        )

        xr.testing.assert_equal(exact_exp_params, exact_cal_params)

        # compare the rest
        # using numpy because it outputs the differences and how many values are off
        import numpy as np

        # the tols are set to the best we can do
        # NOTE: it is always rather few values that are off
        np.testing.assert_allclose(
            expected_params.hm_coeffs,
            calibrated_params.hm_coeffs,
            atol=1e-5,
            rtol=1 / 3,
        )
        # NOTE: would have to be atol is 1e12 here - not doing that
        np.testing.assert_allclose(
            expected_params.lambda_coeffs, calibrated_params.lambda_coeffs, rtol=0.6
        )
        np.testing.assert_allclose(
            expected_params.ar1_slope, calibrated_params.ar1_slope, atol=1e-5, rtol=1e-4
        )
        np.testing.assert_allclose(
            expected_params.ar1_intercept,
            calibrated_params.ar1_intercept,
            atol=1e-4,
            rtol=1e-2 / 3,
        )
        np.testing.assert_allclose(
            expected_params.localized_covariance,
            calibrated_params.localized_covariance,
            atol=1e-4,
            rtol=1e-2,
        )
