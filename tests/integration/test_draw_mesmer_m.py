import importlib

import pytest
import xarray as xr

import mesmer


def test_make_emulations_mesmer_m(update_expected_files=False):

    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    esm = "IPSL-CM6A-LR"
    scenario = "ssp585"

    nr_emus = 10
    buffer = 20
    seed = 0

    # define paths and load data
    TEST_DATA_PATH = importlib.resources.files("mesmer").parent / "tests" / "test-data"
    TEST_PATH = TEST_DATA_PATH / "output" / "tas" / "mon"
    cmip6_data_path = TEST_DATA_PATH / "calibrate-coarse-grid" / "cmip6-ng"

    path_tas_ann = cmip6_data_path / "tas" / "ann" / "g025"
    fN_hist_ann = path_tas_ann / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_proj_ann = path_tas_ann / f"tas_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

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

    # load parameters
    PARAMS_PATH = TEST_PATH / "test-params"
    hm_params = xr.open_dataset(
        PARAMS_PATH
        / "harmonic_model"
        / f"params_harmonic_model_tas_{esm}_{scenario}.nc",
        use_cftime=True,
    )
    pt_params = xr.open_dataset(
        PARAMS_PATH
        / "power_transformer"
        / f"params_power_transformer_tas_{esm}_{scenario}.nc",
        use_cftime=True,
    )
    AR1_params = xr.open_dataset(
        PARAMS_PATH / "local_variability" / f"params_AR1_tas_{esm}_{scenario}.nc",
        use_cftime=True,
    )
    localized_ecov = xr.open_dataset(
        PARAMS_PATH
        / "local_variability"
        / f"params_localized_ecov_tas_{esm}_{scenario}.nc",
        use_cftime=True,
    )
    m_time = xr.open_dataset(
        PARAMS_PATH / "time" / f"params_monthly_time_tas_{esm}_{scenario}.nc",
        use_cftime=True,
    )

    # preprocess yearly data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    ref = tas_y.sel(time=REFERENCE_PERIOD).mean("time", keep_attrs=True)
    tas_y = tas_y - ref
    tas_stacked_y = mask_and_stack(tas_y, threshold_land=THRESHOLD_LAND)

    # generate monthly data with harmonic model
    monthly_harmonic_emu = mesmer.stats.predict_harmonic_model(
        tas_stacked_y.tas, hm_params.coeffs, m_time.monthly_time
    )

    # generate variability around 0 with AR(1) model
    local_variability_transformed = mesmer.stats.draw_auto_regression_monthly(
        AR1_params,
        localized_ecov.localized_covariance,
        time=m_time.monthly_time,
        n_realisations=nr_emus,
        seed=seed,
        buffer=buffer,
    )

    # invert the power transformation
    local_variability_inverted = mesmer.stats.inverse_yeo_johnson_transform(
        local_variability_transformed,
        pt_params.lambda_coeffs,
        tas_stacked_y.tas,
    )

    # add the local variability to the monthly harmonic
    result = monthly_harmonic_emu + local_variability_inverted.inverted
    result = result.to_dataset(name="tas")

    # save
    test_file = TEST_PATH / "test_mesmer_m_realisations_expected.nc"
    if update_expected_files:
        result.to_netcdf(test_file)
        pytest.skip("Updated emulations.")

    # testing
    else:
        exp = xr.open_dataset(test_file, use_cftime=True)
        xr.testing.assert_allclose(result, exp)

        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = exp.set_index(z=("lat", "lon")).unstack("z")
        expected_dims = {"lat", "lon", "time", "gridcell", "realisation"}

        assert set(exp_reshaped.dims) == expected_dims
