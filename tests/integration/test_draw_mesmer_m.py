import importlib

import pytest
import xarray as xr

import mesmer


def test_make_emulations_mesmer_m(update_expected_files=False):

    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    esm = "IPSL-CM6A-LR"

    nr_emus = 10
    buffer = 20
    seed = 0

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

    # load parameters
    params = xr.open_dataset(TEST_PATH / "test-mesmer_m-params.nc", use_cftime=True)

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
        tas_stacked_y.tas, params.hm_coeffs, params.monthly_time
    )

    # generate variability around 0 with AR(1) model
    AR1_fit = xr.merge([params.ar1_slope, params.ar1_intercept]).rename(
        {"ar1_slope": "slope", "ar1_intercept": "intercept"}
    )
    local_variability_transformed = mesmer.stats.draw_auto_regression_monthly(
        AR1_fit,
        params.localized_covariance,
        time=params.monthly_time,
        n_realisations=nr_emus,
        seed=seed,
        buffer=buffer,
    )

    # invert the power transformation
    pt_coeffs = xr.merge([params.pt_xi0, params.pt_xi1]).rename(
        {"pt_xi0": "xi_0", "pt_xi1": "xi_1"}
    )
    local_variability_inverted = mesmer.stats.inverse_yeo_johnson_transform(
        local_variability_transformed,
        pt_coeffs,
        tas_stacked_y.tas,
    )

    # add the local variability to the monthly harmonic
    result = monthly_harmonic_emu + local_variability_inverted.inverted
    result = result.to_dataset(name="tas")

    # save
    if update_expected_files:
        result.to_netcdf(TEST_PATH / "test_mesmer_m_realisations_expected.nc")
        pytest.skip("Updated emulations.")

    # testing
    else:
        exp = xr.open_dataset(
            TEST_PATH / "test_mesmer_m_realisations_expected.nc", use_cftime=True
        )
        xr.testing.assert_allclose(result, exp)

        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = exp.set_index(z=("lat", "lon")).unstack("z")
        expected_dims = {"lat", "lon", "time", "gridcell", "realisation"}

        assert set(exp_reshaped.dims) == expected_dims
