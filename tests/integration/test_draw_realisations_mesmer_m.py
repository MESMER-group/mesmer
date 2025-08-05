import filefisher
import pytest
import xarray as xr

import mesmer


@pytest.mark.filterwarnings("ignore:`lambda_coeffs` does not have `lambda_function`")
def test_make_emulations_mesmer_m(test_data_root_dir, update_expected_files):

    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    esm = "IPSL-CM6A-LR"
    scenario = "ssp585"

    nr_emus = 2
    buffer = 20
    seed = 0

    # define paths and load data
    test_path = test_data_root_dir / "output" / "tas" / "mon"
    cmip6_data_path = mesmer.example_data.cmip6_ng_path()

    path_tas_ann = cmip6_data_path / "tas" / "ann" / "g025"
    fN_hist_ann = path_tas_ann / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_proj_ann = path_tas_ann / f"tas_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    tas_y = xr.open_mfdataset(
        [fN_hist_ann, fN_proj_ann],
        combine="by_coords",
        decode_times=time_coder,
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        drop_variables=["height", "file_qf"],
    ).load()

    param_filefinder = filefisher.FileFinder(
        path_pattern=test_path / "one_scen_one_ens" / "test-params/{module}/",
        file_pattern="params_{module}_{variable}_{esm}_{scen}.nc",
    )

    scen_str = scenario

    keys = {"esm": esm, "scen": scen_str, "variable": "tas"}

    local_hm_file = param_filefinder.create_full_name(keys, module="harmonic-model")
    local_pt_file = param_filefinder.create_full_name(keys, module="power-transformer")
    local_ar_file = param_filefinder.create_full_name(keys, module="local-variability")
    localized_ecov_file = param_filefinder.create_full_name(keys, module="covariance")
    time_file = param_filefinder.create_full_name(keys, module="monthly-time")

    # load parameters

    hm_params = xr.open_dataset(local_hm_file, decode_times=time_coder)
    pt_params = xr.open_dataset(local_pt_file, decode_times=time_coder)
    AR1_params = xr.open_dataset(local_ar_file, decode_times=time_coder)
    localized_ecov = xr.open_dataset(localized_ecov_file, decode_times=time_coder)
    m_time = xr.open_dataset(time_file, decode_times=time_coder)

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
    ).samples

    # invert the power transformation
    yj_transformer = mesmer.stats.YeoJohnsonTransformer("logistic")
    local_variability_inverted = yj_transformer.inverse_transform(
        tas_stacked_y.tas,
        local_variability_transformed,
        pt_params.lambda_coeffs,
    )

    # add the local variability to the monthly harmonic
    result = monthly_harmonic_emu + local_variability_inverted.inverted
    result = result.to_dataset(name="tas")

    # save
    test_file = test_path / "test_mesmer_m_realisations_expected.nc"
    if update_expected_files:
        result.to_netcdf(test_file)
        pytest.skip("Updated emulations.")

    # testing
    else:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        exp = xr.open_dataset(test_file, decode_times=time_coder)
        xr.testing.assert_allclose(result, exp)

        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = exp.set_index(z=("lat", "lon")).unstack("z")
        expected_dims = {"lat", "lon", "time", "gridcell", "realisation"}

        assert set(exp_reshaped.dims) == expected_dims
