import datetime as dt
import importlib

import xarray as xr

import mesmer


def main():
    start = dt.datetime.now()

    # load data
    model = "IPSL-CM6A-LR"
    TEST_DATA_PATH = importlib.resources.files("mesmer").parent / "tests" / "test-data"
    cmip6_data_path = TEST_DATA_PATH / "calibrate-coarse-grid" / "cmip6-ng"

    path_tas = cmip6_data_path / "tas" / "ann" / "g025"
    fN_hist = path_tas / f"tas_ann_{model}_historical_r1i1p1f1_g025.nc"
    fN_proj = path_tas / f"tas_ann_{model}_ssp585_r1i1p1f1_g025.nc"

    tas_y = xr.open_mfdataset(
        [fN_hist, fN_proj],
        combine="by_coords",
        use_cftime=True,
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        drop_variables=["height", "file_qf"],
    ).load()

    path_tas = cmip6_data_path / "tas" / "mon" / "g025"
    fN_hist = path_tas / f"tas_mon_{model}_historical_r1i1p1f1_g025.nc"
    fN_proj = path_tas / f"tas_mon_{model}_ssp585_r1i1p1f1_g025.nc"
    tas_m = xr.open_mfdataset(
        [fN_hist, fN_proj],
        combine="by_coords",
        use_cftime=True,
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        drop_variables=["height", "file_qf"],
    ).load()

    # preprocess tas
    ref_period = slice("1850", "1900")
    ref_y = tas_y.sel(time=ref_period).mean("time", keep_attrs=True)
    ref_m = tas_m.sel(time=ref_period).mean("time", keep_attrs=True)

    tas_y = tas_y - ref_y
    tas_m = tas_m - ref_m

    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    tas_stacked_y = mask_and_stack(tas_y, threshold_land=1 / 3)
    tas_stacked_m = mask_and_stack(tas_m, threshold_land=1 / 3)

    # fit harmonic model
    harmonic_model_fit = mesmer.stats.fit_harmonic_model(
        tas_stacked_y.tas, tas_stacked_m.tas
    )

    # train power transformer
    resids_after_hm = tas_stacked_m - harmonic_model_fit.predictions
    pt_coefficients = mesmer.stats.fit_yeo_johnson_transform(
        resids_after_hm.tas, tas_stacked_y.tas
    )
    transformed_hm_resids = mesmer.stats.yeo_johnson_transform(
        resids_after_hm.tas, pt_coefficients, tas_stacked_y.tas
    )

    # fit cyclo-stationary AR(1) process
    AR1_fit = mesmer.stats.fit_auto_regression_monthly(
        transformed_hm_resids.transformed, time_dim="time"
    )

    # work out covariance matrix
    geodist = mesmer.geospatial.geodist_exact(tas_stacked_y.lon, tas_stacked_y.lat)
    LOCALISATION_RADII = [
        1250,
        1500,
        1750,
        2000,
        2250,
        2500,
        2750,
        3000,
        3250,
        3500,
        3750,
        4000,
        4250,
        4500,
        4750,
        5000,
        5250,
        5500,
        6000,
        6250,
        6500,
        7000,
        7500,
        8000,
        8500,
    ]
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist, localisation_radii=LOCALISATION_RADII
    )

    weights = xr.ones_like(AR1_fit.residuals.isel(gridcell=0))
    weights.name = "weights"

    localized_ecov = mesmer.stats.find_localized_empirical_covariance_monthly(
        AR1_fit.residuals, weights, phi_gc_localizer, "time", 30
    )

    # make emulations
    nr_emus = 10
    buffer = 20
    seed = 0
    ref_period = slice("1850", "1900")

    # preprocess tas
    ref = tas_y.sel(time=ref_period).mean("time", keep_attrs=True)
    tas_y = tas_y - ref
    tas_stacked_y = mask_and_stack(tas_y, threshold_land=1 / 3)

    # we need to keep the original grid
    grid_orig = ref[["lat", "lon"]]
    # we need to get the original time coordinate to be able to validate our results
    m_time = tas_stacked_m.time

    # generate monthly data with harmonic model
    monthly_harmonic_emu = mesmer.stats.predict_harmonic_model(
        tas_stacked_y.tas, harmonic_model_fit.coeffs, m_time
    )

    # generate variability around 0 with AR(1) model
    local_variability_transformed = mesmer.stats.draw_auto_regression_monthly(
        AR1_fit,
        localized_ecov.localized_covariance,
        time=m_time,
        n_realisations=nr_emus,
        seed=seed,
        buffer=buffer,
    )

    # invert the power transformation
    local_variability_inverted = mesmer.stats.inverse_yeo_johnson_transform(
        local_variability_transformed, pt_coefficients, tas_stacked_y.tas
    )

    # add the local variability to the monthly harmonic
    emulations = monthly_harmonic_emu + local_variability_inverted.inverted

    # unstack to original grid
    emulations_unstacked = mesmer.grid.unstack_lat_lon_and_align(emulations, grid_orig)

    # save
    # TODO

    print(emulations_unstacked)
    print("total time:", dt.datetime.now() - start)


if __name__ == "__main__":
    main()
