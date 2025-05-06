import glob
import pathlib

import pytest
import xarray as xr
from filefisher import FileFinder

import mesmer
from mesmer.core._datatreecompat import map_over_datasets
import mesmer.core.datatree


@pytest.mark.filterwarnings("ignore:No local minimum found")
@pytest.mark.parametrize(
    "scenarios, use_tas2, use_hfds, outname",
    (
        # tas
        pytest.param(
            ["ssp126"],
            False,
            False,
            "tas/one_scen_one_ens",
        ),
        pytest.param(
            ["ssp585"],
            False,
            False,
            "tas/one_scen_multi_ens",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            False,
            False,
            "tas/multi_scen_multi_ens",
        ),
        # tas and tas**2
        pytest.param(
            ["ssp126"],
            True,
            False,
            "tas_tas2/one_scen_one_ens",
            marks=pytest.mark.slow,
        ),
        # tas and hfds
        pytest.param(
            ["ssp126"],
            False,
            True,
            "tas_hfds/one_scen_one_ens",
            marks=pytest.mark.slow,
        ),
        # tas, tas**2, and hfds
        pytest.param(
            ["ssp126"],
            True,
            True,
            "tas_tas2_hfds/one_scen_one_ens",
        ),
        pytest.param(
            ["ssp585"],
            True,
            True,
            "tas_tas2_hfds/one_scen_multi_ens",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            True,
            True,
            "tas_tas2_hfds/multi_scen_multi_ens",
            #marks=pytest.mark.slow,
        ),
    ),
)
def test_calibrate_mesmer(
    scenarios,
    use_tas2,
    use_hfds,
    outname,
    test_data_root_dir,
    update_expected_files,
) -> None:

    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    LOCALISATION_RADII = range(1750, 2001, 250)

    esm = "IPSL-CM6A-LR"
    test_cmip_generation = 6
    variables = ["tas"]
    if use_hfds:
        variables.append("hfds")

    # define paths and load data
    TEST_DATA_PATH = pathlib.Path(test_data_root_dir)
    TEST_PATH = TEST_DATA_PATH / "output" / outname

    PARAM_FILEFINDER = FileFinder(
        path_pattern=TEST_PATH / "test-params/{module}/",
        file_pattern="params_{module}_{esm}_{scen}.nc",
    )

    cmip_data_path = mesmer.example_data.cmip6_ng_path()

    CMIP_FILEFINDER = FileFinder(
        path_pattern=str(cmip_data_path / "{variable}/{time_res}/{resolution}"),
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member}_{resolution}.nc",
    )

    fc_scens = CMIP_FILEFINDER.find_files(
        variable=variables, scenario=scenarios, model=esm, resolution="g025", time_res="ann"
    )

    # only get the historical members that are also in the future scenarios, but only once
    unique_scen_members = fc_scens.df.member.unique()

    fc_hist = CMIP_FILEFINDER.find_files(
        variable=variables,
        scenario="historical",
        model=esm,
        resolution="g025",
        time_res="ann",
        member=unique_scen_members,
    )

    fc_all = fc_hist.concat(fc_scens)

    scenarios_incl_hist = scenarios.copy()
    scenarios_incl_hist.append("historical")
    
    data = xr.DataTree()
    for scen in scenarios_incl_hist:
        # load data for each scenario
        data[scen] = xr.DataTree()

        for var in variables:
            files = fc_all.search(variable=var, scenario=scen)

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
            data[scen] = data[scen].assign({f"{var}": scen_data[var]})

    # data preprocessing
    # create global mean tas anomlies timeseries
    data = mesmer.grid.wrap_to_180(data)
    # convert the 0..360 grid to a -180..180 grid to be consistent with legacy code

    # calculate anomalies w.r.t. the reference period
    anoms = mesmer.anomaly.calc_anomaly(data, REFERENCE_PERIOD)

    globmean = mesmer.weighted.global_mean(anoms)

    # train global trend module
    globmean_ensmean = globmean.mean(dim="member")
    globmean_smoothed = mesmer.stats.lowess(
        globmean_ensmean, "time", n_steps=50, use_coords=False
    )
    hist_lowess_residuals = (
        globmean["historical"] - globmean_smoothed["historical"]
    )

    volcanic_params = mesmer.volc.fit_volcanic_influence(hist_lowess_residuals.tas)

    globmean_smoothed["historical"]["tas"] = mesmer.volc.superimpose_volcanic_influence(
        globmean_smoothed["historical"]["tas"],
        volcanic_params,
    )

    # train global variability module
    tas_glob_mean = map_over_datasets(lambda ds: ds.drop_vars("hfds"), globmean)
    tas_resid_novolc = tas_glob_mean - globmean_smoothed
    tas_resid_novolc = map_over_datasets(lambda ds: ds.rename({"tas": "tas_resids"}), tas_resid_novolc)

    ar_order = mesmer.stats.select_ar_order_scen_ens(
        tas_resid_novolc, dim="time", ens_dim="member", maxlag=12, ic="bic"
    )
    global_ar_params = mesmer.stats.fit_auto_regression_scen_ens(
        tas_resid_novolc, dim="time", ens_dim="member", lags=ar_order
    )

    # train local forced response module
    # create local gridded data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    target = map_over_datasets(lambda ds: ds.drop_vars("hfds"), anoms)
    target = mask_and_stack(target, threshold_land=THRESHOLD_LAND)

    predictors = mesmer.datatree.merge([globmean_smoothed, tas_resid_novolc])

    if use_tas2:
        predictors = map_over_datasets(lambda ds: ds.assign(tas2=ds.tas**2), predictors)

    weights = mesmer.weighted.equal_scenario_weights_from_datatree(
        target, ens_dim="member", time_dim="time"
    )

    predictors_stacked, target_stacked, weights_stacked = (
        mesmer.core.datatree.broadcast_and_stack_scenarios(predictors, target, weights)
    )

    local_forced_response_lr = mesmer.stats.LinearRegression()

    local_forced_response_lr.fit(
        predictors=predictors_stacked,
        target=target_stacked.tas,
        dim="sample",
        weights=weights_stacked.weights,
    )

    # train local variability module
    # train local AR process
    tas_stacked_residuals = local_forced_response_lr.residuals(
        predictors=predictors_stacked, target=target_stacked.tas
    )

    tas_un_stacked_residuals = tas_stacked_residuals.set_index(
        sample=("time", "member", "scenario")
    ).unstack("sample")

    dt_resids = xr.DataTree()
    for scenario in tas_un_stacked_residuals.scenario.values:
        dt_resids[scenario] = xr.DataTree(
            tas_un_stacked_residuals.sel(scenario=scenario)
            .dropna("member", how="all")
            .dropna("time")
            .drop_vars("scenario")
            .rename("residuals")
            .to_dataset()
        )

    local_ar_params = mesmer.stats.fit_auto_regression_scen_ens(
        dt_resids,
        ens_dim="member",
        dim="time",
        lags=1,
    )

    # train covariance
    geodist = mesmer.geospatial.geodist_exact(
        target["historical"].ds.lon, target["historical"].ds.lat
    )
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist, localisation_radii=LOCALISATION_RADII
    )

    dim = "sample"
    k_folds = 30

    localized_ecov = mesmer.stats.find_localized_empirical_covariance(
        tas_stacked_residuals, weights_stacked.weights, phi_gc_localizer, dim, k_folds
    )

    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    # parameter paths
    scen_str = "-".join(scenarios)

    volcanic_file = PARAM_FILEFINDER.create_full_name(
        module="volcanic",
        esm=esm,
        scen=scen_str,
    )
    global_ar_file = PARAM_FILEFINDER.create_full_name(
        module="global-variability",
        esm=esm,
        scen=scen_str,
    )
    local_forced_file = PARAM_FILEFINDER.create_full_name(
        module="local-trends",
        esm=esm,
        scen=scen_str,
    )
    local_ar_file = PARAM_FILEFINDER.create_full_name(
        module="local-variability",
        esm=esm,
        scen=scen_str,
    )
    localized_ecov_file = PARAM_FILEFINDER.create_full_name(
        module="covariance",
        esm=esm,
        scen=scen_str,
    )

    if update_expected_files:
        # save the parameters
        volcanic_params.to_netcdf(volcanic_file)
        global_ar_params.to_netcdf(global_ar_file)
        local_forced_response_lr.to_netcdf(local_forced_file)
        local_ar_params.to_netcdf(local_ar_file)
        localized_ecov.to_netcdf(localized_ecov_file)
        pytest.skip("Updated param files.")

    else:
        # testing
        assert_params_allclose(
            volcanic_params,
            global_ar_params,
            local_forced_response_lr.params,
            local_ar_params,
            localized_ecov,
            volcanic_file,
            global_ar_file,
            local_forced_file,
            local_ar_file,
            localized_ecov_file,
        )


def assert_params_allclose(
    volcanic_params,
    global_ar_params,
    local_forced_params,
    local_ar_params,
    localized_ecov,
    volcanic_file,
    global_ar_file,
    local_forced_file,
    local_ar_file,
    localized_ecov_file,
):
    # test params
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    exp_volcanic_params = xr.open_dataset(volcanic_file, decode_times=time_coder)
    exp_global_ar_params = xr.open_dataset(global_ar_file, decode_times=time_coder)
    exp_local_forced_params = xr.open_dataset(
        local_forced_file, decode_times=time_coder
    )
    exp_local_ar_params = xr.open_dataset(local_ar_file, decode_times=time_coder)
    exp_localized_ecov = xr.open_dataset(localized_ecov_file, decode_times=time_coder)

    xr.testing.assert_allclose(volcanic_params, exp_volcanic_params)
    xr.testing.assert_allclose(global_ar_params, exp_global_ar_params)

    # order by sample to avoid re-creating the expected data for now
    # TODO: replace data instead
    indexes = {"sample": ("time", "member", "scenario")}
    local_forced_params = local_forced_params.set_index(indexes)
    exp_local_forced_params = exp_local_forced_params.set_index(indexes)
    exp_local_forced_params = exp_local_forced_params.reindex_like(local_forced_params)

    xr.testing.assert_allclose(local_forced_params, exp_local_forced_params)
    xr.testing.assert_allclose(local_ar_params, exp_local_ar_params)
    xr.testing.assert_allclose(localized_ecov, exp_localized_ecov)
