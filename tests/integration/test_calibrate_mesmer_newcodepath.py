import pathlib

import pytest
import xarray as xr
from filefisher import FileFinder

import mesmer
import mesmer.core.datatree
from mesmer.core._datatreecompat import map_over_datasets


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
            marks=pytest.mark.slow,
        ),
    ),
)
def test_calibrate_mesmer(
    scenarios,
    use_tas2,
    use_hfds,
    outname,
    test_data_root_dir,
    update_expected_files=False,
):

    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    HIST_PERIOD = slice("1850", "2014")

    LOCALISATION_RADII = range(1750, 2001, 250)

    esm = "IPSL-CM6A-LR"
    test_cmip_generation = 6

    # define paths and load data
    TEST_DATA_PATH = pathlib.Path(test_data_root_dir)
    TEST_PATH = TEST_DATA_PATH / "output" / outname

    PARAM_FILEFINDER = FileFinder(
        path_pattern=TEST_PATH / "test-params/{module}/",
        file_pattern="params_{module}_{esm}_{scen}.nc",
    )

    cmip_data_path = (
        TEST_DATA_PATH / "calibrate-coarse-grid" / f"cmip{test_cmip_generation}-ng"
    )

    CMIP_FILEFINDER = FileFinder(
        path_pattern=cmip_data_path / "{variable}/{time_res}/{resolution}",  # type: ignore
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member}_{resolution}.nc",
    )

    fc_scens = CMIP_FILEFINDER.find_files(
        variable="tas", scenario=scenarios, model=esm, resolution="g025", time_res="ann"
    )

    # only get the historical members that are also in the future scenarios, but only once
    unique_scen_members = fc_scens.df.member.unique()

    fc_hist = CMIP_FILEFINDER.find_files(
        variable="tas",
        scenario="historical",
        model=esm,
        resolution="g025",
        time_res="ann",
        member=unique_scen_members,
    )

    fc_all = fc_hist.concat(fc_scens)

    scenarios_incl_hist = scenarios.copy()
    scenarios_incl_hist.append("historical")

    # load data for each scenario
    dt = xr.DataTree()
    for scen in scenarios_incl_hist:
        files = fc_all.search(scenario=scen)

        # load all members for a scenario
        members = []
        for fN, meta in files.items():
            ds = xr.open_dataset(fN, use_cftime=True)
            # drop unnecessary variables
            ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
            # assign member-ID as coordinate
            ds = ds.assign_coords({"member": meta["member"]})
            members.append(ds)

        # create a Dataset that holds each member along the member dimension
        scen_data = xr.concat(members, dim="member")
        # put the scenario dataset into the DataTree
        dt[scen] = xr.DataTree(scen_data)

    # load additional data
    if use_hfds:
        fc_hfds = CMIP_FILEFINDER.find_files(
            variable="hfds",
            scenario=scenarios_incl_hist,
            model=esm,
            resolution="g025",
            time_res="ann",
            member=unique_scen_members,
        )

        dt_hfds = xr.DataTree()
        for scen in scenarios_incl_hist:
            files = fc_hfds.search(scenario=scen)

            members = []
            for fN, meta in files.items():
                ds = xr.open_dataset(fN, use_cftime=True)
                ds = ds.drop_vars(
                    ["height", "time_bnds", "file_qf", "area"], errors="ignore"
                )
                ds = ds.assign_coords({"member": meta["member"]})
                members.append(ds)

            scen_data = xr.concat(members, dim="member")
            dt_hfds[scen] = xr.DataTree(scen_data)
    else:
        dt_hfds = None

    # data preprocessing
    # create global mean tas anomlies timeseries
    dt = map_over_datasets(mesmer.grid.wrap_to_180, dt)
    # convert the 0..360 grid to a -180..180 grid to be consistent with legacy code

    # calculate anomalies w.r.t. the reference period
    tas_anoms = mesmer.anomaly.calc_anomaly(dt, REFERENCE_PERIOD)

    tas_globmean = map_over_datasets(mesmer.weighted.global_mean, tas_anoms)

    # create local gridded tas data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    tas_stacked = map_over_datasets(
        mask_and_stack, tas_anoms, kwargs={"threshold_land": THRESHOLD_LAND}
    )

    # train global trend module
    tas_globmean_ensmean = tas_globmean.mean(dim="member")
    tas_globmean_smoothed = map_over_datasets(
        mesmer.stats.lowess,
        tas_globmean_ensmean,
        "time",
        kwargs={"n_steps": 50, "use_coords": False},
    )
    hist_lowess_residuals = (
        tas_globmean["historical"] - tas_globmean_smoothed["historical"]
    )

    volcanic_params = mesmer.volc.fit_volcanic_influence(
        hist_lowess_residuals.tas, hist_period=HIST_PERIOD, dim="time"
    )

    tas_globmean_smoothed["historical"] = mesmer.volc.superimpose_volcanic_influence(
        tas_globmean_smoothed["historical"],
        volcanic_params,
        hist_period=HIST_PERIOD,
        dim="time",
    )

    # train global variability module
    tas_resid_novolc = tas_globmean - tas_globmean_smoothed

    ar_order = mesmer.stats.select_ar_order_scen_ens(
        tas_resid_novolc, dim="time", ens_dim="member", maxlag=12, ic="bic"
    )
    global_ar_params = mesmer.stats.fit_auto_regression_scen_ens(
        tas_resid_novolc, dim="time", ens_dim="member", lags=ar_order
    )

    if dt_hfds is not None:

        hfds_anoms = mesmer.anomaly.calc_anomaly(dt_hfds, REFERENCE_PERIOD)

        hfds_globmean = map_over_datasets(mesmer.weighted.global_mean, hfds_anoms)

        hfds_globmean_ensmean = hfds_globmean.mean(dim="member")
        hfds_globmean_smoothed = map_over_datasets(
            mesmer.stats.lowess,
            hfds_globmean_ensmean,
            "time",
            kwargs={"n_steps": 50, "use_coords": False},
        )
    else:
        hfds_globmean_smoothed = None

    # train local forced response module
    # broadcast so all datasets have all the dimensions
    # gridcell can be excluded because it will be mapped in the Linear Regression
    target = tas_stacked
    predictors = xr.DataTree.from_dict(
        {"tas": tas_globmean_smoothed, "tas_resids": tas_resid_novolc}
    )
    if use_tas2:
        predictors["tas2"] = tas_globmean_smoothed**2
    if hfds_globmean_smoothed is not None:
        predictors["hfds"] = hfds_globmean_smoothed

    weights = mesmer.weighted.equal_scenario_weights_from_datatree(
        target, ens_dim="member", time_dim="time"
    )

    predictors_stacked, target_stacked, weights_stacked = (
        mesmer.core.datatree.stack_datatrees_for_linear_regression(
            predictors, target, weights, stacking_dims=["member", "time"]
        )
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
        tas_stacked["historical"].ds.lon, tas_stacked["historical"].ds.lat
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
    exp_volcanic_params = xr.open_dataset(volcanic_file, use_cftime=True)
    exp_global_ar_params = xr.open_dataset(global_ar_file, use_cftime=True)
    exp_local_forced_params = xr.open_dataset(local_forced_file, use_cftime=True)
    exp_local_ar_params = xr.open_dataset(local_ar_file, use_cftime=True)
    exp_localized_ecov = xr.open_dataset(localized_ecov_file, use_cftime=True)

    xr.testing.assert_allclose(volcanic_params, exp_volcanic_params)
    xr.testing.assert_allclose(global_ar_params, exp_global_ar_params)
    xr.testing.assert_allclose(local_forced_params, exp_local_forced_params)
    xr.testing.assert_allclose(local_ar_params, exp_local_ar_params)
    xr.testing.assert_allclose(localized_ecov, exp_localized_ecov)
