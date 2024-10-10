import pathlib

import joblib
import numpy as np
import pandas
import pytest
import xarray as xr
from datatree import DataTree, map_over_subtree
from filefinder import FileContainer, FileFinder

import mesmer


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
            # marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            False,
            False,
            "tas/multi_scen_multi_ens",
        ),
        # TODO: Add the other test cases too
        # # tas and tas**2
        # pytest.param(
        #     ["ssp126"],
        #     True,
        #     False,
        #     "tas_tas2/one_scen_one_ens",
        #     marks=pytest.mark.slow,
        # ),
        # # tas and hfds
        # pytest.param(
        #     ["ssp126"],
        #     False,
        #     True,
        #     "tas_hfds/one_scen_one_ens",
        #     marks=pytest.mark.slow,
        # ),
        # # tas, tas**2, and hfds
        # pytest.param(
        #     ["ssp126"],
        #     True,
        #     True,
        #     "tas_tas2_hfds/one_scen_one_ens",
        # ),
        # pytest.param(
        #     ["ssp585"],
        #     True,
        #     True,
        #     "tas_tas2_hfds/one_scen_multi_ens",
        #     marks=pytest.mark.slow,
        # ),
        # pytest.param(
        #     ["ssp126", "ssp585"],
        #     True,
        #     True,
        #     "tas_tas2_hfds/multi_scen_multi_ens",
        #     marks=pytest.mark.slow,
        # ),
    ),
)
def test_calibrate_mesmer(
    scenarios,
    use_tas2,
    use_hfds,
    outname,
    test_data_root_dir,
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

    cmip_data_path = (
        TEST_DATA_PATH / "calibrate-coarse-grid" / f"cmip{test_cmip_generation}-ng"
    )

    CMIP_FILEFINDER = FileFinder(
        path_pattern=cmip_data_path / "{variable}/{time_res}/{resolution}",
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

    fc_all = FileContainer(pandas.concat([fc_hist.df, fc_scens.df]))

    scenarios_whist = scenarios.copy()
    scenarios_whist.append("historical")

    # load data for each scenario
    dt = DataTree()
    for scen in scenarios_whist:
        files = fc_all.search(scenario=scen)

        # load all members for a scenario
        members = []
        for fN, meta in files:
            ds = xr.open_dataset(fN, use_cftime=True)
            # drop unnecessary variables
            ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
            # assign member-ID as coordinate
            ds = ds.assign_coords({"member": meta["member"]})
            members.append(ds)

        # create a Dataset that holds each member along the member dimension
        scen_data = xr.concat(members, dim="member")
        # put the scenario dataset into the DataTree
        dt[f"{scen}"] = DataTree(scen_data)

    # data preprocessing
    # create global mean tas anomlies timeseries
    dt = map_over_subtree(mesmer.grid.wrap_to_180)(dt)
    # convert the 0..360 grid to a -180..180 grid to be consistent with legacy code

    # calculate anomalies w.r.t. the reference period
    ref = dt["historical"].sel(time=REFERENCE_PERIOD).mean("time")
    tas_anoms = dt - ref.ds
    tas_globmean = map_over_subtree(mesmer.weighted.global_mean)(tas_anoms)

    # create local gridded tas data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    tas_stacked = map_over_subtree(mask_and_stack)(
        tas_anoms, threshold_land=THRESHOLD_LAND
    )

    # train global trend module
    tas_globmean_smoothed = map_over_subtree(mesmer.stats.lowess)(
        tas_globmean.mean(dim="member"), "time", n_steps=50, use_coords=False
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

    # train local forced response module
    # broadcast so all datasets have all the dimensions
    # gridcell can be excluded because it will be mapped in the Linear Regression
    tas_globmean_smoothed_bc = tas_globmean_smoothed.broadcast_like(
        tas_stacked, exclude={"gridcell"}
    )
    tas_globmean_resids_bc = tas_resid_novolc.broadcast_like(
        tas_stacked, exclude={"gridcell"}
    )

    weights = mesmer.weighted.create_equal_scenario_weights_from_datatree(
        tas_globmean_smoothed_bc
    )

    tas_local_ds = mesmer.utils.collapse_datatree_into_dataset(
        tas_stacked, dim="scenario"
    )
    tas_glob_smoothed_ds = mesmer.utils.collapse_datatree_into_dataset(
        tas_globmean_smoothed_bc, dim="scenario"
    )
    tas_glob_resid_ds = mesmer.utils.collapse_datatree_into_dataset(
        tas_globmean_resids_bc, dim="scenario"
    )
    weights_ds = mesmer.utils.collapse_datatree_into_dataset(weights, dim="scenario")

    # stack the dimensions and drop nans
    tas_local_ds = tas_local_ds.stack(
        sample=("time", "member", "scenario"), create_index=False
    )
    tas_local_ds = tas_local_ds.dropna("sample")

    predictors = {
        "tas_globmean": tas_glob_smoothed_ds,
        "tas_globmean_resid": tas_glob_resid_ds,
    }
    for key, data in predictors.items():
        predictors[key] = data.stack(
            sample=("time", "member", "scenario"), create_index=False
        ).tas
        predictors[key] = predictors[key].dropna("sample")

    weights_stacked = weights_ds.stack(
        sample=("time", "member", "scenario"), create_index=False
    ).weights
    weights_stacked = weights_stacked.dropna("sample")

    local_forced_response_lr = mesmer.stats.LinearRegression()

    local_forced_response_lr.fit(
        predictors=predictors,
        target=tas_local_ds.tas,
        dim="sample",
        weights=weights_stacked,
    )

    # train local variability module
    # train local AR process
    tas_stacked_residuals = local_forced_response_lr.residuals(
        predictors=predictors, target=tas_local_ds.tas
    ).T

    tas_un_stacked_residuals = tas_stacked_residuals.set_index(
        sample=("time", "member", "scenario")
    ).unstack("sample")

    dt_resids = {}
    for scenario in tas_un_stacked_residuals.scenario.values:
        dt_resids[scenario] = (
            tas_un_stacked_residuals.sel(scenario=scenario)
            .dropna("member", how="all")
            .dropna("time")
            .drop_vars("scenario")
            .rename("residuals")
        )
    dt_resids = DataTree.from_dict(dt_resids)

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
        tas_stacked_residuals, weights_stacked, phi_gc_localizer, dim, k_folds
    )

    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    # testing
    assert_params_allclose(
        TEST_PATH,
        global_ar_params,
        local_forced_response_lr,
        local_ar_params,
        localized_ecov,
    )


def assert_params_allclose(
    TEST_PATH,
    global_ar_params,
    local_forced_response_lr,
    local_ar_params,
    localized_ecov,
):
    fN_bundle = TEST_PATH / "test-mesmer-bundle.pkl"
    bundle = joblib.load(fN_bundle)

    # TODO: Test volcanic influence params too (not in bundle)

    # global variability
    np.testing.assert_allclose(
        bundle["params_gv"]["AR_int"], global_ar_params.intercept
    )
    np.testing.assert_equal(
        bundle["params_gv"]["AR_order_sel"], global_ar_params.lags.max().values
    )
    np.testing.assert_allclose(bundle["params_gv"]["AR_coefs"], global_ar_params.coeffs)
    np.testing.assert_allclose(
        bundle["params_gv"]["AR_var_innovs"], global_ar_params.variance
    )

    # local forced response
    np.testing.assert_allclose(
        bundle["params_lt"]["intercept"]["tas"],
        local_forced_response_lr.params.intercept,
    )

    np.testing.assert_allclose(
        bundle["params_lt"]["coef_gttas"]["tas"],
        local_forced_response_lr.params.tas_globmean,
    )

    np.testing.assert_allclose(
        bundle["params_lv"]["coef_gvtas"]["tas"],
        local_forced_response_lr.params.tas_globmean_resid,
    )

    # local variability
    # AR process
    np.testing.assert_allclose(
        bundle["params_lv"]["AR1_coef"]["tas"], local_ar_params.coeffs.squeeze()
    )
    np.testing.assert_allclose(
        bundle["params_lv"]["AR1_int"]["tas"], local_ar_params.intercept.squeeze()
    )
    np.testing.assert_allclose(
        bundle["params_lv"]["AR1_var_innovs"]["tas"],
        local_ar_params.variance.squeeze(),
    )

    # covariance
    assert bundle["params_lv"]["L"]["tas"] == localized_ecov.localization_radius

    np.testing.assert_allclose(
        bundle["params_lv"]["ecov"]["tas"], localized_ecov.covariance
    )

    np.testing.assert_allclose(
        bundle["params_lv"]["loc_ecov"]["tas"],
        localized_ecov.localized_covariance,
        atol=1e-7,
    )
