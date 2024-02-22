import importlib

import cartopy.crs as ccrs
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

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
        # TODO: Add the other test cases too
        # pytest.param(
        #     ["h-ssp585"],
        #     False,
        #     False,
        #     "tas/one_scen_multi_ens",
        #     marks=pytest.mark.slow,
        # ),
        # pytest.param(
        #     ["h-ssp126", "h-ssp585"],
        #     False,
        #     False,
        #     "tas/multi_scen_multi_ens",
        # ),
        # # tas and tas**2
        # pytest.param(
        #     ["h-ssp126"],
        #     True,
        #     False,
        #     "tas_tas2/one_scen_one_ens",
        #     marks=pytest.mark.slow,
        # ),
        # # tas and hfds
        # pytest.param(
        #     ["h-ssp126"],
        #     False,
        #     True,
        #     "tas_hfds/one_scen_one_ens",
        #     marks=pytest.mark.slow,
        # ),
        # # tas, tas**2, and hfds
        # pytest.param(
        #     ["h-ssp126"],
        #     True,
        #     True,
        #     "tas_tas2_hfds/one_scen_one_ens",
        # ),
        # pytest.param(
        #     ["h-ssp585"],
        #     True,
        #     True,
        #     "tas_tas2_hfds/one_scen_multi_ens",
        #     marks=pytest.mark.slow,
        # ),
        # pytest.param(
        #     ["h-ssp126", "h-ssp585"],
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
    tmpdir,
    update_expected_files,
):

    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    HIST_PERIOD = slice("1850", "2014")
    PROJ_PERIOD = slice("2015", "2100")

    LOCALISATION_RADII = range(1750, 2001, 250)

    esm = "IPSL-CM6A-LR"
    scenario = scenarios[0]
    test_cmip_generation = 6

    # define paths
    TEST_DATA_PATH = importlib.resources.files("mesmer").parent / "tests" / "test-data"
    TEST_PATH = TEST_DATA_PATH / "output" / "tas" / "one_scen_one_ens"
    PARAMS_PATH = TEST_PATH / "params"

    cmip_data_path = (
        TEST_DATA_PATH / "calibrate-coarse-grid" / f"cmip{test_cmip_generation}-ng"
    )

    path_tas = cmip_data_path / "tas" / "ann" / "g025"

    fN_hist = path_tas / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_proj = path_tas / f"tas_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

    tas = xr.open_mfdataset(
        [fN_hist, fN_proj],
        combine="by_coords",
        use_cftime=True,
        combine_attrs="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        drop_variables=["height", "file_qf"],
    ).load()

    # data preprocessing
    ## create global mean tas anomlies timeseries
    tas = mesmer.grid.wrap_to_180(
        tas
    )  # convert the 0..360 grid to a -180..180 grid to be consistent with legacy code

    ref = tas.sel(time=REFERENCE_PERIOD).mean("time", keep_attrs=True)
    tas = tas - ref
    tas_globmean = mesmer.weighted.global_mean(tas)

    ## create local gridded tas data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds)
        return ds

    grid_orig = tas[["lat", "lon"]]  # we need to keep the original grid
    tas_stacked = mask_and_stack(tas, threshold_land=THRESHOLD_LAND)

    # train global trend module
    tas_globmean_lowess = mesmer.stats.lowess(
        tas_globmean, "time", n_steps=50, use_coords=False
    )
    tas_lowess_residuals = tas_globmean - tas_globmean_lowess

    volcanic_params = mesmer.volc.fit_volcanic_influence(
        tas_lowess_residuals.tas, hist_period=HIST_PERIOD, dim="time"
    )

    tas_globmean_volc = mesmer.volc.superimpose_volcanic_influence(
        tas_globmean_lowess, volcanic_params, hist_period=HIST_PERIOD, dim="time"
    )

    # train global variability module
    def _split_hist_proj(
        obj, dim="time", hist_period=HIST_PERIOD, proj_period=PROJ_PERIOD
    ):
        hist = obj.sel({dim: hist_period})
        proj = obj.sel({dim: proj_period})

        return hist, proj

    tas_hist_globmean_smooth_volc, tas_proj_smooth = _split_hist_proj(tas_globmean_volc)

    tas_hist_resid_novolc = tas_globmean - tas_hist_globmean_smooth_volc
    tas_proj_resid = tas_globmean - tas_proj_smooth

    data = (tas_hist_resid_novolc.tas, tas_proj_resid.tas)

    ar_order = mesmer.stats._select_ar_order_scen_ens(
        *data, dim="time", ens_dim="ens", maxlag=12, ic="bic"
    )
    global_ar_params = mesmer.stats._fit_auto_regression_scen_ens(
        *data, dim="time", ens_dim="ens", lags=ar_order
    )

    # train local forced response module
    predictors_split = {
        "tas_globmean": [tas_hist_globmean_smooth_volc.tas, tas_proj_smooth.tas],
        "tas_globmean_resid": [tas_hist_resid_novolc.tas, tas_proj_resid.tas],
    }

    predictors = dict()
    for key, value in predictors_split.items():
        predictors[key] = xr.concat(value, dim="time")

    local_forced_response_lr = mesmer.stats.LinearRegression()

    local_forced_response_lr.fit(
        predictors=predictors,
        target=tas_stacked.tas,
        dim="time",  # switch to sample?
    )

    local_forced_response_params = local_forced_response_lr.params

    # train local variability module
    ## train local AR process
    tas_stacked_residuals = local_forced_response_lr.residuals(
        predictors=predictors, target=tas_stacked.tas
    )

    tas_stacked_residuals_hist, tas_stacked_residuals_proj = _split_hist_proj(
        tas_stacked_residuals
    )

    data = (tas_stacked_residuals_hist, tas_stacked_residuals_proj)
    local_ar_params = mesmer.stats._fit_auto_regression_scen_ens(
        *data,
        ens_dim="none",
        dim="time",
        lags=1,
    )

    ## train covariance
    geodist = mesmer.geospatial.geodist_exact(tas_stacked.lon, tas_stacked.lat)
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist, localisation_radii=LOCALISATION_RADII
    )

    weights = xr.ones_like(tas_globmean.tas)  # equal weights (for now?)
    weights.name = "weights"

    dim = "time"  # rename to "sample"
    k_folds = 30

    localized_ecov = mesmer.stats.find_localized_empirical_covariance(
        tas_stacked_residuals, weights, phi_gc_localizer, dim, k_folds
    )

    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    # ==================================================================== #
    # testing
    fN_bundle = TEST_PATH / "test-mesmer-bundle.pkl"

    bundle = joblib.load(fN_bundle)

    # def load_params(*folders, file):
    #     fN = os.path.join(PARAMS_PATH, *folders, file)

    #     return joblib.load(fN)

    # load data

    ## global trend
    # is not in the bundle

    # params_gt_lowess_tas = load_params(
    #     "global",
    #     "global_trend",
    #     file=f"params_gt_LOWESS_OLSVOLC_saod_tas_{esm}_h-{scenario}.pkl",
    # )

    # np.testing.assert_allclose(
    #     volcanic_params.aod.values, params_gt_lowess_tas["saod"]
    # )

    # np.testing.assert_allclose(
    #     params_gt_lowess_tas["hist"], tas_hist_globmean_smooth_volc.tas.values
    # )
    # np.testing.assert_allclose(
    #     params_gt_lowess_tas[scenario], tas_proj_smooth.tas.values
    # )

    ## global variability
    # params_gv_T = load_params(
    #     "global",
    #     "global_variability",
    #     file=f"params_gv_AR_tas_{esm}_hist_{scenario}.pkl",
    # )

    np.testing.assert_allclose(
        bundle["params_gv"]["AR_int"], global_ar_params.intercept
    )
    np.testing.assert_equal(
        bundle["params_gv"]["AR_order_sel"], global_ar_params.lags.max().values
    )
    np.testing.assert_allclose(bundle["params_gv"]["AR_coefs"], global_ar_params.coeffs)
    np.testing.assert_allclose(
        bundle["params_gv"]["AR_std_innovs"], global_ar_params.standard_deviation
    )

    np.testing.assert_allclose(  # this is not necessarily the same
        bundle["params_gv"]["AR_std_innovs"] ** 2, global_ar_params.variance, atol=2e-5
    )

    ## local forced response
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

    ## local variability
    ### AR process
    np.testing.assert_allclose(
        bundle["params_lv"]["AR1_coef"]["tas"], local_ar_params.coeffs.squeeze()
    )
    np.testing.assert_allclose(
        bundle["params_lv"]["AR1_int"]["tas"], local_ar_params.intercept.squeeze()
    )
    np.testing.assert_allclose(
        bundle["params_lv"]["AR1_std_innovs"]["tas"],
        local_ar_params.standard_deviation.squeeze(),
    )

    ### covariance
    assert bundle["params_lv"]["L"]["tas"] == localized_ecov.localization_radius

    np.testing.assert_allclose(
        bundle["params_lv"]["ecov"]["tas"], localized_ecov.covariance
    )

    np.testing.assert_allclose(
        bundle["params_lv"]["loc_ecov"]["tas"],
        localized_ecov.localized_covariance,
        atol=1e-7,
    )
