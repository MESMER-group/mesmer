import importlib

import pytest
import xarray as xr

# from datatree import Datatree, map_over_subtree
import mesmer
import mesmer.mesmer_x


@pytest.mark.parametrize(
    ("expr", "option_2ndfit", "outname", "update_expected_files"),
    [
        pytest.param(
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            False,
            "exp1",
            False,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            True,
            "exp1_2ndfit",
            False,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_calibrate_mesmer_x(expr, option_2ndfit, outname, update_expected_files):
    # set some configuration parameters
    THRESHOLD_LAND = 1 / 3

    # load data
    TEST_DATA_PATH = importlib.resources.files("mesmer").parent / "tests" / "test-data"
    TEST_PATH = TEST_DATA_PATH / "output" / "txx"
    cmip6_data_path = TEST_DATA_PATH / "calibrate-coarse-grid" / "cmip6-ng"

    # load predictor data
    path_tas = cmip6_data_path / "tas" / "ann" / "g025"

    fN_hist = path_tas / "tas_ann_IPSL-CM6A-LR_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_tas / "tas_ann_IPSL-CM6A-LR_ssp585_r1i1p1f1_g025.nc"

    tas_hist = xr.open_dataset(fN_hist, use_cftime=True).drop_vars(
        ["height", "file_qf", "time_bnds"]
    )
    tas_ssp585 = xr.open_dataset(fN_ssp585, use_cftime=True).drop_vars(
        ["height", "file_qf", "time_bnds"]
    )

    # tas = DataTree({"hist": tas_hist, "ssp585": tas_ssp585})

    # make global mean
    # global_mean_dt = map_over_subtree(mesmer.weighted.global_mean)
    tas_glob_mean_hist = mesmer.weighted.global_mean(tas_hist)
    tas_glob_mean_ssp585 = mesmer.weighted.global_mean(tas_ssp585)

    # load target data
    path_txx = cmip6_data_path / "txx" / "ann" / "g025"

    fN_hist = path_txx / "txx_ann_IPSL-CM6A-LR_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_txx / "txx_ann_IPSL-CM6A-LR_ssp585_r1i1p1f1_g025.nc"

    txx_hist = xr.open_dataset(fN_hist, use_cftime=True)
    txx_ssp585 = xr.open_dataset(fN_ssp585, use_cftime=True)

    # txx = DataTree({"hist": txx_hist, "ssp585": txx_ssp585})

    # stack target data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds, stack_dim="gridpoint")
        return ds

    # mask_and_stack_dt = map_over_subtree(mask_and_stack)
    txx_stacked_hist = mask_and_stack(txx_hist, threshold_land=THRESHOLD_LAND)
    txx_stacked_ssp585 = mask_and_stack(txx_ssp585, threshold_land=THRESHOLD_LAND)

    # collect scenarios in a tuple
    # NOTE: each of the datasets below could have a dimension along member
    predictor = ((tas_glob_mean_hist, "hist"), (tas_glob_mean_ssp585, "ssp585"))
    target = ((txx_stacked_hist, "hist"), (txx_stacked_ssp585, "ssp585"))

    # do the training
    transform_params, _ = mesmer.mesmer_x.xr_train_distrib(
        predictors=predictor,
        target=target,
        target_name="tasmax",
        expr=expr,
        expr_name=outname,
        option_2ndfit=option_2ndfit,
        r_gasparicohn_2ndfit=500,
        scores_fit=["func_optim", "NLL", "BIC"],
    )

    # probability integral transform: projection of the data on a standard normal distribution
    transf_target = mesmer.mesmer_x.probability_integral_transform(  # noqa: F841
        data=target,
        target_name="tasmax",
        expr_start=expr,
        coeffs_start=transform_params,
        preds_start=predictor,
        expr_end="norm(loc=0, scale=1)",
    )

    # transformed target into DataArrays
    transf_target_xr_hist = xr.DataArray(
        transf_target[0][0],
        dims=["time", "gridpoint"],
        coords={"time": txx_stacked_hist.time},
    )
    transf_target_xr_ssp585 = xr.DataArray(
        transf_target[1][0],
        dims=["time", "gridpoint"],
        coords={"time": txx_stacked_ssp585.time},
    )

    # training of auto-regression with spatially correlated innovations
    local_ar_params = mesmer.stats._fit_auto_regression_scen_ens(
        transf_target_xr_hist,
        transf_target_xr_ssp585,
        ens_dim=None,
        dim="time",
        lags=1,
    )

    # estimate covariance matrix
    # prep distance matrix
    geodist = mesmer.geospatial.geodist_exact(
        txx_stacked_hist.lon, txx_stacked_hist.lat
    )
    # prep localizer
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist, range(2000, 9001, 500)
    )

    # stack target
    transf_target_stacked = xr.concat(
        [transf_target_xr_hist, transf_target_xr_ssp585], dim="scenario"
    )
    transf_target_stacked = transf_target_stacked.assign_coords(
        scenario=["hist", "ssp585"]
    )
    transf_target_stacked = transf_target_stacked.stack(
        {"sample": ["time", "scenario"]}, create_index=False
    ).dropna("sample")

    # make weights
    weights = xr.ones_like(transf_target_stacked.isel(gridpoint=0))

    # find covariance
    dim = "sample"
    k_folds = 15

    localized_ecov = mesmer.stats.find_localized_empirical_covariance(
        transf_target_stacked, weights, phi_gc_localizer, dim, k_folds
    )

    # Adjust regularized covariance matrix # TODO: varify if this is actually done in MESMER-X
    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    if update_expected_files:
        # save the parameters
        transform_params.to_netcdf(
            TEST_PATH / f"test-mesmer_x-transf_params_{outname}.nc"
        )
        local_ar_params.to_netcdf(
            TEST_PATH / f"test-mesmer_x-local_ar_params_{outname}.nc"
        )
        localized_ecov.to_netcdf(
            TEST_PATH / f"test-mesmer_x-localized_ecov_{outname}.nc"
        )

    else:
        # load the parameters
        expected_transform_params = xr.open_dataset(
            TEST_PATH / f"test-mesmer_x-transf_params_{outname}.nc"
        )
        xr.testing.assert_allclose(transform_params, expected_transform_params)

        expected_local_ar_params = xr.open_dataset(
            TEST_PATH / f"test-mesmer_x-local_ar_params_{outname}.nc"
        )
        xr.testing.assert_allclose(local_ar_params, expected_local_ar_params)

        expected_localized_ecov = xr.open_dataset(
            TEST_PATH / f"test-mesmer_x-localized_ecov_{outname}.nc"
        )
        xr.testing.assert_allclose(localized_ecov, expected_localized_ecov)
