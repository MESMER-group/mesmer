import importlib

import pytest
import xarray as xr

# from datatree import Datatree, map_over_subtree
import mesmer
import mesmer.mesmer_x

# TODO: extend to more scenarios and members
# TODO: extend predictors


@pytest.mark.parametrize(
    (
        "scenario",
        "target_name",
        "expr",
        "expr_name",
        "option_2ndfit",
        "update_expected_files",
    ),
    [
        pytest.param(
            "ssp585",
            "tasmax",
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr1",
            False,
            False,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "ssp585",
            "tasmax",
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr1_2ndfit",
            True,
            False,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_calibrate_mesmer_x(
    scenario, target_name, expr, expr_name, option_2ndfit, update_expected_files
):
    # set some configuration parameters
    THRESHOLD_LAND = 1 / 3
    esm = "IPSL-CM6A-LR"

    # TODO: replace with filefinder later
    # load data
    TEST_DATA_PATH = importlib.resources.files("mesmer").parent / "tests" / "test-data"
    TEST_PATH = TEST_DATA_PATH / "output" / target_name / "one_scen_one_ens" / "params"
    cmip6_data_path = TEST_DATA_PATH / "calibrate-coarse-grid" / "cmip6-ng"

    # load predictor data
    path_tas = cmip6_data_path / "tas" / "ann" / "g025"

    fN_hist = path_tas / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_tas / f"tas_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

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
    path_target = cmip6_data_path / target_name / "ann" / "g025"

    fN_hist = path_target / f"{target_name}_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_target / f"{target_name}_ann_{esm}_{scenario}_r1i1p1f1_g025.nc"

    targ_hist = xr.open_dataset(fN_hist, use_cftime=True)
    targ_ssp585 = xr.open_dataset(fN_ssp585, use_cftime=True)

    # target = DataTree({"hist": targ_hist, "ssp585": targ_ssp585})

    # stack target data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds, stack_dim="gridpoint")
        return ds

    # mask_and_stack_dt = map_over_subtree(mask_and_stack)
    targ_stacked_hist = mask_and_stack(targ_hist, threshold_land=THRESHOLD_LAND)
    targ_stacked_ssp585 = mask_and_stack(targ_ssp585, threshold_land=THRESHOLD_LAND)

    # collect scenarios in a tuple
    # NOTE: each of the datasets below could have a dimension along member
    predictor = ((tas_glob_mean_hist, "hist"), (tas_glob_mean_ssp585, "ssp585"))
    target = ((targ_stacked_hist, "hist"), (targ_stacked_ssp585, "ssp585"))

    # do the training
    transform_params, _ = mesmer.mesmer_x.xr_train_distrib(
        predictors=predictor,
        target=target,
        target_name=target_name,
        expr=expr,
        expr_name=expr_name,
        option_2ndfit=option_2ndfit,
        r_gasparicohn_2ndfit=500,
        scores_fit=["func_optim", "NLL", "BIC"],
    )

    # probability integral transform: projection of the data on a standard normal distribution
    transf_target = mesmer.mesmer_x.probability_integral_transform(  # noqa: F841
        data=target,
        target_name=target_name,
        expr_start=expr,
        coeffs_start=transform_params,
        preds_start=predictor,
        expr_end="norm(loc=0, scale=1)",
    )
    # TODO: add expression as varibale here or in function or before saving?

    # make transformed target into DataArrays
    transf_target_xr_hist = xr.DataArray(
        transf_target[0][0],
        dims=["time", "gridpoint"],
        coords={"time": targ_stacked_hist.time},
    ).assign_coords(targ_stacked_hist.gridpoint.coords)
    transf_target_xr_ssp585 = xr.DataArray(
        transf_target[1][0],
        dims=["time", "gridpoint"],
        coords={"time": targ_stacked_ssp585.time},
    ).assign_coords(targ_stacked_hist.gridpoint.coords)

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
        targ_stacked_hist.lon, targ_stacked_hist.lat
    )
    # prep localizer
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist, range(4000, 6001, 500)
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

    # Adjust regularized covariance matrix
    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    file_end = f"{target_name}_{expr_name}_{esm}_{scenario}"
    distrib_file = TEST_PATH / "distrib" / f"params_transform_distrib_{file_end}.nc"
    local_ar_file = TEST_PATH / "local_variability" / f"params_local_AR_{file_end}.nc"
    localized_ecov_file = (
        TEST_PATH / "local_variability" / f"params_localized_ecov_{file_end}.nc"
    )

    if update_expected_files:
        # save the parameters
        transform_params.to_netcdf(distrib_file)
        local_ar_params.to_netcdf(local_ar_file)
        localized_ecov.to_netcdf(localized_ecov_file)
        pytest.skip(f"Updated param files.")

    else:
        # load the parameters
        expected_transform_params = xr.open_dataset(distrib_file)
        xr.testing.assert_allclose(transform_params, expected_transform_params)

        expected_local_ar_params = xr.open_dataset(local_ar_file)
        xr.testing.assert_allclose(
            local_ar_params["intercept"],
            expected_local_ar_params["intercept"],
            atol=1e-7,
        )
        xr.testing.assert_allclose(
            local_ar_params["coeffs"], expected_local_ar_params["coeffs"]
        )
        xr.testing.assert_allclose(
            local_ar_params["variance"], expected_local_ar_params["variance"]
        )
        xr.testing.assert_equal(
            local_ar_params["nobs"], expected_local_ar_params["nobs"]
        )

        expected_localized_ecov = xr.open_dataset(localized_ecov_file)
        xr.testing.assert_allclose(localized_ecov, expected_localized_ecov)
