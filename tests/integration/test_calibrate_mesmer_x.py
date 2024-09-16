import importlib

import pytest
import xarray as xr

# from datatree import Datatree, map_over_subtree
import mesmer
import mesmer.mesmer_x


@pytest.mark.parametrize(
    ("expr", "outname", "update_expected_files"),
    [
        pytest.param("norm(loc=c1 + c2 * __tas__, scale=c3)", "exp1", False),
    ],
)
def test_calibrate_mesmer_x(expr, outname, update_expected_files):
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
    result, _ = mesmer.mesmer_x.xr_train_distrib(
        predictors=predictor,
        target=target,
        target_name="tasmax",
        expr=expr,
        expr_name=outname,
        option_2ndfit=False,
        r_gasparicohn_2ndfit=500,
        scores_fit=["func_optim", "NLL", "BIC"],
    )

    # probability integral transform: projection of the data on a standard normal distribution
    # transf_target = mesmer.mesmer_x.probability_integral_transform(  # noqa: F841
    #     data=target,
    #     expr_start=expr,
    #     coeffs_start=result,
    #     preds_start=predictor,
    #     expr_end="norm(loc=0, scale=1)",
    #     target_name="tasmax",
    # )

    if update_expected_files:
        # save the parameters
        result.to_netcdf(TEST_PATH / f"test-mesmer_x-params_{outname}.nc")

    else:
        # load the parameters
        expected = xr.open_dataset(TEST_PATH / f"test-mesmer_x-params_{outname}.nc")
        xr.testing.assert_allclose(result, expected)
