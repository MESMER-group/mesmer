import importlib

import pytest
import xarray as xr

# from datatree import Datatree, map_over_subtree
import mesmer
import mesmer.mesmer_x


@pytest.mark.parametrize(
    ("expr", "outname", "update_expected_files"),
    [
        pytest.param(
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "exp1",
            False,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "exp1_2ndfit",
            False,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_emulations_mesmer_x(expr, outname, update_expected_files):
    # set some configuration parameters
    n_realizations = 1  # TODO: more is not possible atm
    seed = 0
    buffer = 10
    targ = "tasmax"

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

    # concat
    predictor = xr.concat([tas_glob_mean_hist, tas_glob_mean_ssp585], dim="time")
    time = predictor.time

    # load the parameters
    transform_params = xr.open_dataset(
        TEST_PATH / f"test-mesmer_x-transf_params_{outname}.nc"
    )

    local_ar_params = xr.open_dataset(
        TEST_PATH / f"test-mesmer_x-local_ar_params_{outname}.nc"
    )

    localized_ecov = xr.open_dataset(
        TEST_PATH / f"test-mesmer_x-localized_ecov_{outname}.nc"
    )

    # txx = DataTree({"hist": txx_hist, "ssp585": txx_ssp585})

    # generate realizations based on the auto-regression with spatially correlated innovations
    transf_emus = mesmer.stats.draw_auto_regression_correlated(
        local_ar_params,
        localized_ecov.localized_covariance_adjusted,
        time=time,
        realisation=n_realizations,
        seed=seed,
        buffer=buffer,
    )
    transf_emus = xr.Dataset({"tasmax": transf_emus})

    # back-transform the realizations
    emus = mesmer.mesmer_x.probability_integral_transform(
        target_name=targ,
        data=[(transf_emus.sel(realisation=0), "ssp585")],
        expr_start="norm(loc=0, scale=1)",
        expr_end=expr,
        coeffs_end=transform_params,
        preds_end=[(predictor, "ssp585")],
    )

    emus = xr.DataArray(emus[0][0], dims=["time", "gridpoint"], coords={"time": time})
    emus = emus.assign_coords(local_ar_params.gridpoint.coords)

    expected_output_file = TEST_PATH / f"test-mesmer_x-emus_{outname}.nc"
    if update_expected_files:
        # save output
        emus.to_netcdf(expected_output_file)
        pytest.skip(f"Updated {expected_output_file}")
    else:
        # load the output
        expected_emus = xr.open_dataarray(expected_output_file, use_cftime=True)
        xr.testing.assert_allclose(emus, expected_emus)

        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = expected_emus.set_index(gridpoint=("lat", "lon")).unstack(
            "gridpoint"
        )
        expected_dims = {"lon", "lat", "time"}

        assert set(exp_reshaped.dims) == expected_dims
