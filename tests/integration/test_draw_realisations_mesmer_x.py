import pytest
import xarray as xr
from filefisher import FileFinder

import mesmer
from mesmer.mesmer_x import (
    ConditionalDistribution,
    Expression,
    ProbabilityIntegralTransform,
)


@pytest.mark.parametrize(
    ("scenario", "targ_var", "expr_name"),
    [
        pytest.param(
            "ssp126",
            "tasmax",
            "expr1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "ssp126",
            "tasmax",
            "expr1_2ndfit",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_make_realisations_mesmer_x(
    scenario, targ_var, expr_name, test_data_root_dir, update_expected_files
):
    # set some configuration parameters
    n_realizations = 1  # TODO: can now do more than 1 realization. change output data?
    seed = 0
    buffer = 10
    esm = "IPSL-CM6A-LR"

    # load data
    test_path = test_data_root_dir / "output" / targ_var / "one_scen_one_ens"
    cmip6_data_path = mesmer.example_data.cmip6_ng_path()

    # load predictor data
    path_tas = cmip6_data_path / "tas" / "ann" / "g025"

    fN_hist = path_tas / f"tas_ann_{esm}_historical_r1i1p1f1_g025.nc"
    fN_ssp585 = path_tas / f"tas_ann_{esm}_ssp585_r1i1p1f1_g025.nc"
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    tas_hist = xr.open_dataset(fN_hist, decode_times=time_coder).drop_vars(
        ["height", "file_qf", "time_bnds"]
    )
    tas_ssp585 = xr.open_dataset(fN_ssp585, decode_times=time_coder).drop_vars(
        ["height", "file_qf", "time_bnds"]
    )

    # make global mean
    # global_mean_dt = map_over_datasets(mesmer.weighted.global_mean)
    tas_glob_mean_hist = mesmer.weighted.global_mean(tas_hist)
    tas_glob_mean_ssp585 = mesmer.weighted.global_mean(tas_ssp585)

    # concat
    predictor = xr.concat([tas_glob_mean_hist, tas_glob_mean_ssp585], dim="time")
    time = predictor.time

    # load the parameters
    PARAM_FILEFINDER = FileFinder(
        path_pattern=test_path / "test-params/{module}/",
        file_pattern="params_{module}_{targ_var}_{expr_name}_{esm}_{scen}.nc",
    )

    distrib_file = PARAM_FILEFINDER.find_single_file(
        module="distrib",
        targ_var=targ_var,
        expr_name=expr_name,
        esm=esm,
        scen=scenario,
    ).paths.pop()
    local_ar_file = PARAM_FILEFINDER.find_single_file(
        module="local_trends",
        targ_var=targ_var,
        expr_name=expr_name,
        esm=esm,
        scen=scenario,
    ).paths.pop()
    localized_ecov_file = PARAM_FILEFINDER.find_single_file(
        module="local_variability",
        targ_var=targ_var,
        expr_name=expr_name,
        esm=esm,
        scen=scenario,
    ).paths.pop()

    distrib_orig = ConditionalDistribution.from_netcdf(distrib_file)

    local_ar_params = xr.open_dataset(local_ar_file)

    localized_ecov = xr.open_dataset(localized_ecov_file)

    # generate realizations based on the auto-regression with spatially correlated innovations
    transf_emus = mesmer.stats.draw_auto_regression_correlated(
        local_ar_params,
        localized_ecov.localized_covariance_adjusted,
        time=time,
        realisation=n_realizations,
        seed=seed,
        buffer=buffer,
    )
    transf_emus = transf_emus.rename({"samples": targ_var})

    # back-transform the realizations
    expr_tranf = Expression("norm(loc=0, scale=1)", "standard_normal")
    distrib_transf = ConditionalDistribution(expr_tranf)

    back_pit = ProbabilityIntegralTransform(
        distrib_orig=distrib_transf,
        distrib_targ=distrib_orig,
    )

    emus = back_pit.transform(
        data=transf_emus, target_name=targ_var, preds_orig=None, preds_targ=predictor
    )

    expected_output_file = (
        test_path / f"test_make_realisations_expected_output_{expr_name}.nc"
    )
    if update_expected_files:
        # save output
        emus.to_netcdf(expected_output_file)
        pytest.skip(f"Updated {expected_output_file}")
    else:
        # load the output
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        expected_emus = xr.open_dataarray(expected_output_file, decode_times=time_coder)
        xr.testing.assert_allclose(emus[targ_var], expected_emus)

        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = expected_emus.set_index(gridpoint=("lat", "lon")).unstack(
            "gridpoint"
        )
        expected_dims = {"realisation", "lon", "lat", "time"}

        assert set(exp_reshaped.dims) == expected_dims
