import numpy as np
import pytest
import xarray as xr
from filefisher import FileFinder

import mesmer
from mesmer.mesmer_x import (
    ConditionalDistribution,
    ConditionalDistributionOptions,
    Expression,
    ProbabilityIntegralTransform,
)


@pytest.mark.parametrize(
    (
        "scenarios",
        "targ_var",
        "pred_vars",
        "expr",
        "expr_name",
        "option_2ndfit",
        "outname",
    ),
    [
        pytest.param(
            ["ssp126"],
            "tasmax",
            ["tas"],
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr1",
            False,
            "tasmax/one_scen_one_ens",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126"],
            "tasmax",
            ["tas"],
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr1_2ndfit",
            True,
            "tasmax/one_scen_one_ens",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            "tasmax",
            ["tas"],
            "norm(loc=c1 + c2 * __tas__, scale=c3)",
            "expr2",
            False,
            "tasmax/multi_scen_multi_ens",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            "tasmax",
            ["tas", "hfds"],
            "norm(loc=c1 + c2 * __tas__ + c3 * __hfds__, scale=c4)",
            "expr3",
            False,
            "tasmax/multi_scen_multi_ens",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_calibrate_mesmer_x(
    scenarios,
    targ_var,
    pred_vars,
    expr,
    expr_name,
    option_2ndfit,
    test_data_root_dir,
    outname,
    update_expected_files,
):
    # set some configuration parameters
    THRESHOLD_LAND = 1 / 3
    REFERENCE_PERIOD = slice("1850", "1900")
    esm = "IPSL-CM6A-LR"

    # load data
    test_path = test_data_root_dir / "output" / outname

    cmip_data_path = mesmer.example_data.cmip6_ng_path()

    CMIP_FILEFINDER = FileFinder(
        path_pattern=str(cmip_data_path / "{variable}/{time_res}/{resolution}"),
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member}_{resolution}.nc",
    )

    fc_scens_pred = CMIP_FILEFINDER.find_files(
        variable=pred_vars,
        scenario=scenarios,
        model=esm,
        resolution="g025",
        time_res="ann",
    )

    # only get the historical members that are also in the future scenarios, but only once
    unique_scen_members = fc_scens_pred.df.member.unique()

    fc_hist_pred = CMIP_FILEFINDER.find_files(
        variable=pred_vars,
        scenario="historical",
        model=esm,
        resolution="g025",
        time_res="ann",
        member=unique_scen_members,
    )

    fc_pred = fc_hist_pred.concat(fc_scens_pred)

    fc_scens_targ = CMIP_FILEFINDER.find_files(
        variable=targ_var,
        scenario=scenarios,
        model=esm,
        resolution="g025",
        time_res="ann",
        member=unique_scen_members,
    )

    fc_hist_targ = CMIP_FILEFINDER.find_files(
        variable=targ_var,
        scenario="historical",
        model=esm,
        resolution="g025",
        time_res="ann",
        member=unique_scen_members,
    )

    fc_targ = fc_hist_targ.concat(fc_scens_targ)

    scenarios_incl_hist = scenarios.copy()
    scenarios_incl_hist.append("historical")

    def load_data(fc):
        data = xr.DataTree()

        scenarios = fc.df.scenario.unique().tolist()

        for scen in scenarios:
            # load data for each scenario
            data_scen = []

            for var in fc.df.variable.unique():
                files = fc.search(variable=var, scenario=scen)

                # load all members for a scenario
                members = []
                for fN, meta in files.items():
                    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                    ds = xr.open_dataset(fN, decode_times=time_coder)
                    # drop unnecessary variables
                    ds = ds.drop_vars(
                        ["height", "time_bnds", "file_qf", "area"], errors="ignore"
                    )
                    # assign member-ID as coordinate
                    ds = ds.assign_coords({"member": meta["member"]})
                    members.append(ds)

                # create a Dataset that holds each member along the member dimension
                data_var = xr.concat(members, dim="member")
                data_scen.append(data_var)

            data_scen = xr.merge(data_scen)
            data[scen] = xr.DataTree(data_scen)
        return data

    pred_data_orig = load_data(fc_pred)
    targ_data_orig = load_data(fc_targ)

    pred_data = mesmer.anomaly.calc_anomaly(pred_data_orig, REFERENCE_PERIOD)
    targ_data = mesmer.anomaly.calc_anomaly(targ_data_orig, REFERENCE_PERIOD)

    # make global mean of pred data
    pred_data = mesmer.weighted.global_mean(pred_data)

    # stack target data
    def mask_and_stack(ds, threshold_land):
        ds = mesmer.mask.mask_ocean_fraction(ds, threshold_land)
        ds = mesmer.mask.mask_antarctica(ds)
        ds = mesmer.grid.stack_lat_lon(ds, stack_dim="gridpoint")
        return ds

    # mask_and_stack_dt = map_over_subtree(mask_and_stack)
    targ_data = mask_and_stack(targ_data, threshold_land=THRESHOLD_LAND)

    # stack datasets
    # weights
    weights = mesmer.core.weighted.get_weights_density(pred_data=pred_data)

    # stacking
    stacked_pred, stacked_targ, stacked_weights = (
        mesmer.core.datatree.broadcast_and_pool_scen_ens(
            predictors=pred_data,
            target=targ_data,
            weights=weights,
            member_dim="member",
        )
    )

    # declaring analytical form of the conditional distribution
    expression = Expression(expr, expr_name)
    distrib = ConditionalDistribution(expression, ConditionalDistributionOptions())

    # preparing first guess
    coeffs_fg = distrib.find_first_guess(
        predictors=stacked_pred,
        target=stacked_targ.tasmax,
        first_guess=None,
        weights=stacked_weights.weights,
    )

    # training the conditional distribution
    distrib.fit(
        predictors=stacked_pred,
        target=stacked_targ.tasmax,
        weights=stacked_weights.weights,
        first_guess=coeffs_fg,
    )
    transform_coeffs = distrib.coefficients

    # second round if necessary
    if option_2ndfit:
        distrib.fit(
            predictors=stacked_pred,
            target=stacked_targ.tasmax,
            weights=stacked_weights.weights,
            first_guess=transform_coeffs,
            smooth_coeffs=True,
            r_gasparicohn=500,
        )
        transform_coeffs = distrib.coefficients

    # probability integral transform on non-stacked data for AR(1) process
    target_expr = Expression("norm(loc=0, scale=1)", "standard_normal")
    target_distrib = ConditionalDistribution(
        target_expr,
        ConditionalDistributionOptions(),
    )

    pit = ProbabilityIntegralTransform(distrib, target_distrib)
    transf_target = pit.transform(
        data=targ_data,
        target_name=targ_var,
        preds_orig=pred_data,
    )

    # training of auto-regression with spatially correlated innovations
    local_ar_params = mesmer.stats.fit_auto_regression_scen_ens(
        transf_target,
        ens_dim="member",
        dim="time",
        lags=1,
    )

    # estimate covariance matrix
    # prep distance matrix
    geodist = mesmer.core.geospatial.geodist_exact(
        lon=targ_data["historical"].lon, lat=targ_data["historical"].lat
    )
    # prep localizer (relatively coarse)
    LOCALISATION_RADII = range(10_000, 16_001, 1000)
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist=geodist, localisation_radii=LOCALISATION_RADII
    )

    # TODO: should we both for MESMER and for MESMER-X remove the
    # residuals from the AR(1) process before calculating the covariance?
    # is that we is happening in 'adjust_covariance_ar1'?
    # TODO: using here weights from MESMER-X. I noticed that it affects the
    # calculation in find_localized_empirical_covariance. Need to solve that.
    localized_ecov = mesmer.stats.find_localized_empirical_covariance(
        data=stacked_targ[targ_var],
        weights=stacked_weights.weights,
        localizer=phi_gc_localizer,
        dim="sample",
        k_folds=30,
    )

    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    # parameter files
    PARAM_FILEFINDER = FileFinder(
        path_pattern=test_path / "test-params/{module}/",
        file_pattern="params_{module}_{targ_var}_{expr_name}_{esm}_{scen}.nc",
    )

    scen_str = "_".join(scenarios)

    distrib_file = PARAM_FILEFINDER.create_full_name(
        module="distrib",
        targ_var=targ_var,
        expr_name=expr_name,
        esm=esm,
        scen=scen_str,
    )
    local_ar_file = PARAM_FILEFINDER.create_full_name(
        module="local_trends",
        targ_var=targ_var,
        expr_name=expr_name,
        esm=esm,
        scen=scen_str,
    )
    localized_ecov_file = PARAM_FILEFINDER.create_full_name(
        module="local_variability",
        targ_var=targ_var,
        expr_name=expr_name,
        esm=esm,
        scen=scen_str,
    )

    if update_expected_files:
        # save the parameters
        distrib.to_netcdf(distrib_file)
        local_ar_params.to_netcdf(local_ar_file)
        localized_ecov.to_netcdf(localized_ecov_file)
        pytest.skip("Updated param files.")

    else:
        # load the parameters
        expected_transform_params = xr.open_dataset(distrib_file)

        for coeff in expected_transform_params.data_vars:
            mesmer.testing.assert_allclose_allowed_failures(
                transform_coeffs[coeff].values,
                expected_transform_params[coeff].values,
                rtol=1e-5,
                atol=1e-5,
                err_msg=coeff,
                # because of https://github.com/MESMER-group/mesmer/issues/735
                allowed_failures=1,
            )

        expected_local_ar_params = xr.open_dataset(local_ar_file)

        mesmer.testing.assert_allclose_allowed_failures(
            local_ar_params["intercept"].values,
            expected_local_ar_params["intercept"].values,
            rtol=1e-5,
            atol=1e-5,
            allowed_failures=1,
        )

        mesmer.testing.assert_allclose_allowed_failures(
            local_ar_params["intercept"].values,
            expected_local_ar_params["intercept"].values,
            rtol=1e-5,
            atol=1e-5,
            allowed_failures=1,
        )

        mesmer.testing.assert_allclose_allowed_failures(
            local_ar_params["coeffs"].values,
            expected_local_ar_params["coeffs"].values,
            rtol=1e-5,
            atol=1e-5,
            allowed_failures=1,
        )
        mesmer.testing.assert_allclose_allowed_failures(
            local_ar_params["variance"].values,
            expected_local_ar_params["variance"].values,
            rtol=1e-5,
            atol=1e-5,
            allowed_failures=1,
        )
        np.testing.assert_equal(
            local_ar_params["nobs"].values,
            expected_local_ar_params["nobs"].values,
        )

        expected_localized_ecov = xr.open_dataset(localized_ecov_file)

        mesmer.testing.assert_allclose_allowed_failures(
            local_ar_params["variance"].values,
            expected_local_ar_params["variance"].values,
            rtol=1e-5,
            atol=1e-5,
            allowed_failures=1,
        )

        for key in localized_ecov.data_vars:
            np.testing.assert_allclose(
                localized_ecov[key].values,
                expected_localized_ecov[key].values,
                rtol=1e-5,
                atol=1e-5,
                err_msg=key,
                # because of https://github.com/MESMER-group/mesmer/issues/735
                # allowed_failures=1,
            )
