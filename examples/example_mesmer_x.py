import pathlib

import filefinder  # FutureWarning: filefinder --> filefisher
import numpy as np
import xarray as xr

import mesmer
import mesmer.mesmer_x
from mesmer.core._datatreecompat import map_over_datasets


def main():

    # ==============================================================
    # 0. OPTIONS FOR THE SCRIPT
    # target
    target = "mrso_minmon"  # txx, mrso, fwils, fwisa, fwixd, fwixx, mrso_minmon
    target_anomaly = False
    dir_target = pathlib.Path("/net/exo/landclim/yquilcaille/annual_indicators")

    # predictor
    predictor = "tas"
    dir_pred = pathlib.Path("/net/ch4/data/cmip6-Next_Generation/")

    # Earth Sytem Model to emulate
    esm = "IPSL-CM6A-LR"
    # ==============================================================

    # ==============================================================
    # 1. CONFIGURATION PARAMETERS
    THRESHOLD_LAND = 1 / 3
    REFERENCE_PERIOD = slice("1850", "1900")
    scenarios = ["historical", "ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
    # ==============================================================

    # ==============================================================
    # 2. LOADING: MUST BE ADAPTED BASED ON THE ARCHIVE OF THE USER
    # preparing file finders: the structure depends on the archive!
    CMIP_FILEFINDER_pred = filefinder.FileFinder(
        path_pattern=dir_pred / "{variable}/{time_res}/{resolution}",
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member}_{resolution}.nc",
    )
    CMIP_FILEFINDER_target = filefinder.FileFinder(
        path_pattern=dir_target / "{variable}/{time_res}/{resolution}",
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member}_{resolution}.nc",
    )

    # finding files: careful about naming of resolutions
    fc_pred = CMIP_FILEFINDER_pred.find_files(
        variable=predictor,
        scenario=scenarios,
        model=esm,
        resolution="g025",
        time_res="ann",
    )
    fc_target = CMIP_FILEFINDER_target.find_files(
        variable=target,
        scenario=scenarios,
        model=esm,
        resolution="g025",
        time_res="ann",
    )

    # filtering
    runs_targ, runs_pred = [], []
    for r_pred in fc_pred.items():
        r_targ = fc_target.search(**r_pred[1] | {"variable": target})
        if len(r_targ) == 1:
            runs_pred.append(r_pred[0])
            runs_targ.append(r_targ[0][0])

    # loading: the preprocessing steps depend on the archive!
    # TODO: the current filtering does not check whether runs stop in 2100/2300
    cut_off = 2100

    def preprocess_data(ds, stack_lat_lon):
        # creating coordinate for member
        ds = ds.assign_coords({"member": ds.attrs["variant_label"]})
        # mask and stack
        ds = mesmer.core.mask.mask_ocean_fraction(ds, THRESHOLD_LAND)
        ds = mesmer.core.mask.mask_antarctica(ds)
        if stack_lat_lon:
            ds = mesmer.core.grid.stack_lat_lon(ds, stack_dim="gridpoint")
        ds = mesmer.core.grid.wrap_to_180(ds)
        ds["time"] = ds.time.dt.year
        ds = ds.sel(time=slice(1850, cut_off))
        for dump_dim in ["time_bnds", "file_qf", "height"]:
            if dump_dim in ds:
                ds = ds.drop_vars(dump_dim)
        return ds

    # loading & create datatree
    def create_datatree_from_list(list_paths_ds, name_dt, stack_lat_lon):
        out = dict()
        for run in list_paths_ds:
            # load
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds = preprocess_data(
                ds=xr.open_dataset(run, decode_times=time_coder),
                stack_lat_lon=stack_lat_lon,
            )
            scen = ds.attrs["experiment_id"]
            if scen not in out:
                # initialize
                out[scen] = xr.DataTree(ds)
            else:
                # add run
                dt_scen_bef = xr.Dataset(out[scen])
                out[scen] = xr.DataTree(xr.concat([dt_scen_bef, ds], dim="member"))
        return xr.DataTree.from_dict(out, name=name_dt)

    tas_data = create_datatree_from_list(
        list_paths_ds=runs_pred, name_dt=predictor, stack_lat_lon=False
    )
    targ_data = create_datatree_from_list(
        list_paths_ds=runs_targ, name_dt=target, stack_lat_lon=True
    )

    # calculating anomalies if necessary
    tas_data = mesmer.core.anomaly.calc_anomaly(tas_data, REFERENCE_PERIOD)
    if target_anomaly:
        targ_data = mesmer.core.anomaly.calc_anomaly(targ_data, REFERENCE_PERIOD)

    # calculating predictors
    tas_globmean = map_over_datasets(mesmer.core.weighted.global_mean, tas_data)

    # example to calculate additional predictors
    # calculating gmt at t-1 to set at t
    tmp = dict()
    for scen, (ds,) in xr.group_subtrees(tas_globmean):
        if scen == ".":
            continue
        ds = ds.to_dataset()
        if scen == "historical":
            # repeating first value of historical
            fill_value = ds.isel(time=0)
        else:
            # taking last value of historical
            fill_value = (
                tas_globmean["historical"]
                .to_dataset()
                .isel(time=-1)
                .sel(member=ds["member"])
            )
        dat = ds.shift(
            shifts={"time": 1}
        )  # , fill_value=fill_value)# doesnt work for some reason?
        dat[{"time": 0}] = fill_value
        tmp[scen] = dat
    tas_glomean_tm1 = xr.DataTree.from_dict(tmp)

    # creating predictors for training
    pred_data = xr.DataTree.from_dict(
        {"gmt": tas_globmean, "gmt_tm1": tas_glomean_tm1}, name="predictors"
    )
    # ==============================================================

    # ==============================================================
    # 3. EXAMPLE TO TRAIN MESMER-X
    # weight
    weights = mesmer.mesmer_x.get_weights_uniform(
        targ_data=targ_data, target=target, dims=("member", "time")
    )
    # weights_test = mesmer.core.weighted.equal_scenario_weights_from_datatree(targ_data, ens_dim="member", time_dim="time")
    # NB: mesmer.core.weighted.equal_scenario_weights_from_datatree and
    # mesmer.mesmer_x.get_weights_uniform are roughly equivalent, but
    # the rescale differs two. The weights from MESMER-X sum up to 1 over
    # all (member, time, scenario), while those of MESMER sum up to 1 over
    # only (member).

    # or

    weights = mesmer.mesmer_x.get_weights_density(
        pred_data=pred_data,
        predictor=predictor,
        targ_data=targ_data,
        target=target,
        dims=("member", "time"),
    )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # WARNING: FOR THE SAKE OF TIME, WRITING EASY STACKED DATASETS FROM NOW!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    stacked_pred, stacked_targ, stacked_weights = (
        mesmer.core.datatree.stack_datatrees_for_linear_regression(
            predictors=pred_data,
            target=targ_data,
            weights=weights,
            stacking_dims=["member", "time"],
        )
    )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # WARNING: FOR TESTING, TAKES ONLY A SUBSET OF THE DATA
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    test_targ_data = targ_data.isel(gridpoint=slice(0, 5))
    test_stacked_targ = stacked_targ.isel(gridpoint=slice(0, 5))

    # declaring analytical form of the conditional distribution
    expr_name = "cfgA"
    expr = "norm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __gmt__ + c4 * __gmt_tm1__ - c5) ), scale=c6)"
    expression_fit = mesmer.mesmer_x.Expression(expr, expr_name)

    # preparing tests that will be used for first guess and training
    tests_mx = mesmer.mesmer_x.distrib_tests(
        expr_fit=expression_fit,
        threshold_min_proba=1.0e-9,
        boundaries_params=None,
        boundaries_coeffs=None,
    )

    # preparing optimizers that will be used for first guess and training
    optim_mx = mesmer.mesmer_x.distrib_optimizer(
        expr_fit=expression_fit,
        class_tests=tests_mx,
        options_optim=None,
        options_solver=None,
    )

    # preparing first guess
    fg_mx = mesmer.mesmer_x.distrib_firstguess(
        expr_fit=expression_fit,
        class_tests=tests_mx,
        class_optim=optim_mx,
        first_guess=None,
        func_first_guess=None,
    )
    coeffs_fg = fg_mx.find_fg(
        predictors=stacked_pred,
        target=test_stacked_targ,
        weights=stacked_weights,
        dim="sample",
    )

    # training the conditional distribution
    train_mx = mesmer.mesmer_x.distrib_train(
        expr_fit=expression_fit, class_tests=tests_mx, class_optim=optim_mx
    )
    # first round
    coefficients = train_mx.fit(
        predictors=stacked_pred,
        target=test_stacked_targ,
        first_guess=coeffs_fg,
        weights=stacked_weights,
        dim="sample",
    )
    scores = train_mx.eval_quality_fit(
        predictors=stacked_pred,
        target=test_stacked_targ,
        coefficients_fit=coefficients,
        weights=stacked_weights,
        dim="sample",
        scores_fit=["func_optim", "nll", "bic"],
    )

    # second round if necessary
    if False:
        coefficients = train_mx.fit(
            predictors=stacked_pred,
            target=test_stacked_targ,
            first_guess=coefficients,
            weights=stacked_weights,
            dim="sample",
            option_smooth_coeffs=True,
            r_gasparicohn=500,
        )
        scores = train_mx.eval_quality_fit(
            predictors=stacked_pred,
            target=test_stacked_targ,
            coefficients_fit=coefficients,
            weights=stacked_weights,
            dim="sample",
            scores_fit=["func_optim", "nll", "bic"],
        )

    # probability integral transform on non-stacked data for AR(1) process
    pit = mesmer.mesmer_x.probability_integral_transform(
        expr_start=expr,
        coeffs_start=coefficients,
        expr_end="norm(loc=0, scale=1)",
        coeffs_end=None,
    )
    transf_target = pit.transform(
        data=test_targ_data, target_name=target, preds_start=pred_data, preds_end=None
    )

    # training of auto-regression with spatially correlated innovations
    # not applied on residuals, but on 'transf_target'
    # (code based on test_calibrate_mesmer_newcodepath.py)
    local_ar_params = mesmer.stats.fit_auto_regression_scen_ens(
        transf_target,
        ens_dim="member",
        dim="time",
        lags=1,
    )

    # train covariance
    geodist = mesmer.core.geospatial.geodist_exact(
        lon=test_stacked_targ.lon, lat=test_stacked_targ.lat
    )
    LOCALISATION_RADII = range(1750, 2001, 250)
    phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
        geodist=geodist, localisation_radii=LOCALISATION_RADII
    )
    # TODO: should we both for MESMER and for MESMER-X remove the
    # residuals from the AR(1) process before calculating the covariance?
    # is that we is happening in 'adjust_covariance_ar1'?
    # TODO: using here weights from MESMER-X. I noticed that it affects the
    # calculation in find_localized_empirical_covariance. Need to solve that.
    localized_ecov = mesmer.stats.find_localized_empirical_covariance(
        data=test_stacked_targ[target],
        weights=stacked_weights.weight,
        localizer=phi_gc_localizer,
        dim="sample",
        k_folds=30,
    )

    localized_ecov["localized_covariance_adjusted"] = (
        mesmer.stats.adjust_covariance_ar1(
            localized_ecov.localized_covariance, local_ar_params.coeffs
        )
    )

    # TODO: add save of parameters. preferred option: netCDF. PR?
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # 3.3. EXAMPLE OF EMULATION
    # example of scenario: GMT rises by 1% every year from 2015 to 2100
    name_newscenario = "new_scenario"
    gmt_0 = tas_globmean["historical"]["tas"].sel(time=2014).mean("member").values
    new_tas_globmean = xr.DataArray(
        gmt_0 * (1.01) ** np.arange(1, 2100 - 2015 + 1),
        coords={"time": np.arange(2015, 2100)},
        dims=["time"],
    )
    new_tas_globmean = xr.Dataset({predictor: new_tas_globmean})
    new_tas_globmean_tm1 = new_tas_globmean.shift(time=1, fill_value=gmt_0)
    new_pred_data = xr.DataTree.from_dict(
        {
            "gmt": xr.DataTree.from_dict({name_newscenario: new_tas_globmean}),
            "gmt_tm1": xr.DataTree.from_dict({name_newscenario: new_tas_globmean_tm1}),
        },
        name="predictors",
    )

    # compute local variability
    local_variability = mesmer.stats.draw_auto_regression_correlated(
        local_ar_params,
        localized_ecov["localized_covariance_adjusted"],
        time=new_tas_globmean.time,
        realisation=100,
        seed=42,
        buffer=42,
    )
    dataset_localvar = xr.Dataset({target: local_variability})
    datatree_localvar = xr.DataTree.from_dict({name_newscenario: dataset_localvar})

    # compute back-probability integral transform = emulations
    back_pit = mesmer.mesmer_x.probability_integral_transform(
        expr_start="norm(loc=0, scale=1)",
        coeffs_start=None,
        expr_end=expr,
        coeffs_end=coefficients,
    )
    # version with datatree
    emulations = back_pit.transform(
        data=datatree_localvar,
        target_name=target,
        preds_start=None,
        preds_end=new_pred_data,
    )
    # version with dataset
    emulations = back_pit.transform(
        data=dataset_localvar,
        target_name=target,
        preds_start=None,
        preds_end=xr.Dataset(
            {"gmt": new_tas_globmean["tas"], "gmt_tm1": new_tas_globmean_tm1["tas"]}
        ),
    )
    # --------------------------------------------------------------
    # ==============================================================


if __name__ == "__main__":
    main()

# %%
