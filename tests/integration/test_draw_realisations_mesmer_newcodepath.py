import os.path

import joblib
import numpy as np
import pathlib
import pandas
import pytest
import xarray as xr
import datatree
from datatree import DataTree, map_over_subtree
from filefisher import FileFinder, FileContainer

import mesmer.create_emulations

def create_forcing_data(test_data_root_dir, scenarios, use_hfds, use_tas2):
    # define config values
    THRESHOLD_LAND = 1 / 3

    REFERENCE_PERIOD = slice("1850", "1900")

    esm = "IPSL-CM6A-LR"
    test_cmip_generation = 6

    # define paths and load data
    TEST_DATA_PATH = pathlib.Path(test_data_root_dir)

    cmip_data_path = (
        TEST_DATA_PATH / "calibrate-coarse-grid" / f"cmip{test_cmip_generation}-ng"
    )

    CMIP_FILEFINDER = FileFinder(
        path_pattern=cmip_data_path / "{variable}/{time_res}/{resolution}", # type: ignore
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member_label}_{resolution}.nc",
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
        member_label=unique_scen_members,
    )

    # load data for each scenario
    def _get_hist(meta, fc_hist):

        meta_hist = meta | {"scenario": "historical"}

        fc = fc_hist.search(**meta_hist)

        if len(fc) == 0:
            raise FileNotFoundError("no hist file found")
        if len(fc) != 1:
            raise ValueError("more than one hist file found")

        fN, meta_hist = fc[0]

        return fN, meta_hist

    def load_hist(meta, fc_hist):
        fN, __ = _get_hist(meta, fc_hist)
        return xr.open_dataset(fN, use_cftime=True)

    def load_hist_scen_continuous(fc_hist, fc_scens):
        dt = DataTree()
        for scen in fc_scens.df.scenario.unique():
            files = fc_scens.search(scenario=scen)

            members = []
            
            for fN, meta in fc_scens.items():

                try:
                    hist = load_hist(meta, fc_hist)
                except FileNotFoundError:
                    continue

                proj = xr.open_dataset(fN, use_cftime=True)

                ds = xr.combine_by_coords(
                    [hist, proj],
                    combine_attrs="override",
                    data_vars="minimal",
                    compat="override",
                    coords="minimal",
                )

                ds = ds.drop_vars(("height", "time_bnds", "file_qf"), errors="ignore")

                ds = mesmer.grid.wrap_to_180(ds)

                # assign member-ID as coordinate
                ds = ds.assign_coords({"member": meta["member"]})

                members.append(ds)
            
            # create a Dataset that holds each member along the member dimension
            scen_data = xr.concat(members, dim="member")
            # put the scenario dataset into the DataTree
            dt[scen] = DataTree(scen_data)
        return dt

    tas = load_hist_scen_continuous(fc_hist, fc_scens)
    ref = tas.sel(time=REFERENCE_PERIOD).mean("time")
    tas_anoms = tas - ref
    tas_globmean = map_over_subtree(mesmer.weighted.global_mean)(tas_anoms)
    
    # TODO mask out ocean here??
    tas_globmean_forcing = map_over_subtree(mesmer.stats.lowess)(tas_globmean.mean(dim="member"), dim="time", n_steps=30, use_coords=False)

    if use_hfds:
        fc_hfds = CMIP_FILEFINDER.find_files(
            variable="hfds",
            scenario=scenarios,
            model=esm,
            resolution="g025",
            time_res="ann",
            member=unique_scen_members,
        )

        fc_hfds_hist = CMIP_FILEFINDER.find_files(
            variable="hfds",
            scenario="historical",
            model=esm,
            resolution="g025",
            time_res="ann",
            member=unique_scen_members,
        )


        hfds = load_hist_scen_continuous(fc_hfds, fc_hfds_hist)

        hfds_ref = hfds["historical"].sel(time=REFERENCE_PERIOD).mean("time")
        hfds_anoms = hfds - hfds_ref.ds
        hfds_globmean = map_over_subtree(mesmer.weighted.global_mean)(hfds_anoms)
        hfds_globmean_smoothed = map_over_subtree(mesmer.stats.lowess)(
            hfds_globmean.mean(dim="member"), "time", n_steps=50, use_coords=False
        )

    else:
        hfds_globmean_smoothed = None
    
    tas2 = tas_globmean**2 if use_tas2 else None

    return tas_globmean_forcing, hfds_globmean_smoothed, tas2

@pytest.mark.parametrize(
    "scenarios, use_tas2, use_hfds, n_realisations, outname",
    (
        # tas
        pytest.param(
            ["ssp126"],
            False,
            False,
            1,
            "tas/one_scen_one_ens",
        ),
        pytest.param(
            ["ssp585"],
            False,
            False,
            30,
            "tas/one_scen_multi_ens",
            # marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            False,
            False,
            100,
            "tas/multi_scen_multi_ens",
        ),
        # tas and tas**2
        pytest.param(
            ["ssp126"],
            True,
            False,
            1,
            "tas_tas2/one_scen_one_ens",
            # marks=pytest.mark.slow,
        ),
        # tas and hfds
        pytest.param(
            ["ssp126"],
            False,
            True,
            1,
            "tas_hfds/one_scen_one_ens",
            # marks=pytest.mark.slow,
        ),
        # tas, tas**2, and hfds
        pytest.param(
            ["ssp126"],
            True,
            True,
            1,
            "tas_tas2_hfds/one_scen_one_ens",
        ),
        pytest.param(
            ["ssp585"],
            True,
            True,
            1,
            "tas_tas2_hfds/one_scen_multi_ens",
            # marks=pytest.mark.slow,
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            True,
            True,
            1,
            "tas_tas2_hfds/multi_scen_multi_ens",
            # marks=pytest.mark.slow,
        ),
    ),
)
def test_make_realisations(
    scenarios,
    use_tas2,
    use_hfds,
    n_realisations,
    outname,
    test_data_root_dir,
    update_expected_files,
):
    esm = "IPSL-CM6A-LR"

    TEST_DATA_PATH = pathlib.Path(test_data_root_dir)
    TEST_PATH = TEST_DATA_PATH / "output" / outname

    PARAM_FILEFINDER = FileFinder(
        path_pattern = TEST_PATH / "params/{scope}/{param_type}",
        file_pattern = "params_{short_type}_{method}_tas_{esm}_{scen}_new.nc"
    )
    scen_str = "-".join(scenarios)

    volcanic_file = PARAM_FILEFINDER.find_single_file(scope="global", param_type="global_trend",
                                                      short_type="gt", method="LOWESS_OLSVOLC_saod",
                                                      esm=esm, scen=scen_str).paths.pop()
    global_ar_file = PARAM_FILEFINDER.find_single_file(scope="global", param_type="global_variability",
                                                      short_type="gv", method="AR_tas",
                                                      esm=esm, scen=scen_str).paths.pop()
    local_forced_file = PARAM_FILEFINDER.find_single_file(scope="local", param_type="local_trends",
                                                      short_type="lt", method="OLS_gttas",
                                                      esm=esm, scen=scen_str).paths.pop()
    local_ar_file = PARAM_FILEFINDER.find_single_file(scope="local", param_type="local_variability",
                                                      short_type="lv", method="OLS_AR1_sci_gvtas",
                                                      esm=esm, scen=scen_str).paths.pop()
    localized_ecov_file = PARAM_FILEFINDER.find_single_file(scope="local", param_type="local_variability",
                                                            short_type="lv", method="localized_ecov",
                                                            esm=esm, scen=scen_str).paths.pop()

    expected_output_file = TEST_PATH / "test_make_realisations_expected_output.nc"

    HIST_PERIOD = slice("1850", "2014")

    seed_list = [981, 314, 272, 42] # we need a maximum of 4 seeds if there are max 2 scenarios

    seed_global_variability = DataTree()
    seed_local_variability = DataTree()

    for scen in scenarios:
        seed_global_variability[scen] = DataTree(xr.Dataset({"seed": xr.DataArray(seed_list.pop())}))
        seed_local_variability[scen] = DataTree(xr.Dataset({"seed": xr.DataArray(seed_list.pop())}))
    
    tas_forcing, hfds, tas2 = create_forcing_data(test_data_root_dir, scenarios, use_hfds, use_tas2)
    time = tas_forcing["ssp126"].time
    
    buffer_global_variability = 50
    buffer_local_variability = 20

    volcanic_params = xr.open_dataset(volcanic_file)
    tas_forcing = map_over_subtree(mesmer.volc.superimpose_volcanic_influence)(
        tas_forcing, volcanic_params, hist_period=HIST_PERIOD
    )

    # 2.) compute the global variability
    global_ar_params = xr.open_dataset(global_ar_file)
    global_variability = map_over_subtree(mesmer.stats.draw_auto_regression_uncorrelated)(
        global_ar_params,
        realisation=n_realisations,
        time=time,
        seed=seed_global_variability,
        buffer=buffer_global_variability,
    ).rename({"samples": "tas"})

    # 3.) compute the local forced response
    lr = mesmer.stats.LinearRegression().from_netcdf(local_forced_file)

    predictors = DataTree()
    for scen in scenarios:
        predictors[scen] = DataTree.from_dict({"tas": tas_forcing[scen],
                                               "tas_resids": global_variability[scen]})
        
        if tas2 is not None:
            predictors[scen]["tas2"] = tas2[scen]
        if hfds is not None:
            predictors[scen]["hfds"] = hfds[scen]

    # uses ``exclude`` to split the linear response
    local_forced_response = DataTree()
    local_variability_from_global_var = DataTree()

    for key in predictors.children:
        local_forced_response[key] = DataTree(lr.predict(predictors[key], exclude={"tas_resids"}).rename("tas"))

        # 4.) compute the local variability part driven by global variabilty
        exclude = {"tas", "intercept"}
        if use_tas2:
            exclude.add("tas2")
        if use_hfds:
            exclude.add("hfds")

        local_variability_from_global_var[key] = DataTree(lr.predict(
            predictors[key], 
            exclude=exclude
        ).rename("tas"))

    # 5.) compute local variability
    local_ar = xr.open_dataset(local_ar_file)
    localized_covariance = xr.open_dataset(localized_ecov_file)
    localized_covariance_adjusted = localized_covariance.localized_covariance_adjusted

    local_variability = map_over_subtree(mesmer.stats.draw_auto_regression_correlated)(
        local_ar,
        localized_covariance_adjusted,
        time=time,
        realisation=n_realisations,
        seed=seed_local_variability,
        buffer=buffer_local_variability,
    ).rename({"samples": "tas"})

    local_variability_total = local_variability_from_global_var + local_variability

    result = local_forced_response + local_variability_total

    # save the emulations
    if update_expected_files:
        result.to_netcdf(expected_output_file)
        pytest.skip("Updated expected output file.")
    else:
        expected = datatree.open_datatree(expected_output_file, use_cftime=True)
        for scen in scenarios:
            exp_scen = expected[scen].to_dataset()
            res_scen = result[scen].to_dataset()
            xr.testing.assert_allclose(res_scen, exp_scen)

            # make sure we can get onto a lat lon grid from what is saved
            exp_reshaped = exp_scen.set_index(gridcell=("lat", "lon")).unstack("gridcell")
            expected_dims = {"realisation", "lon", "lat", "time"}

            assert set(exp_reshaped.dims) == expected_dims

