import pathlib

import pytest
import xarray as xr
from filefisher import FileFinder

import mesmer
from mesmer.core.datatree import map_over_datasets


def create_forcing_data(scenarios, use_hfds):
    # define config values
    REFERENCE_PERIOD = slice("1850", "1900")

    esm = "IPSL-CM6A-LR"
    variables = ["tas"]
    if use_hfds:
        variables.append("hfds")

    # define paths and load data
    cmip_data_path = mesmer.example_data.cmip6_ng_path()

    CMIP_FILEFINDER = FileFinder(
        path_pattern=str(cmip_data_path / "{variable}/{time_res}/{resolution}"),
        file_pattern="{variable}_{time_res}_{model}_{scenario}_{member}_{resolution}.nc",
    )

    fc_scens = CMIP_FILEFINDER.find_files(
        variable=variables,
        scenario=scenarios,
        model=esm,
        resolution="g025",
        time_res="ann",
    )

    # only get the historical members that are also in the future scenarios, but only once
    unique_scen_members = fc_scens.df.member.unique()

    fc_hist = CMIP_FILEFINDER.find_files(
        variable=variables,
        scenario="historical",
        model=esm,
        resolution="g025",
        time_res="ann",
        member=unique_scen_members,
    )

    # load data for each scenario
    def _get_hist_path(meta, fc_hist):

        meta_hist = meta | {"scenario": "historical"}

        fc = fc_hist.search(**meta_hist)

        if len(fc) == 0:
            raise FileNotFoundError("no hist file found")
        if len(fc) != 1:
            raise ValueError("more than one hist file found")

        return fc.paths.pop()

    def load_hist(meta, fc_hist):
        fN = _get_hist_path(meta, fc_hist)
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(fN, decode_times=time_coder)

    def load_hist_scen_continuous(fc_hist, fc_scens):
        dt = xr.DataTree()
        for scen in fc_scens.df.scenario.unique():
            dt[scen] = xr.DataTree()
            for var in fc_scens.df.variable.unique():
                files = fc_scens.search(variable=var, scenario=scen)

                members = []

                for fN, meta in files.items():

                    try:
                        hist = load_hist(meta, fc_hist)
                    except FileNotFoundError:
                        continue

                    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                    proj = xr.open_dataset(fN, decode_times=time_coder)

                    ds = xr.combine_by_coords(
                        [hist, proj],
                        combine_attrs="override",
                        data_vars="minimal",
                        compat="override",
                        coords="minimal",
                    )

                    ds = ds.drop_vars(
                        ("height", "time_bnds", "file_qf", "area"), errors="ignore"
                    )

                    ds = mesmer.grid.wrap_to_180(ds)

                    # assign member-ID as coordinate
                    ds = ds.assign_coords({"member": meta["member"]})

                    members.append(ds)

                # create a Dataset that holds each member along the member dimension
                scen_data = xr.concat(members, dim="member")
                # put the scenario dataset into the DataTree
                dt[scen] = dt[scen].assign({f"{var}": scen_data[var]})
        return dt

    data = load_hist_scen_continuous(fc_hist, fc_scens)
    ref = data.sel(time=REFERENCE_PERIOD).mean("time")
    anoms = data - ref
    globmean = mesmer.weighted.global_mean(anoms)

    globmean_ensmean = globmean.mean(dim="member")
    globmean_forcing = mesmer.stats.lowess(
        globmean_ensmean, dim="time", n_steps=30, use_coords=False
    )

    return globmean_forcing


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
        ),
        # tas and hfds
        pytest.param(
            ["ssp126"],
            False,
            True,
            1,
            "tas_hfds/one_scen_one_ens",
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
        ),
        pytest.param(
            ["ssp126", "ssp585"],
            True,
            True,
            1,
            "tas_tas2_hfds/multi_scen_multi_ens",
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
        path_pattern=TEST_PATH / "test-params/{module}/",
        file_pattern="params_{module}_{esm}_{scen}.nc",
    )
    scen_str = "-".join(scenarios)

    volcanic_file = PARAM_FILEFINDER.find_single_file(
        module="volcanic", esm=esm, scen=scen_str
    ).paths.pop()
    global_ar_file = PARAM_FILEFINDER.find_single_file(
        module="global-variability", esm=esm, scen=scen_str
    ).paths.pop()
    local_forced_file = PARAM_FILEFINDER.find_single_file(
        module="local-trends", esm=esm, scen=scen_str
    ).paths.pop()
    local_ar_file = PARAM_FILEFINDER.find_single_file(
        module="local-variability", esm=esm, scen=scen_str
    ).paths.pop()
    localized_ecov_file = PARAM_FILEFINDER.find_single_file(
        module="covariance", esm=esm, scen=scen_str
    ).paths.pop()

    expected_output_file = TEST_PATH / f"test_realisations_expected_{scen_str}.nc"

    HIST_PERIOD = slice("1850", "2014")

    # we need a maximum of 4 seeds if there are max 2 scenarios (1 for global and 1 for local)
    seed_list = [981, 314, 272, 42]

    seed_global_variability = xr.DataTree()
    seed_local_variability = xr.DataTree()

    for scen in scenarios:
        seed_global_variability[scen] = xr.DataTree(
            xr.Dataset({"seed": xr.DataArray(seed_list.pop())})
        )
        seed_local_variability[scen] = xr.DataTree(
            xr.Dataset({"seed": xr.DataArray(seed_list.pop())})
        )

    forcing_data = create_forcing_data(scenarios, use_hfds)
    scen0 = scenarios[0]
    time = forcing_data[scen0].time

    buffer_global_variability = 50
    buffer_local_variability = 20

    # 1.) superimpose volcanic influence
    volcanic_params = xr.open_dataset(volcanic_file)

    for scen in scenarios:
        forcing_data[scen]["tas"] = mesmer.volc.superimpose_volcanic_influence(
            forcing_data[scen]["tas"],
            volcanic_params,
            hist_period=HIST_PERIOD,
        )

    if use_tas2:
        forcing_data = map_over_datasets(
            lambda ds: ds.assign(tas2=ds.tas**2), forcing_data
        )

    # 2.) compute the global variability
    global_ar_params = xr.open_dataset(global_ar_file)
    global_variability = mesmer.stats.draw_auto_regression_uncorrelated(
        global_ar_params,
        realisation=n_realisations,
        time=time,
        seed=seed_global_variability,
        buffer=buffer_global_variability,
    )

    global_variability = map_over_datasets(
        xr.Dataset.rename, global_variability, {"samples": "tas_resids"}
    )

    # 3.) compute the local forced response
    lr = mesmer.stats.LinearRegression().from_netcdf(local_forced_file)

    predictors = mesmer.datatree.merge([forcing_data, global_variability])

    # uses ``exclude`` to split the linear response
    local_forced_response = lr.predict(predictors, exclude={"tas_resids"})

    # 4.) compute the local variability part driven by global variabilty
    exclude = {"tas", "intercept"}
    if use_tas2:
        exclude.add("tas2")
    if use_hfds:
        exclude.add("hfds")
    local_variability_from_global_var = lr.predict(predictors, exclude=exclude)

    # 5.) compute local variability
    local_ar = xr.open_dataset(local_ar_file)
    localized_covariance = xr.open_dataset(localized_ecov_file)
    localized_covariance_adjusted = localized_covariance.localized_covariance_adjusted

    local_variability = mesmer.stats.draw_auto_regression_correlated(
        local_ar,
        localized_covariance_adjusted,
        time=time,
        realisation=n_realisations,
        seed=seed_local_variability,
        buffer=buffer_local_variability,
    )

    # rename datavar in set to be able to add with linear regression predictions
    local_variability = map_over_datasets(
        xr.Dataset.rename, local_variability, kwargs={"samples": "prediction"}
    )

    local_variability_total = local_variability_from_global_var + local_variability

    result = local_forced_response + local_variability_total
    result = map_over_datasets(xr.Dataset.rename, result, kwargs={"prediction": "tas"})

    # save the emulations
    if update_expected_files:
        result.to_netcdf(expected_output_file)
        pytest.skip("Updated expected output file.")
    else:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        expected = xr.open_datatree(expected_output_file, decode_times=time_coder)
        for scen in scenarios:
            exp_scen = expected[scen].to_dataset()
            res_scen = result[scen].to_dataset()
            xr.testing.assert_allclose(res_scen, exp_scen)

            # make sure we can get onto a lat lon grid from what is saved
            exp_reshaped = exp_scen.set_index(gridcell=("lat", "lon")).unstack(
                "gridcell"
            )
            expected_dims = {"realisation", "lon", "lat", "time"}

            assert set(exp_reshaped.dims) == expected_dims
