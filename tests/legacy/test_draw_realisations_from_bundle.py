import os.path
import sys

import joblib
import numpy as np
import pytest
import xarray as xr

import mesmer.create_emulations

# TODO:
# - write test of what happens if you pass in a time dictionary of the wrong length

# NOTE: the scenario only changes the params - so we only do "one_scen_one_ens"


# skip slow legacy tests on windows
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="slow on windows")


@pytest.mark.parametrize(
    "use_tas2, use_hfds, n_realisations, outname",
    (
        # tas
        pytest.param(
            False,
            False,
            1,
            "tas/one_scen_one_ens",
        ),
        # tas, tas**2, and hfds
        pytest.param(
            True,
            True,
            30,
            "tas_tas2_hfds/one_scen_one_ens",
        ),
    ),
)
def test_make_realisations(
    use_tas2,
    use_hfds,
    n_realisations,
    outname,
    test_data_root_dir,
    update_expected_files,
):

    output_dir = os.path.join(test_data_root_dir, "output", outname)

    expected_output_file = os.path.join(
        output_dir, "test_make_realisations_expected_output_legacy.nc"
    )

    tseeds = {"IPSL-CM6A-LR": {"all": {"gv": 0, "lv": 1_000_000}}}

    bundle_path = os.path.join(output_dir, "test-mesmer-bundle.pkl")

    # TODO: split out load_mesmer_bundle function
    mesmer_bundle = joblib.load(bundle_path)

    params_lt = mesmer_bundle["params_lt"]
    params_lv = mesmer_bundle["params_lv"]
    params_gv_T = mesmer_bundle["params_gv"]
    land_fractions = mesmer_bundle["land_fractions"]

    # TODO: split out function to format e.g. ScmRun correctly (put in
    # mesmer-magicc repo, not here)
    # can hardcode scenario for now
    scenario = "ssp126"
    ttime = {
        "hist": np.arange(1850, 2014 + 1),
        scenario: np.arange(2015, 2100 + 1),
    }
    hist_length = len(ttime["hist"])
    scen_length = len(ttime[scenario])

    hist_tas = np.linspace(0, 1, hist_length)
    scen_tas = np.linspace(1, 2, scen_length)

    preds_lt = {"gttas": {"hist": hist_tas, scenario: scen_tas}}

    if use_tas2:
        preds_lt["gttas2"] = {"hist": hist_tas**2, scenario: scen_tas**2}

    if use_hfds:
        hist_hfds = np.linspace(0, 2, hist_length)
        scen_hfds = np.linspace(2, 3, scen_length)

        preds_lt["gthfds"] = {"hist": hist_hfds, scenario: scen_hfds}

    result = mesmer.create_emulations.make_realisations(
        preds_lt,
        params_lt,
        params_lv,
        params_gv_T,
        n_realisations=n_realisations,
        land_fractions=land_fractions,
        seeds=tseeds,
        time=ttime,
    )

    if update_expected_files:
        result.to_netcdf(expected_output_file)
        pytest.skip(f"Updated {expected_output_file}")

    else:
        exp = xr.open_dataset(expected_output_file)

        xr.testing.assert_allclose(result, exp)

        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = exp.set_index(z=("lat", "lon")).unstack("z")
        expected_dims = {"scenario", "realisation", "lon", "lat", "year"}

        assert set(exp_reshaped.dims) == expected_dims
