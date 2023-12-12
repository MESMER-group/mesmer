import os.path

import joblib
import numpy as np
import pytest
import xarray as xr

import mesmer.create_emulations

# TODO:
# - write test of what happens if you pass in a time dictionary of the wrong length


def test_make_realisations(test_data_root_dir, update_expected_files):

    ouput_dir = os.path.join(
        test_data_root_dir, "output", "tas_tas2_hfds", "one_scen_one_ens"
    )

    expected_output_file = os.path.join(
        ouput_dir, "test_make_realisations_expected_output.nc"
    )

    tseeds = {"IPSL-CM6A-LR": {"all": {"gv": 0, "lv": 1_000_000}}}

    bundle_path = os.path.join(ouput_dir, "test-mesmer-bundle.pkl")

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
    hist_hfds = np.linspace(0, 2, hist_length)
    scen_hfds = np.linspace(2, 3, scen_length)

    preds_lt = {
        "gttas": {"hist": hist_tas, scenario: scen_tas},
        "gttas2": {"hist": hist_tas**2, scenario: scen_tas**2},
        "gthfds": {"hist": hist_hfds, scenario: scen_hfds},
    }

    result = mesmer.create_emulations.make_realisations(
        preds_lt,
        params_lt,
        params_lv,
        params_gv_T,
        n_realisations=30,
        land_fractions=land_fractions,
        seeds=tseeds,
        time=ttime,
    )

    if update_expected_files:
        result.to_netcdf(expected_output_file)
        pytest.skip(f"Updated {expected_output_file}")

    else:
        exp = xr.open_dataset(expected_output_file)

        rtol = 1e-4
        wrong_tol = 0.1
        for v in exp.data_vars:
            # check that less than 10% of output differs. Something weird is
            # happening with numpy's random seed (we get different values
            # depending on the operating system) so we currently can't do any
            # better than this.
            differing_spots = np.sum(
                ~np.isclose(result[v].values, exp[v].values, rtol=rtol)
            )
            frac_differing = differing_spots / result[v].values.size
            assert frac_differing < wrong_tol

        # # Ideally we would use the below, but we can't because of numpy's
        # # random seed issue (see comment above).
        # xr.testing.assert_allclose(result, exp, rtol=rtol)

        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = exp.set_index(z=("lat", "lon")).unstack("z")
        expected_dims = {"scenario", "realisation", "lon", "lat", "year"}

        assert set(exp_reshaped.dims) == expected_dims
