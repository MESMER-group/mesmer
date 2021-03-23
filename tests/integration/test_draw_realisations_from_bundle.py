import os.path

import numpy as np
import pytest
import xarray as xr

import mesmer.create_emulations


def test_make_realisations(
    test_data_root_dir, test_mesmer_bundle, update_expected_files
):
    expected_output_file = os.path.join(
        test_data_root_dir,
        "test_make_realisations_expected_output.nc",
    )

    # TODO: split out load_mesmer_bundle function
    params_lt = test_mesmer_bundle["params_lt"]
    params_lv = test_mesmer_bundle["params_lv"]
    params_gv_T = test_mesmer_bundle["params_gv_T"]
    seeds = test_mesmer_bundle["seeds"]
    land_fractions = test_mesmer_bundle["land_fractions"]
    time = test_mesmer_bundle["time"]

    # TODO: split out function to format e.g. ScmRun correctly (put in
    # mesmer-magicc repo, not here)
    hist_length = 165
    scen_length = 86

    hist_tas = np.linspace(0, 1, hist_length)
    scen_tas = np.linspace(1, 2, scen_length)
    hist_hfds = np.linspace(0, 2, hist_length)
    scen_hfds = np.linspace(2, 3, scen_length)

    preds_lt = {
        "gttas": {"hist": hist_tas, "ssp126": scen_tas},
        "gttas2": {"hist": hist_tas ** 2, "ssp126": scen_tas ** 2},
        "gthfds": {"hist": hist_hfds, "ssp126": scen_hfds},
    }

    result = mesmer.create_emulations.make_realisations(
        preds_lt,
        params_lt,
        params_lv,
        params_gv_T,
        n_realisations=30,
        seeds=seeds,
        land_fractions=land_fractions,
        time=time,
    )

    if update_expected_files:
        result.to_netcdf(expected_output_file)
        pytest.skip(f"Updated {expected_output_file}")

    else:
        exp = xr.open_dataset(expected_output_file)
        assert result.identical(exp)
        # make sure we can get onto a lat lon grid from what is saved
        exp_reshaped = exp.set_index(z=("lat", "lon")).unstack("z")
        assert set(exp_reshaped.dims) == {
            'scenario', 'realisation', 'lon', 'lat', 'year'
        }
