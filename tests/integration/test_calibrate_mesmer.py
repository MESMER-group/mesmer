import os.path
import shutil

import joblib
import numpy as np
import numpy.testing as npt
import sklearn.linear_model
import xarray as xr
import xarray.testing as xrt

from mesmer.calibrate_mesmer import _calibrate_and_draw_realisations


def _check_dict(first, second, first_name, second_name):
    for k in first:
        first_val = first[k]
        try:
            second_val = second[k]
        except KeyError:
            raise AssertionError(
                "Key `{}` is in '{}' but is not in '{}'".format(k, first_name, second_name)
            )

        assert type(first_val) == type(second_val)
        if isinstance(first_val, dict):
            _check_dict(first_val, second_val, first_name, second_name)
        elif isinstance(first_val, sklearn.linear_model.LinearRegression):
            # not sure if there's a better way to test this...
            npt.assert_allclose(first_val.coef_, second_val.coef_)
        elif isinstance(first_val, np.ndarray):
            npt.assert_allclose(first_val, second_val)
        elif isinstance(first_val, xr.DataArray):
            xrt.assert_allclose(first_val, second_val)
        else:
            assert first_val == second_val, k


def test_calibrate_mesmer(test_data_root_dir, tmpdir, update_expected_files):
    expected_output_file = os.path.join(test_data_root_dir, "test-mesmer-bundle.pkl")

    test_esms = ["IPSL-CM6A-LR"]
    test_scenarios_to_train = ["h-ssp126"]
    test_target_variable = "tas"
    test_reg_type = "srex"
    test_threshold_land = 1 / 3
    test_output_file = os.path.join(tmpdir, "test_calibrate_mesmer_output.pkl")
    test_scen_seed_offset_v = 0
    test_cmip_generation = 6
    test_cmip_data_root_dir = os.path.join(
        test_data_root_dir,
        "calibrate-coarse-grid",
        "cmip{}-ng".format(test_cmip_generation),
    )
    test_observations_root_dir = os.path.join(
        test_data_root_dir,
        "calibrate-coarse-grid",
        "observations",
    )
    test_auxiliary_data_dir = os.path.join(
        test_data_root_dir,
        "calibrate-coarse-grid",
        "auxiliary",
    )

    _calibrate_and_draw_realisations(
        esms=test_esms,
        scenarios_to_train=test_scenarios_to_train,
        target_variable=test_target_variable,
        reg_type=test_reg_type,
        threshold_land=test_threshold_land,
        output_file=test_output_file,
        scen_seed_offset_v=test_scen_seed_offset_v,
        cmip_data_root_dir=test_cmip_data_root_dir,
        cmip_generation=test_cmip_generation,
        observations_root_dir=test_observations_root_dir,
        auxiliary_data_dir=test_auxiliary_data_dir,
    )

    res = joblib.load(test_output_file)

    if update_expected_files:
        shutil.copyfile(test_output_file, expected_output_file)
    else:
        exp = joblib.load(expected_output_file)

        assert isinstance(res, dict)
        assert type(res) == type(exp)
        assert res.keys() == exp.keys()

        # check all keys of res match exp
        _check_dict(res, exp, "result", "expected")
        # check all keys of exp match res
        _check_dict(exp, res, "expected", "result")
