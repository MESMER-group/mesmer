import os.path

import joblib
import numpy as np
import numpy.testing as npt
import sklearn.linear_model
import xarray as xr
import xarray.testing as xrt

from mesmer.calibrate_mesmer import _calibrate_mesmer


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

    _calibrate_mesmer(
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
    exp = joblib.load(expected_output_file)

    assert isinstance(res, dict)
    assert type(res) == type(exp)
    assert res.keys() == exp.keys()

    def _check_dict(res, exp):
        for k in res:
            res_val = res[k]
            exp_val = exp[k]

            assert type(res_val) == type(exp_val)
            if isinstance(res_val, dict):
                _check_dict(res_val, exp_val)
            elif isinstance(res_val, sklearn.linear_model.LinearRegression):
                # not sure if there's a better way to test this...
                npt.assert_allclose(res_val.coef_, exp_val.coef_)
            elif isinstance(res_val, np.ndarray):
                npt.assert_allclose(res_val, exp_val)
            elif isinstance(res_val, xr.DataArray):
                xrt.assert_allclose(res_val, exp_val)
            else:
                assert res_val == exp_val, k

    _check_dict(res, exp)
