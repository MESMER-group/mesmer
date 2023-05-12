import os.path
import shutil

import joblib
import pytest

from mesmer.calibrate_mesmer import _calibrate_and_draw_realisations
from mesmer.testing import assert_dict_allclose


@pytest.mark.filterwarnings("ignore:No local minimum found")
@pytest.mark.parametrize(
    "scenarios, outname",
    (
        [["h-ssp126"], "one_scen_one_ens"],
        [["h-ssp585"], "one_scen_multi_ens"],
        [["h-ssp126", "h-ssp585"], "multi_scen_multi_ens"],
    ),
)
def test_calibrate_mesmer(
    scenarios, outname, test_data_root_dir, tmpdir, update_expected_files
):

    ouput_dir = os.path.join(test_data_root_dir, "output", outname)

    expected_output_file = os.path.join(ouput_dir, "test-mesmer-bundle.pkl")
    params_output_dir = os.path.join(ouput_dir, "params")

    test_esms = ["IPSL-CM6A-LR"]
    test_scenarios_to_train = scenarios
    test_threshold_land = 1 / 3
    test_output_file = os.path.join(tmpdir, "test_calibrate_mesmer_output.pkl")
    test_cmip_generation = 6
    test_cmip_data_root_dir = os.path.join(
        test_data_root_dir,
        "calibrate-coarse-grid",
        f"cmip{test_cmip_generation}-ng",
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
        threshold_land=test_threshold_land,
        output_file=test_output_file,
        cmip_data_root_dir=test_cmip_data_root_dir,
        cmip_generation=test_cmip_generation,
        observations_root_dir=test_observations_root_dir,
        auxiliary_data_dir=test_auxiliary_data_dir,
        # save params as well - they are .gitignored
        save_params=update_expected_files,
        params_output_dir=params_output_dir,
    )

    res = joblib.load(test_output_file)

    if update_expected_files:
        shutil.copyfile(test_output_file, expected_output_file)
        pytest.skip(f"Updated {expected_output_file}")
    else:
        exp = joblib.load(expected_output_file)

        assert isinstance(res, dict)
        assert type(res) == type(exp)
        assert res.keys() == exp.keys()

        # check all keys of res match exp
        assert_dict_allclose(res, exp, "result", "expected")
        # check all keys of exp match res
        assert_dict_allclose(exp, res, "expected", "result")
