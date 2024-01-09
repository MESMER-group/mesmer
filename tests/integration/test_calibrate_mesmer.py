import os.path
import shutil

import joblib
import pytest

from mesmer.calibrate_mesmer import _calibrate_tas
from mesmer.testing import assert_dict_allclose


@pytest.mark.filterwarnings("ignore:No local minimum found")
@pytest.mark.parametrize(
    "scenarios, use_tas2, use_hfds, outname",
    (
        # tas
        pytest.param(
            ["h-ssp126"],
            False,
            False,
            "tas/one_scen_one_ens",
        ),
        pytest.param(
            ["h-ssp585"],
            False,
            False,
            "tas/one_scen_multi_ens",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ["h-ssp126", "h-ssp585"],
            False,
            False,
            "tas/multi_scen_multi_ens",
        ),
        # tas and tas**2
        pytest.param(
            ["h-ssp126"],
            True,
            False,
            "tas_tas2/one_scen_one_ens",
            marks=pytest.mark.slow,
        ),
        # tas and hfds
        pytest.param(
            ["h-ssp126"],
            False,
            True,
            "tas_hfds/one_scen_one_ens",
            marks=pytest.mark.slow,
        ),
        # tas, tas**2, and hfds
        pytest.param(
            ["h-ssp126"],
            True,
            True,
            "tas_tas2_hfds/one_scen_one_ens",
        ),
        pytest.param(
            ["h-ssp585"],
            True,
            True,
            "tas_tas2_hfds/one_scen_multi_ens",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ["h-ssp126", "h-ssp585"],
            True,
            True,
            "tas_tas2_hfds/multi_scen_multi_ens",
            marks=pytest.mark.slow,
        ),
    ),
)
def test_calibrate_mesmer(
    scenarios,
    use_tas2,
    use_hfds,
    outname,
    test_data_root_dir,
    tmpdir,
    update_expected_files,
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
    test_auxiliary_data_dir = os.path.join(
        test_data_root_dir,
        "calibrate-coarse-grid",
        "auxiliary",
    )

    _calibrate_tas(
        esms=test_esms,
        scenarios_to_train=test_scenarios_to_train,
        threshold_land=test_threshold_land,
        output_file=test_output_file,
        cmip_data_root_dir=test_cmip_data_root_dir,
        cmip_generation=test_cmip_generation,
        auxiliary_data_dir=test_auxiliary_data_dir,
        use_tas2=use_tas2,
        use_hfds=use_hfds,
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
        assert type(res) == type(exp)  # noqa: E721
        assert res.keys() == exp.keys()

        # check all keys of res match exp
        assert_dict_allclose(res, exp, "result", "expected")
        # check all keys of exp match res
        assert_dict_allclose(exp, res, "expected", "result")
