from mesmer.create_emulations import create_seed_dict
from mesmer.testing import assert_dict_allclose


def test_create_seed_dict():

    result = create_seed_dict(["a", "b"], ["ssp585", "ssp119"])

    expected = {
        "a": {"all": {"gv": 0, "lv": 1000000}},
        "b": {"all": {"gv": 1, "lv": 1000001}},
    }

    assert_dict_allclose(result, expected)


def test_create_seed_dict_scen_seed_offset():

    result = create_seed_dict(["a", "b"], ["ssp585", "ssp119"], scen_seed_offset=6)

    expected = {
        "a": {"ssp585": {"gv": 0, "lv": 1000000}, "ssp119": {"gv": 6, "lv": 1000006}},
        "b": {"ssp585": {"gv": 1, "lv": 1000001}, "ssp119": {"gv": 7, "lv": 1000007}},
    }

    assert_dict_allclose(result, expected)
