import os

import numpy as np
import pytest

import mesmer
from mesmer.io._load_cmipng import _load_cmipng_var
from mesmer.testing import assert_dict_allclose


def _get_default_kwargs(
    test_data_root_dir,
    esm="IPSL-CM6A-LR",
    scen="ssp126",
    type="all",
    start="1850",
    end="2100",
    gen=6,
):

    # get data path
    test_cmip_data_root_dir = os.path.join(
        test_data_root_dir,
        "calibrate-coarse-grid",
        f"cmip{gen}-ng",
    )

    # mock cfg class
    class cfg:
        def __init__(self):
            self.gen = gen
            self.ref = dict(type=type, start=start, end=end)
            self.dir_cmipng = test_cmip_data_root_dir

    return {"esm": esm, "scen": scen, "cfg": cfg()}


def test_load_cmipng_var_missing_data():

    dta, dta_global, lon, lat, time = _load_cmipng_var(
        varn="tas", **_get_default_kwargs("", esm="missing")
    )

    assert dta is None
    assert dta_global is None
    assert lon is None
    assert lat is None
    assert time is None


def test_load_cmipng_var_type_first_error(test_data_root_dir):

    with pytest.raises(ValueError, match="reference type 'first'"):
        _load_cmipng_var(
            varn="tas", **_get_default_kwargs(test_data_root_dir, type="first")
        )


@pytest.mark.parametrize("varn", ["tas", "hfds"])
def test_load_cmipng(test_data_root_dir, varn):

    dta, dta_global, lon, lat, time = _load_cmipng_var(
        varn=varn, **_get_default_kwargs(test_data_root_dir)
    )

    for d in [dta, dta_global, lon, lat]:
        assert isinstance(d, dict)

    time_expected = np.arange(1850, 2100 + 1)
    lon_expected = np.arange(-180, 180, 18)
    lat_expected = np.arange(-85.5, 90, 9)

    # check expected shapes
    assert dta[1].shape == (251, 20, 20)
    assert dta_global[1].shape == (251,)

    np.testing.assert_allclose(lon["c"], lon_expected)
    np.testing.assert_allclose(lat["c"], lat_expected)
    np.testing.assert_allclose(time, time_expected)


@pytest.mark.slow
@pytest.mark.parametrize("varn", ["tas", "hfds"])
@pytest.mark.parametrize("start, end", [("1850", "1900"), ("1850", "1860")])
def test_load_cmipng_ref_two_ens_all(test_data_root_dir, varn, start, end):

    dta, dta_global, lon, lat, time = _load_cmipng_var(
        varn=varn,
        **_get_default_kwargs(test_data_root_dir, scen="ssp585", start=start, end=end),
    )
    time = np.asarray(time)
    sel = (time >= int(start)) & (time <= int(end))

    # stack ensemble members, new axis at end
    dta_arr = np.stack(list(dta.values()), axis=-1)
    dta_global_arr = np.stack(list(dta_global.values()), axis=-1)

    # test mean over all grid points is 0 (over reference period AND ensembles)
    np.testing.assert_allclose(0, np.nanmean(dta_arr[sel, :, :]), atol=1e-6)

    # test individual gridpoints are 0 (over reference period)
    result = np.mean(dta_arr[sel, :, :], axis=(0, -1))

    expected = np.zeros_like(result)
    expected[np.isnan(result)] = np.NaN
    np.testing.assert_allclose(expected, result, atol=1e-6)

    # test global mean is 0 (over reference period)
    np.testing.assert_allclose(0, np.mean(dta_global_arr[sel]), atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("varn", ["tas", "hfds"])
@pytest.mark.parametrize("start, end", [("1850", "1900"), ("1850", "1860")])
def test_load_cmipng_ref_two_ens_indiv(test_data_root_dir, varn, start, end):

    dta, dta_global, lon, lat, time = _load_cmipng_var(
        varn=varn,
        **_get_default_kwargs(
            test_data_root_dir, scen="ssp585", type="individ", start=start, end=end
        ),
    )
    time = np.asarray(time)
    sel = (time >= int(start)) & (time <= int(end))

    # test mean over all grid points is 0 (over reference period)
    for arr in dta.values():
        np.testing.assert_allclose(0, np.nanmean(arr[sel, :, :]), atol=1e-6)

    # test individual gridpoints are 0 (over reference period)
    for arr in dta.values():
        result = np.mean(arr[sel, :, :], axis=0)
        expected = np.zeros_like(result)
        expected[np.isnan(result)] = np.NaN
        np.testing.assert_allclose(expected, result, atol=1e-6)

    # test global mean is 0 (over reference period)
    for arr_global in dta_global.values():
        np.testing.assert_allclose(0, np.mean(arr_global[sel]), atol=1e-6)


@pytest.mark.parametrize("varn", ["tas", "hfds"])
@pytest.mark.parametrize("start, end", [("1850", "1900"), ("1850", "1860")])
def test_load_cmipng_ref(test_data_root_dir, varn, start, end):

    dta, dta_global, lon, lat, time = _load_cmipng_var(
        varn=varn, **_get_default_kwargs(test_data_root_dir, start=start, end=end)
    )
    time = np.asarray(time)
    sel = (time >= int(start)) & (time <= int(end))

    # test mean over all grid points is 0 (over reference period)
    np.testing.assert_allclose(0, np.nanmean(dta[1][sel, :, :]), atol=1e-6)

    # test individual gridpoints are 0 (over reference period)
    result = np.mean(dta[1][sel, :, :], axis=0)
    expected = np.zeros_like(result)
    expected[np.isnan(result)] = np.NaN
    np.testing.assert_allclose(expected, result, atol=1e-6)

    # test global mean is 0 (over reference period)
    np.testing.assert_allclose(0, np.mean(dta_global[1][sel]), atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("varn", ["tas", "hfds"])
def test_load_cmimpng_vs_load_var(test_data_root_dir, varn):
    # compare data loaded with mesmer.io._load_cmipng._load_cmipng_var to
    # mesmer.io.load_cmipng.load_cmipng_{varn}

    # load the data indirectly
    dta_i, dta_global_i, lon_i, lat_i, time_i = _load_cmipng_var(
        varn=varn, **_get_default_kwargs(test_data_root_dir)
    )

    # load data directly
    func = getattr(mesmer.io, f"load_cmipng_{varn}")
    dta_d, dta_global_d, lon_d, lat_d, time_d = func(
        **_get_default_kwargs(test_data_root_dir)
    )

    assert_dict_allclose(dta_d, dta_i, "direct", "indirect")
    assert_dict_allclose(dta_global_d, dta_global_i, "direct", "indirect")
    assert_dict_allclose(lon_d, lon_i, "direct", "indirect")
    assert_dict_allclose(lat_d, lat_i, "direct", "indirect")
    np.testing.assert_allclose(time_d, time_i)
