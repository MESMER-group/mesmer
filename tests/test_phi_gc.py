import numpy as np
import pytest

from mesmer.io import (
    calc_geodist_exact,
    gaspari_cohn,
    load_phi_gc,
    load_regs_ls_wgt_lon_lat,
)


class mock_cfg:
    def __init__(self, tmp_path, threshold_land):
        self.dir_aux = str(tmp_path) + "/"
        self.threshold_land = threshold_land


def test_phi_gc_end_to_end(tmp_path):

    lat = dict(c=np.arange(15, -1, -10.0))
    lon = dict(c=np.arange(5, 21, 10.0))

    # prepare the necessary data
    # TODO: potentially simplify - test all points that are passed
    reg_dict, ls, wgt, lon, lat = load_regs_ls_wgt_lon_lat("srex", lon, lat)

    threshold = 0.6
    cfg = mock_cfg(tmp_path, threshold)
    ls["idx_grid_l"] = ls["grid_no_ANT"] > threshold

    actual = load_phi_gc(lon, lat, ls, cfg, L_start=700, L_end=2000, L_interval=1000)

    expected_700 = np.array(
        [[1.0, 0.01237, 0.0], [0.01237, 1.0, 0.00845111], [0.0, 0.00845111, 1.0]]
    )

    expected_1700 = np.array(
        [
            [1.0, 0.5460012, 0.27314733],
            [0.5460012, 1.0, 0.52703897],
            [0.27314733, 0.52703897, 1.0],
        ]
    )

    np.testing.assert_allclose(expected_700, actual[700], rtol=1e-5)
    np.testing.assert_allclose(expected_1700, actual[1700], rtol=1e-5)

    # get one more landpoint

    threshold = 0.5
    cfg = mock_cfg(tmp_path, threshold)
    ls["idx_grid_l"] = ls["grid_no_ANT"] > threshold

    actual = load_phi_gc(lon, lat, ls, cfg, L_start=1000, L_end=1000, L_interval=1000)

    expected = np.array(
        [
            [1.0, 0.15897513, 0.1412142, 0.01044674],
            [0.15897513, 1.0, 0.01044674, 0.1412142],
            [0.1412142, 0.01044674, 1.0, 0.13962033],
            [0.01044674, 0.1412142, 0.13962033, 1.0],
        ]
    )
    np.testing.assert_allclose(expected, actual[1000], rtol=1e-5)


def test_gaspari_cohn():

    assert gaspari_cohn(0) == 1
    assert gaspari_cohn(2) == 0

    values = np.arange(0, 2.1, 0.5)
    expected = np.array([1.0, 0.68489583, 0.20833333, 0.01649306, 0.0])

    actual = gaspari_cohn(values)
    np.testing.assert_allclose(expected, actual, rtol=1e-6)

    # the function is symmetric around 0
    actual = gaspari_cohn(-values)
    np.testing.assert_allclose(expected, actual, rtol=1e-6)

    # make sure shape is conserved
    values = np.arange(9).reshape(3, 3)
    assert gaspari_cohn(values).shape == (3, 3)


def test_calc_geodist_exact_shape():

    msg = "lon and lat need to be 1D arrays of the same shape"

    # not the same shape
    with pytest.raises(ValueError, match=msg):
        calc_geodist_exact([0, 0], [0])

    # not 1D
    with pytest.raises(ValueError, match=msg):
        calc_geodist_exact([[0, 0]], [[0, 0]])


def test_calc_geodist_exact_equal():
    """test points with distance 0"""

    expected = np.array([[0, 0], [0, 0]])

    lat = [0, 0]
    lons = [[0, 0], [0, 360], [1, 361], [180, -180]]

    for lon in lons:
        result = calc_geodist_exact(lon, lat)
        np.testing.assert_equal(result, expected)

    result = calc_geodist_exact(lon, lat)
    np.testing.assert_equal(result, expected)


def test_calc_geodist_exact():
    """test some random points"""
    result = calc_geodist_exact([-180, 0, 3], [0, 0, 5])
    expected = np.array(
        [
            [0.0, 20003.93145863, 19366.51816487],
            [20003.93145863, 0.0, 645.70051988],
            [19366.51816487, 645.70051988, 0.0],
        ]
    )

    np.testing.assert_allclose(result, expected)
