import numpy as np

from mesmer.io import load_phi_gc, load_regs_ls_wgt_lon_lat


class mock_cfg:
    def __init__(self, tmp_path, threshold_land):
        self.dir_aux = str(tmp_path)
        self.threshold_land = threshold_land


def test_phi_gc_end_to_end(tmp_path):

    lat = dict(c=np.arange(15, -1, -10.0))
    lon = dict(c=np.arange(5, 21, 10.0))

    # prepare the necessary data
    _, ls, __, lon, lat = load_regs_ls_wgt_lon_lat(lon=lon, lat=lat)

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
