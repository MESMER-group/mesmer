import numpy as np

from mesmer.mesmer_x._weighting import get_weights_density, get_weights_uniform


def test_get_weights_uniform():
    n = 3

    weights_unif = get_weights_uniform(targ_data=np.arange(n), target=None, dims=None)

    np.testing.assert_equal(weights_unif, np.ones(n) / n)


def test_get_weights_density():
    n = 3
    weights_dens = get_weights_density(
        pred_data=np.arange(n),
        predictor=None,
        targ_data=np.arange(n),
        target=None,
        dims=None,
    )

    np.testing.assert_equal(weights_dens, weights_dens / np.sum(weights_dens))
