import numpy as np
import xarray as xr

from mesmer.core.utils import _check_dataset_form, _check_dataarray_form
from mesmer.mesmer_x._weighting import get_weights_density, weighted_median

def test_get_weights_density():
    n = 3
    
    weights = get_weights_density(
        pred_data=np.arange(n),
    )

    np.testing.assert_equal(weights, weights / np.sum(weights))


def test_get_weights_density_ds():
    pred_data = xr.Dataset({
        "predictor1": (("x", "y"), np.arange(9).reshape(3, 3)),
        "predictor2": (("x", "y"), np.arange(9).reshape(3, 3)),
    })

    weights = get_weights_density(
        pred_data=pred_data,
    )

    _check_dataset_form(weights, "weights", required_vars=["weights"])
    _check_dataarray_form(weights.weights, "weights", required_dims = ("x", "y"), shape=(3, 3))


def test_get_weights_density_dt():
    pred_data = xr.DataTree.from_dict({
        "scenario1": xr.Dataset({
            "predictor1": (("x", "y"), np.arange(12).reshape(3, 4)),
            "predictor2": (("x", "y"), np.arange(12).reshape(3, 4)),
        }),
        "scenario2": xr.Dataset({
            "predictor1": (("x", "y"), np.arange(12).reshape(3, 4)),
            "predictor2": (("x", "y"), np.arange(12).reshape(3, 4)),
        }),
    })

    weights = get_weights_density(
        pred_data=pred_data,
    )

    scen1 = weights["scenario1"].to_dataset()
    scen2 = weights["scenario2"].to_dataset()

    _check_dataset_form(scen1, "weights", required_vars=["weights"])
    _check_dataarray_form(scen1.weights, "weights", required_dims = ("x", "y"), shape=(3, 4))
    
    _check_dataset_form(scen2, "weights", required_vars=["weights"])
    _check_dataarray_form(scen2.weights, "weights", required_dims = ("x", "y"), shape=(3, 4))


def test_weighted_median():
    data = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

    median = weighted_median(data, weights)

    # The weighted median should be the value that splits the data into two halves
    expected_median = 3
    np.testing.assert_equal(median, expected_median)