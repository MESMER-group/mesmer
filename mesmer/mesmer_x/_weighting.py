# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

def get_weights_density(pred_data):
    """
    Generate weights for the sample, based on the inverse of the density of the
    predictors. More precisely, the density of the predictors is represented by a
    multidimensional kernel density estimate using gaussian kernels where each
    dimension is one of the predictors. Subsequently, the weights are the inverse
    of this density of the predictors. Consequently, samples in regions of this
    space with low density will have higher weights, this is, "unusual" samples
    will have more weight.

    Parameters
    ----------
    pred_data : xr.DataTree | xr.Dataset | np.array
        Predictors for the training sample. Each branch must be a scenario,
        with a xarray dataset (time, member). Each predictor is a variable.

    Returns
    -------
    weights : DataTree
        Weights for the sample, based on the inverse of the density of the
        predictors, summing to 1.

    Example
    -------
    TODO

    """

    def _weights(data):
        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd that implies too many assumptions
        histo_kde = gaussian_kde(data)

        # calculating density of points over the sample
        density = histo_kde.pdf(x=data)

        # preparing the output
        return (1 / density) / np.sum(1 / density)
    
    if isinstance(pred_data, xr.DataTree):
        n_scens = len(pred_data)
        scens = list(pred_data.keys())
        preds = list(pred_data[scens[0]].data_vars)
        n_preds = len(preds)
        pred_shape = pred_data[scens[0]][preds[0]].shape

        # reshaping data into array
        pred_arrays = {}
        # need predictors of different scnearios together
        for pred in preds:
            if pred not in pred_arrays:
                 pred_arrays[pred] = np.array([])
            for scen in scens:
                pred_arrays[pred] = np.concatenate([pred_arrays[pred], pred_data[scen][pred].values.flatten()])
        
        array_pred = np.array(list(pred_arrays.values()))
        weights = _weights(array_pred)

        # get original shape back
        weights = weights.reshape(n_scens, *pred_shape)

        weights_dt = xr.DataTree()
        for s, scen in enumerate(scens):
            weights_dt[scen] = xr.Dataset({
            "weights": xr.DataArray(
                weights[s, ...],
                dims=pred_data[scens[0]][preds[0]].dims,
                coords=pred_data[scens[0]][preds[0]].coords,
            )
        })

        return weights_dt


    elif isinstance(pred_data, xr.Dataset):
        preds = list(pred_data.data_vars)
        n_preds = len(preds)
        pred_shape = pred_data[preds[0]].shape

        # reshaping data into array
        array_pred = pred_data.to_array("predictor").values.reshape(n_preds, -1)
        weights = _weights(array_pred)

        # get original shape back
        weights = weights.reshape(*pred_shape)
        weights_ds = xr.Dataset({
            "weights": xr.DataArray(
                weights,
                dims=pred_data[preds[0]].dims,
                coords=pred_data[preds[0]].coords,
            )
        })
        return weights_ds


    elif isinstance(pred_data, np.ndarray):
        array_pred = pred_data
        return _weights(array_pred)



def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights

    Source:
      https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
      @author Jack Peterson (jack@tinybike.net)
    """
    data, weights = np.asarray(data).squeeze(), np.asarray(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)

    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median
