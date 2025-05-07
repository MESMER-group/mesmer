# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde


def get_weights_uniform(targ_data, target, dims):
    """
    Generate uniform weights for the training sample.

    Parameters
    ----------
    targ_data : xr.DataTree | xr.Dataset | np.array
        Target for the training sample. Each branch must be a scenario,
        with a xarray dataset (time, member, gridpoint).

    target : str
        Name of the target. Must be the name in the datasets in targ_data.

    dims : list of str
        Dimensions of the data. Must be the same for all scenarios.

    Returns
    -------
    weights : DataTree
        Weights for the sample, uniform, summing to 1.

    Example
    -------
    TODO

    """
    if isinstance(targ_data, xr.DataTree):
        # preparing a datatree with ones everywhere
        factor_rescale = 0
        out = dict()
        for scen in targ_data:
            # identify the extra dimension
            extra_dims = [
                dim for dim in targ_data[scen][target].dims if dim not in dims
            ]
            locator_extra_dims = {dim: 0 for dim in extra_dims}

            # create a DataArray of ones with the required shape
            ones_array = xr.ones_like(
                targ_data[scen][target].loc[locator_extra_dims], dtype=float
            )
            out[scen] = xr.DataTree(xr.Dataset({"weight": ones_array}))

            # accumulate the total size for rescaling
            factor_rescale += ones_array.size

        # rescale
        return xr.DataTree.from_dict(out) / factor_rescale

    elif isinstance(targ_data, xr.Dataset):
        # identify the extra dimension
        extra_dims = [dim for dim in targ_data[target].dims if dim not in dims]
        locator_extra_dims = {dim: 0 for dim in extra_dims}

        # create a DataArray of ones with the required shape
        ones_array = xr.ones_like(
            targ_data[target].loc[locator_extra_dims], dtype=float
        )

        # rescale
        return xr.Dataset({"weight": ones_array / ones_array.size})

    elif isinstance(targ_data, np.ndarray):
        # create a DataArray of ones with the required shape
        # warning, it assumes that this is performed for a single gridpoint
        ones_array = np.ones(targ_data.shape, dtype=float)

        # rescale
        return ones_array / ones_array.size

    else:
        raise Exception(
            "The format for targ_data must be a xr.DataTree, xr.Dataset or a np.array."
        )


def get_weights_density(pred_data, predictor, targ_data, target, dims):
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

    predictor : str
        Name of the predictor. Must be the name in the datasets in pred_data.

    targ_data : DataTree
        Target for the training sample. Each branch must be a scenario,
        with a xarray dataset (time, member, gridpoint).

    target : str
        Name of the target. Must be the name in the datasets in targ_data.

    dims : list of str
        Dimensions of the data. Must be the same for all scenarios.

    Returns
    -------
    weights : DataTree
        Weights for the sample, based on the inverse of the density of the
        predictors, summing to 1.

    Example
    -------
    TODO

    """

    # checking if predictors have been provided
    if len(pred_data) == 0:
        # NB: may use no predictors when training stationary distributions for bencharmking.
        print("no predictors provided, switching to uniform weights")
        return get_weights_uniform(targ_data, target, dims)

    elif isinstance(targ_data, xr.DataTree):
        # reshaping data for histogram
        tmp_pred = {}
        for var in pred_data:
            if var not in tmp_pred:
                tmp_pred[var] = np.array([])
            for scen in pred_data[var]:
                tmp_pred[var] = np.concatenate(
                    [tmp_pred[var], pred_data[var][scen][predictor].values.flatten()]
                )
        array_pred = np.array(list(tmp_pred.values()))

        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd that implies too many assumptions
        histo_kde = gaussian_kde(array_pred)

        # calculating density of points over the sample
        density = histo_kde.pdf(x=array_pred)

        # preparing the datatree
        weight, counter, factor_rescale = dict(), 0, 0
        # using former var, ensuring correct order on dimensions
        dims = pred_data[var][scen][predictor].dims
        for scen in pred_data[var]:
            # reshaping the weights for this scenario
            n_dims = {dim: pred_data[var][scen][dim].size for dim in dims}
            array_tmp = np.reshape(
                density[counter : counter + pred_data[var][scen][predictor].size],
                [n_dims[dim] for dim in dims],
            )
            tmp = xr.DataArray(
                data=array_tmp,
                dims=dims,
                coords={dim: pred_data[var][scen][dim] for dim in dims},
            )

            # inverse of density
            weight[scen] = xr.Dataset({"weight": 1 / tmp})
            factor_rescale += weight[scen]["weight"].sum()

            # preparing next scenario
            counter += pred_data[var][scen][predictor].size

        # preparing the output
        return xr.DataTree.from_dict(weight) / factor_rescale

    elif isinstance(targ_data, xr.Dataset):
        # reshaping data for histogram
        array_pred = pred_data.to_array().values

        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd that implies too many assumptions
        histo_kde = gaussian_kde(array_pred)

        # calculating density of points over the sample
        density = histo_kde.pdf(x=array_pred)

        # preparing the output
        return xr.Dataset({"weight": (1 / density) / np.sum(1 / density)})

    elif isinstance(targ_data, np.ndarray):
        # representation with kernel-density estimate using gaussian kernels
        # NB: more stable than np.histogramdd that implies too many assumptions
        histo_kde = gaussian_kde(pred_data)

        # calculating density of points over the sample
        density = histo_kde.pdf(x=pred_data)

        # preparing the output
        return (1 / density) / np.sum(1 / density)

    else:
        raise Exception(
            "The format for targ_data must be a xr.DataTree, xr.Dataset or a np.array."
        )

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights

    Source:
      https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
      @author Jack Peterson (jack@tinybike.net)
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
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