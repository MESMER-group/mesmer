# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr

from mesmer.core.datatree import collapse_datatree_into_dataset
from mesmer.mesmer_x._expression import Expression


class probability_integral_transform:
    def __init__(
        self,
        expr_start,
        expr_end,
        coeffs_start=None,
        coeffs_end=None,
    ):
        """
        Probability integral transform of the data given parameters of a given distribution
        into their equivalent in a standard normal distribution.

        Parameters
        ----------
        expr_start : str
            string describing the starting expression
        expr_end : str
            string describing the starting expression
        coeffs_start : xarray dataset
            Coefficients of the starting expression. Default: None.
        coeffs_end : xarray dataset
            Coefficients of the ending expression. Default: None.
        """
        # preparation of distributions
        self.expression_start = Expression(expr_start, "start")
        self.expression_end = Expression(expr_end, "end")

        # preparation of coefficients
        self.coeffs_start = self.prepare_coefficients(coeffs_start)
        self.coeffs_end = self.prepare_coefficients(coeffs_end)

    def prepare_coefficients(self, coeffs):
        """
        Prepare coefficients for the expression
        """
        if coeffs is None:
            # creating an empty dataset for use
            return xr.Dataset()

        elif isinstance(coeffs, xr.Dataset):
            # no need to correct the format
            return coeffs

        else:
            raise Exception("coefficients must be a xarray Dataset or None")

    def transform(
        self,
        data,
        target_name,
        preds_start=None,
        preds_end=None,
        threshold_proba=1.0e-9,
    ):
        """
        Probability integral transform of the data given parameters of a given distribution
        into their equivalent in a standard normal distribution.

        Parameters
        ----------
        data : stacked Datatree
            Data to transform. xarray Dataset with coordinate 'gridpoint'.
        target_name : str
            name of the variable to train
        preds_start : stacked Datatree | None
            Covariants of the starting expression. Default: None.
        preds_end : stacked Datatree | None
            Covariants of the ending expression. Default: None.
        threshold_proba : float, default: 1.e-9.
            Threshold for the probability of the sample on the starting distribution.
            Applied both to the lower and upper bounds of the distribution.
        Returns
        -------
        transf_inputs : not sure yet what data format will be used at the end.
            Assumed to be a xarray Dataset with coordinates 'time' and 'gridpoint' and one
            2D variable with both coordinates
        """
        if isinstance(data, xr.DataTree):
            # check on similar format
            if isinstance(preds_start, xr.Dataset) or isinstance(preds_end, xr.Dataset):
                raise Exception("predictors must have the same format as data")

            # looping over scenarios of data
            out = dict()
            for scen, data_scen in data.items():
                # preparing data to transform
                data_scen = data_scen.to_dataset()

                # preparing predictors
                ds_pred_start = self.prepare_predictors(preds_start, scen=scen)
                ds_preds_end = self.prepare_predictors(preds_end, scen=scen)

                # transforming data
                tmp = self._transform(
                    data_scen, target_name, ds_pred_start, ds_preds_end, threshold_proba
                )

                # creating transf_data as a dataset with tmp as a variable
                out[scen] = xr.Dataset(
                    {target_name: (data_scen[target_name].dims, tmp)},
                    coords=data_scen[target_name].coords,
                )

            # creating datatree
            return xr.DataTree.from_dict(out)

        elif isinstance(data, xr.Dataset):
            # check on similar format
            if isinstance(preds_start, xr.DataTree) or isinstance(
                preds_end, xr.DataTree
            ):
                raise Exception("predictors must have the same format as data")

            # preparing predictors
            ds_pred_start = self.prepare_predictors(preds_start)
            ds_preds_end = self.prepare_predictors(preds_end)

            # transforming data
            tmp = self._transform(
                data, target_name, ds_pred_start, ds_preds_end, threshold_proba
            )

            # creating transf_data as a dataset with tmp as a variable
            transf_data = xr.Dataset(
                {target_name: (data[target_name].dims, tmp)},
                coords=data[target_name].coords,
            )
            return transf_data

        else:
            raise Exception("data must be a xarray Dataset or Datatree")

    def prepare_predictors(self, preds, scen=None):
        # preparation of predictors
        if preds is None:
            ds_preds = xr.Dataset()

        elif isinstance(preds, xr.DataTree):
            if scen is not None:
                # taking only the correct scenario, while keeping the same format
                preds = xr.DataTree.from_dict({p: preds[p][scen] for p in preds})

            # correcting format: must be dict(str, DataArray or array) for Expression
            tmp = collapse_datatree_into_dataset(preds, dim="predictor")
            var_name = [var for var in tmp.variables][0]
            ds_preds = xr.Dataset()
            for pp in tmp["predictor"].values:
                ds_preds[pp] = tmp[var_name].sel(predictor=pp)

        elif isinstance(preds, xr.Dataset):
            # no need to correct format
            ds_preds = preds

        return ds_preds

    def _transform(
        self, data, target_name, ds_pred_start, ds_preds_end, threshold_proba
    ):
        # parameters of starting distribution
        params_start = self.expression_start.evaluate_params(
            self.coeffs_start, ds_pred_start, forced_shape=data[target_name].dims
        )

        # probabilities of the sample on the starting distribution
        cdf_data = self.expression_start.distrib.cdf(data[target_name], **params_start)

        # avoiding very unlikely values
        cdf_data[np.where(cdf_data < threshold_proba)] = threshold_proba
        cdf_data[np.where(cdf_data > 1 - threshold_proba)] = 1 - threshold_proba

        # parameters of ending distribution
        params_end = self.expression_end.evaluate_params(
            self.coeffs_end, ds_preds_end, forced_shape=data[target_name].dims
        )

        # values corresponding to probabilities of sample on the ending distribution
        return self.expression_end.distrib.ppf(cdf_data, **params_end)


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
