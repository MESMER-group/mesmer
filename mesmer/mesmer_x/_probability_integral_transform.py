# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr

from mesmer.core.datatree import collapse_datatree_into_dataset
from mesmer.mesmer_x._conditional_distribution import ConditionalDistribution


class ProbabilityIntegralTransform:
    def __init__(
        self,
        distrib_orig: ConditionalDistribution,
        distrib_targ: ConditionalDistribution,
    ):
        """
        Probability integral transform of the data from its original distribution into
        its equivalent in the target distribution.

        Parameters
        ----------
        distrib_orig: ConditionalDistribution
            original distribution
        distrib_targ: ConditionalDistribution
            target distribution, the distribution to which the data
            should be transformed
        """
        # preparation of distributions
        self.expression_orig = distrib_orig.expression
        self.expression_targ = distrib_targ.expression

        # preparation of coefficients
        self.coefficients_orig = distrib_orig.coefficients

        try:
            self.coefficients_targ = distrib_targ.coefficients
        except ValueError:
            # if the target distribution does not have coefficients, we set them empty
            self.coefficients_targ = xr.Dataset()

    def transform(
        self,
        data,
        target_name,
        preds_orig=None,
        preds_targ=None,
        threshold_proba=1.0e-9,
    ):
        """
        Probability integral transform data given coefficients for the
        expression of a conditional distribution.

        Parameters
        ----------
        data : Datatree
            Data to transform.
        target_name : str
            name of the variable to transform
        preds_orig : Datatree | None
            Covariants for the original distribution. If None, ?.
        preds_targ : Datatree | None
            Covariants of the target distribution. If None, ?.
        threshold_proba : float, default: 1.e-9.
            Threshold for the probability of the sample on the original distribution.
            The probabilities of samples outside this threshold (on both sides of the distribtion)
            will be set to the threshold. This should avoid very unlikely values
        Returns1
        -------
        transf_inputs : DataTree
            Transformed data.
        """
        if isinstance(data, xr.DataTree):
            # check on similar format
            if isinstance(preds_orig, xr.Dataset) or isinstance(preds_targ, xr.Dataset):
                raise Exception("predictors must have the same format as data")

            # looping over scenarios of data
            out = dict()
            for scen, data_scen in data.items():
                # preparing data to transform
                data_scen = data_scen.to_dataset()

                # preparing predictors
                ds_preds_orig = self.prepare_predictors(preds_orig, scen=scen)
                ds_preds_targ = self.prepare_predictors(preds_targ, scen=scen)

                # transforming data
                tmp = self._transform(
                    data_scen,
                    target_name,
                    ds_preds_orig,
                    ds_preds_targ,
                    threshold_proba,
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
            if isinstance(preds_orig, xr.DataTree) or isinstance(
                preds_targ, xr.DataTree
            ):
                raise Exception("predictors must have the same format as data")

            # preparing predictors
            ds_preds_orig = self.prepare_predictors(preds_orig)
            ds_preds_targ = self.prepare_predictors(preds_targ)

            # transforming data
            tmp = self._transform(
                data, target_name, ds_preds_orig, ds_preds_targ, threshold_proba
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
        self, data, target_name, ds_pred_orig, ds_preds_targ, threshold_proba
    ):
        # parameters of starting distribution
        params_orig = self.expression_orig.evaluate_params(
            self.coefficients_orig, ds_pred_orig, forced_shape=data[target_name].dims
        )

        # probabilities of the sample on the starting distribution
        cdf_data = self.expression_orig.distrib.cdf(data[target_name], **params_orig)

        # avoiding very unlikely values
        cdf_data[np.where(cdf_data < threshold_proba)] = threshold_proba
        cdf_data[np.where(cdf_data > 1 - threshold_proba)] = 1 - threshold_proba

        # parameters of ending distribution
        params_targ = self.expression_targ.evaluate_params(
            self.coefficients_targ, ds_preds_targ, forced_shape=data[target_name].dims
        )

        # values corresponding to probabilities of sample on the ending distribution
        return self.expression_targ.distrib.ppf(cdf_data, **params_targ)
