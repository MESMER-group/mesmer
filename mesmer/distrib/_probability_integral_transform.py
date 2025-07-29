# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr

from mesmer.core.datatree import _datatree_wrapper
from mesmer.distrib._conditional_distribution import ConditionalDistribution


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
        try:
            self.coefficients_orig = distrib_orig.coefficients
        except ValueError:
            # if the target distribution does not have coefficients, we set them empty
            self.coefficients_orig = xr.Dataset()

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
        *,
        threshold_proba=1.0e-9,
    ):
        """
        Probability integral transform data given coefficients for the
        expression of a conditional distribution.

        Parameters
        ----------
        data : xr.DataTree | xr.Dataset
            Data to transform.
        target_name : str
            name of the variable to transform
        preds_orig : Datatree | xr.Dataset | None
            Covariants for the original distribution. Pass None, if `distrib_orig` does
            not require any.
        preds_targ : Datatree | xr.Dataset | None
            Covariants of the target distribution. Pass None, if `distrib_targ` does
            not require any.
        threshold_proba : float, default: 1.e-9.
            Threshold for the probability of the sample on the original distribution.
            The probabilities of samples outside this threshold (on both sides of the distribtion)
            will be set to the threshold. This should avoid very unlikely values

        Returns
        -------
        transf_inputs : DataTree | xr.Dataset
            Transformed data.
        """

        # transforming data
        return self._transform(
            data,
            target_name,
            preds_orig,
            preds_targ,
            threshold_proba,
        )

    @_datatree_wrapper
    def _transform(
        self, data, target_name, ds_pred_orig, ds_preds_targ, threshold_proba
    ):
        if ds_pred_orig is None:
            ds_pred_orig = xr.Dataset()
        if ds_preds_targ is None:
            ds_preds_targ = xr.Dataset()

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
        trans = self.expression_targ.distrib.ppf(cdf_data, **params_targ)
        return xr.Dataset(
            {target_name: (data[target_name].dims, trans)},
            coords=data[target_name].coords,
        )
