import abc

import numpy as np
import sklearn.linear_model
import xarray as xr


class MesmerCalibrateBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for calibration
    """
    def _flatten_predictors(self, predictors, flat_dim_order):
        predictors_flat = []
        predictor_order = []
        for v, vals in predictors.items():
            predictor_order.append(v)
            predictors_flat.append(self._flatten(vals, flat_dim_order))

        predictors_flat = np.vstack(predictors_flat).T

        return predictors_flat, predictor_order

    @staticmethod
    def _flatten(values, flat_dim_order, dropna=True):
        out = values.transpose(*flat_dim_order).values.flatten()
        # TODO: better handling of nan
        # Probably a better solution is to pass all the flattening up a level
        # i.e. should be done before calibration is attempted/have a function
        # to handle flattening scenarios
        if dropna:
            out = out[np.where(~np.isnan(out))]

        return out

    @staticmethod
    def _get_calibration_groups(target, groups):
        return target.groupby(groups)

    @staticmethod
    def _unflatten_calibrated_values(res, dim_names, coords):
        res_flat = np.vstack(list(res.values()))
        coords = {
            **coords,
            "predictor": list(res.keys())
        }

        out = xr.DataArray(res_flat, dims=["predictor"] + dim_names, coords=coords)

        return out



class MesmerCalibrateTargetOnly(MesmerCalibrateBase):
    @abc.abstractmethod
    def calibrate(self, predictor, **kwargs):
        """
        Calibrate a model

        Parameters
        ----------
        predictor : xarray.DataArray
            Predictor data array

        **kwargs
            Passed onto the calibration method

        Returns
        -------
        xarray.DataArray
            Fitting coefficients (have to think about how best to label and
            store)
        """

class MesmerCalibrateTargetPredictor(MesmerCalibrateBase):
    @abc.abstractmethod
    def calibrate(self, predictor, target, **kwargs):
        """
        Calibrate a model

        Parameters
        ----------
        predictor : xarray.DataArray
            Predictor data array

        target : xarray.DataArray
            Target data

        **kwargs
            Passed onto the calibration method

        Returns
        -------
        xarray.DataArray
            Fitting coefficients (have to think about how best to label and
            store)
        """


class LinearRegression(MesmerCalibrateTargetPredictor):
    """
    following

    https://github.com/MESMER-group/mesmer/blob/d73e8f521a2e1d081a48b775ba14dd764cb671e8/mesmer/calibrate_mesmer/train_lt.py#L165

    All the lines above and below that line are basically just data
    preparation, which makes it very hard to see what the model actually is
    """
    @staticmethod
    def _regress_single_point(target_point, predictor, weights=None):
        # this is the method that actually does the regression
        args = [predictor, target_point]
        if weights is not None:
            args.append(weights)

        reg = sklearn.linear_model.LinearRegression().fit(*args)

        return reg

    def calibrate(self, target, predictors, weights=None, groups="gridpoint"):
        # this method handles the pre-processing to get the data ready for the
        # regression, does the regression (or calls another method to do so),
        # then puts it all back into a DataArray for output

        # would need to be smarter here too to handle multiple predictors and/
        # or target variables but that should be doable
        res = {
            **{k: [] for k in predictors},
            "intercept": [],
        }

        flat_dim_order = [d for d in target.dims if d != groups]

        predictor_numpy, predictor_order = self._flatten_predictors(predictors, flat_dim_order)
        for group_val, target_group in self._get_calibration_groups(target, groups):
            target_group_flat = self._flatten(target_group, flat_dim_order)

            res_group = self._regress_single_point(target_group_flat, predictor_numpy, weights)

            # need to be smarter about this, but idea is to store things using
            # xarray DataArray
            res["intercept"].append(res_group.intercept_)
            for i, p in enumerate(predictor_order):
                res[p].append(res_group.coef_[i])

        res = self._unflatten_calibrated_values(res, dim_names=[groups], coords={groups: getattr(target, groups)})

        return res


class AutoRegression(MesmerCalibrateTargetOnly):
    def _regress_single(target, lags):
        res = AutoReg(target, lags=lags).fit()

        return res

    def calibrate(self, target, groups=["gridpoint"], lags=1):
        # would need to be smarter here too to handle multiple predictors and/
        # or target variables but that should be doable
        res = {
            "intercepts": [],
            "coef": [],
            "stds": [],
        }
        for target_group in self._get_calibration_groups(target, groups):
            res_group = target_group.apply(self._regress_single, lags)
            # need to be smarter about this, but idea is to store things using
            # xarray DataArray
            res["intercepts"].append(res_group.params[0])
            res["coef"].append(res_group.params[1])
            res["stds"].append(np.sqrt(res_group.sigma2))

        res = xr.DataArray(res)

        return res
