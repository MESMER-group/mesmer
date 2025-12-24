import abc

import numpy as np
import sklearn.linear_model
import statsmodels.tsa.ar_model
import xarray as xr


class MesmerCalibrateBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for calibration
    """


class MesmerCalibrateTargetPredictor(MesmerCalibrateBase):
    @abc.abstractmethod
    def calibrate(self, target, predictor, **kwargs):
        """
        [TODO: update this based on however LinearRegression.calibrate's docs
        end up looking]
        """


class LinearRegression(MesmerCalibrateTargetPredictor):
    """
    following

    https://github.com/MESMER-group/mesmer/blob/d73e8f521a2e1d081a48b775ba14dd764cb671e8/mesmer/calibrate_mesmer/train_lt.py#L165

    All the lines above and below that line are basically just data
    preparation, which makes it very hard to see what the model actually is
    """

    @staticmethod
    def _regress_single_group(target_point, predictor, weights=None):
        # this is the method that actually does the regression
        args = [predictor.T, target_point.reshape(-1, 1)]
        if weights is not None:
            args.append(weights)
        reg = sklearn.linear_model.LinearRegression().fit(*args)
        out_array = np.concatenate([reg.intercept_, *reg.coef_])

        return out_array

    def calibrate(
        self,
        target_flattened,
        predictors_flattened,
        stack_coord_name,
        predictor_name="predictor",
        weights=None,
        predictor_temporary_name="__pred_store__",
    ):
        """
        TODO: redo docstring
        """
        if predictor_name not in predictors_flattened.dims:
            raise AssertionError(f"{predictor_name} not in {predictors_flattened.dims}")

        if predictor_temporary_name in predictors_flattened.dims:
            raise AssertionError(
                f"{predictor_temporary_name} already in {predictors_flattened.dims}, choose a different temporary name"
            )

        res = xr.apply_ufunc(
            self._regress_single_group,
            target_flattened,
            predictors_flattened,
            input_core_dims=[[stack_coord_name], [predictor_name, stack_coord_name]],
            output_core_dims=((predictor_temporary_name,),),
            vectorize=True,
            kwargs=dict(weights=weights),
        )

        # assuming that predictor's names are in the 'variable' coordinate
        predictors_plus_intercept_order = ["intercept"] + list(
            predictors_flattened["variable"].values
        )
        res = res.assign_coords(
            {predictor_temporary_name: predictors_plus_intercept_order}
        ).rename({predictor_temporary_name: predictor_name})

        return res


class MesmerCalibrateTarget(MesmerCalibrateBase):
    @abc.abstractmethod
    def calibrate(self, target, **kwargs):
        """
        [TODO: update this based on however LinearRegression.calibrate's docs
        end up looking]
        """

    @staticmethod
    def _check_target_is_one_dimensional(target, return_numpy_values):
        if len(target.dims) > 1:
            raise AssertionError(f"More than one dimension, found {target.dims}")

        if not return_numpy_values:
            return None

        return target.dropna(dim=target.dims[0]).values


class AutoRegression1DOrderSelection(MesmerCalibrateTarget):
    def calibrate(
        self,
        target,
        maxlag=12,
        ic="bic",
    ):
        target_numpy = self._check_target_is_one_dimensional(
            target, return_numpy_values=True
        )

        calibrated = statsmodels.tsa.ar_model.ar_select_order(
            target_numpy, maxlag=maxlag, ic=ic, old_names=False
        )

        return calibrated.ar_lags


class AutoRegression1D(MesmerCalibrateTarget):
    def calibrate(
        self,
        target,
        order,
    ):
        target_numpy = self._check_target_is_one_dimensional(
            target, return_numpy_values=True
        )

        calibrated = statsmodels.tsa.ar_model.AutoReg(
            target_numpy, lags=order, old_names=False
        ).fit()

        return {
            "intercept": calibrated.params[0],
            "lag_coefficients": calibrated.params[1:],
            # I don't know what this is so a better name could probably be chosen
            "standard_innovations": np.sqrt(calibrated.sigma2),
        }
