import abc

import numpy as np
import sklearn.linear_model
import xarray as xr


class MesmerCalibrateBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for calibration
    """


class MesmerCalibrateTargetPredictor(MesmerCalibrateBase):
    @abc.abstractmethod
    def calibrate(self, target, predictor, **kwargs):
        """
        Calibrate a model

        [TODO: update this based on whatever LinearRegression.calibrate's docs
        end up looking like]

        Parameters
        ----------
        target : xarray.DataArray
            Target data

        predictor : xarray.DataArray
            Predictor data array

        **kwargs
            Passed onto the calibration method

        Returns
        -------
        xarray.DataArray
            Fitting coefficients [TODO description of how labelled]
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
