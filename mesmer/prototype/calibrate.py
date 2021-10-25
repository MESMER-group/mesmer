import abc

import numpy as np
import sklearn.linear_model
import xarray as xr


class MesmerCalibrateBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for calibration
    """

    @staticmethod
    def _get_predictor_dims(predictors):
        predictors_dims = {k: v.dims for k, v in predictors.items()}
        predictors_dims_unique = set(predictors_dims.values())
        if len(predictors_dims_unique) > 1:
            raise AssertionError(
                f"Dimensions of predictors are not all the same, we have: {predictors_dims}"
            )

        return list(predictors_dims_unique)[0]

    @staticmethod
    def _get_stack_coord_name(inp_array):
        stack_coord_name = "stacked_coord"
        if stack_coord_name in inp_array.dims:
            stack_coord_name = "memser_stacked_coord"

        if stack_coord_name in inp_array.dims:
            raise NotImplementedError("You have dimensions we can't safely unstack yet")

        return stack_coord_name

    @staticmethod
    def _check_coords_match(obj, obj_other, check_coord):
        coords_match = obj.coords[check_coord].equals(obj_other.coords[check_coord])
        if not coords_match:
            raise AssertionError(
                f"{check_coord} is not the same on {obj} and {obj_other}"
            )

    @staticmethod
    def _flatten_predictors(predictors, dims_to_flatten, stack_coord_name):
        predictors_flat = []
        for v, vals in predictors.items():
            if stack_coord_name in vals.dims:
                raise AssertionError(f"{stack_coord_name} is already in {vals.dims}")

            vals_flat = vals.stack({stack_coord_name: dims_to_flatten}).dropna(
                stack_coord_name
            )
            vals_flat.name = v
            predictors_flat.append(vals_flat)

        out = xr.merge(predictors_flat).to_stacked_array(
            "predictor", sample_dims=[stack_coord_name]
        )

        return out

    def _flatten_predictors_and_target(self, predictors, target):
        dims_to_flatten = self._get_predictor_dims(predictors)
        stack_coord_name = self._get_stack_coord_name(target)

        target_flattened = target.stack({stack_coord_name: dims_to_flatten}).dropna(
            stack_coord_name
        )
        predictors_flattened = self._flatten_predictors(
            predictors, dims_to_flatten, stack_coord_name
        )
        self._check_coords_match(
            target_flattened, predictors_flattened, stack_coord_name
        )

        return predictors_flattened, target_flattened, stack_coord_name

    @staticmethod
    def _get_calibration_groups(target_flattened, stack_coord_name):
        calibration_groups = list(set(target_flattened.dims) - {stack_coord_name})
        if len(calibration_groups) > 1:
            raise NotImplementedError()

        return calibration_groups[0]


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
        args = [predictor, target_point]
        if weights is not None:
            args.append(weights)

        reg = sklearn.linear_model.LinearRegression().fit(*args)

        return reg

    def _calibrate_groups(
        self,
        target_flattened,
        predictor_numpy,
        predictor_names,
        weights,
        calibration_groups,
    ):
        def _calibrate_group(target_group):
            target_group_numpy = target_group.values
            res_group = self._regress_single_group(
                target_group_numpy, predictor_numpy, weights
            )

            res_xr = xr.DataArray(
                np.concatenate([[res_group.intercept_], res_group.coef_]),
                dims=["predictor"],
                coords={"predictor": predictor_names},
            )

            return res_xr

        res = target_flattened.groupby(calibration_groups).apply(_calibrate_group)

        return res

    @staticmethod
    def _get_predictors_numpy_and_output_predictor_order(
        predictors_flattened, stack_coord_name
    ):
        # stacked coord has to go first for the regression to be setup correctly
        predictors_flattened_dim_order = [stack_coord_name] + [
            d for d in predictors_flattened.dims if d != stack_coord_name
        ]
        predictors_flattened_reordered = predictors_flattened.transpose(
            *predictors_flattened_dim_order
        )
        predictors_numpy = predictors_flattened_reordered.values

        predictors_plus_intercept_order = ["intercept"] + list(
            predictors_flattened_reordered["variable"].values
        )

        return predictors_numpy, predictors_plus_intercept_order

    def calibrate(self, target, predictors, weights=None):
        """
        Calibrate a linear regression model

        Parameters
        ----------
        target : xarray.DataArray
            Target data

        predictors : dict[str: xarray.DataArray]
            Predictors to use, each key gives the name of the predictor, the
            values give the values of the predictor.

        weights : optional, xarray.DataArray
            Weights to use for dimensions in ``predictors``

        Returns
        -------
        xarray.DataArray
            Fitting coefficients, the dimensions of which are the combination
            of "predictor" and all dimensions in ``target`` which are not in
            ``predictors``. For example, the resulting dimensions could be
            ["gridpoint", "predictor"].

        Notes
        -----
        We flatten ``target``, ``predictors`` and ``weights`` across the
        dimensions shared by ``target`` and ``predictors`` to create the input
        arrays for the linear regression.
        """
        (
            predictors_flattened,
            target_flattened,
            stack_coord_name,
        ) = self._flatten_predictors_and_target(predictors, target)

        (
            predictors_numpy,
            predictors_plus_intercept_order,
        ) = self._get_predictors_numpy_and_output_predictor_order(
            predictors_flattened, stack_coord_name
        )

        calibration_groups = self._get_calibration_groups(
            target_flattened, stack_coord_name
        )

        return self._calibrate_groups(
            target_flattened,
            predictors_numpy,
            predictor_names=predictors_plus_intercept_order,
            weights=weights,
            calibration_groups=calibration_groups,
        )
