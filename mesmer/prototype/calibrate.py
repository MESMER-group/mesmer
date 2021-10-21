import abc


class MesmerCalibrateBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for calibration
    """

    # then we could build up utils, whether they were static methods of the
    # base class or their own utils module doesn't really matter...
    # an example is below, but we could have others to handle grouping, adding
    # dimension information back to numpy arrays etc.
    @staticmethod
    def _flatten(target, predictor):
        """
        Flatten xarray dataarrays into basic numpy arrays

        Parameters
        ----------
        target : xarray.DataArray

        predictor : xarray.DataArray

        Returns
        -------
        (np.ndarray, np.ndarray, dict)
            Flattened target, flattened predictor and info required to convert
            back into xarray if desired
        """

    @staticmethod
    def _get_calibration_groups(target, groups):
        return target.groupby(groups)


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

        reg = LinearRegression().fit(*args)

        return reg

    def calibrate(self, target, predictor, weights=None, groups=["gridpoint"]):
        # this method handles the pre-processing to get the data ready for the
        # regression, does the regression (or calls another method to do so),
        # then puts it all back into a DataArray for output

        # would need to be smarter here too to handle multiple predictors and/
        # or target variables but that should be doable
        res = {
            "intercepts": [],
            "coef": [],
        }
        for target_group in self._get_calibration_groups(target, groups):
            res_group = target_group.apply(self._regress_single_point, predictor, weights)
            # need to be smarter about this, but idea is to store things using
            # xarray DataArray
            res["intercepts"].append(res_group.intercept_)
            res["coef"].append(res_group.coef_)

        res = xr.DataArray(res)
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
