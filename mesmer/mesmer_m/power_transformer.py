import numpy as np
import xarray as xr

# from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.preprocessing import PowerTransformer, StandardScaler


def lambda_function(xi_0, xi_1, local_yearly_T):
    r"""Use logistic function to calculate lambda depending on the local yearly
    temperature. The function is defined as 
    
    .. math::

        \lambda = \frac{2}{\xi_0 + e^{\xi_1 \cdot T_y}}
    
    
    It ranges between 0 and 2.

    Parameters
    ----------
    xi_0 : float
        First coefficient of the logistic function (controlling the intercept).
    xi_1 : float
        Second coefficient of the logistic function (controlling the slope).
    local_yearly_T : ndarray of shape (n_years,)
            yearly temperature values of one gridcell and month used as predictor
            for lambda.

    Returns
    -------
    lambdas : ndarray of float of shape (n_years,)
        The parameters of the power transformation for each gridcell and month
    """
    return 2 / (1 + xi_0 * np.exp(local_yearly_T * xi_1))


class PowerTransformerVariableLambda(PowerTransformer):
    """Apply a power transform gridcellwise to make monthly residuals more
    Gaussian-like. The class inherits from Sklearn's Power transofrmer class.
    It is modified to allow for transformation parameters (lambda) which have
    a functional dependency on spatially resolved yearly mean temperature.
    Every month requires its own PowerTransform.

    Please refer to [1] for a description of the  Power transformer class.
    Please refer to [2] for an explanation of the modifications.

    Parameters
    ----------
    **kwargs :
        refer to the Power Transformer class to see a full list of possible options

    Attributes
    ----------
    coeffs_ : ndarray of shape (n_gridcell, n_coefficients)
        The coefficients to calculate lambda depending on the local yearly
        temperature. Defined via function lambda_function(coeff, local_yearly_T)
        as exponential dependency following [2]:
            def lambda_function(coeff, local_yearly_T):
                return(2/(1+coeff[0]*np.exp(local_yearly_T*coeff[1])))
            and n_coefficients = 2
    lambdas_ : ndarray of float of shape (n_gridcell, n_years)
        The parameters of the power transformation for each gridcell, calculated
        using lambda_function.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    .. [2] Nath, S., Lejeune, Q., Beusch, L., Schleussner, C. F., & Seneviratne, S. I. (2021).
           MESMER-M: an Earth System Model emulator for spatially resolved monthly temperatures.
           Earth System Dynamics Discussions, 1-38..
    Examples
    --------
    TBD
    """

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def fit(self, monthly_residuals, yearly_T, n_gridcells):
        """Estimate the optimal parameter lambda for each gridcell, given
        temperature residuals for one month of the year.
        The optimal lambda parameter for minimizing skewness is estimated on
        each gridcell independently using maximum likelihood.
        Parameters
        ----------
        monthly_residuals : ndarray of shape (n_years, n_gridcells)
            Monthly residuals after removing harmonic model fits, used to fit for
            the optimal transformation parameters (lambdas).
            Contains the value of one month for all years, e.g. all January values.
        yearly_T :  ndarray of shape (n_years, n_gridcells)
            yearly temperature values used as predictors for the lambdas.
        Returns
        -------
        self : object
        """
        # TODO: write what is in self
        # TODO: infer n_gridcells from data
        monthly_residuals = (
            monthly_residuals.copy()
        )  # force copy so that fit does not change X inplace

        self.coeffs_ = np.array(
            Parallel(n_jobs=-1, verbose=False)(
                delayed(self._yeo_johnson_optimize_lambda)(
                    monthly_residuals[:, gridcell], yearly_T[:, gridcell]
                )
                for gridcell in np.arange(n_gridcells)
            )
        )

        self.mins_ = np.amin(monthly_residuals, axis=0)
        self.maxs_ = np.amax(monthly_residuals, axis=0)
        # print(self.coeffs_)

        if self.standardize:
            self._scaler = StandardScaler(copy=True)
            self._scaler.fit(monthly_residuals)

        return self

    def _yeo_johnson_optimize_lambda(self, local_monthly_residuals, local_yearly_T):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed local monthly residual temperatures.
        Like for Box-Cox, MLE is done via the brent optimizer.
        Parameters
        ----------
        local_monthly_residuals : ndarray of shape (n_years,)
            Monthly residuals of one gridcell, used to fit for the optimal lambda.
        local_yearly_T :  ndarray of shape (n_years,)
            yearly temperature values of one gridcell used as predictor for lambda.
        Returns
        -------
        res: optimized coefficients for lambda function
        """

        def _neg_log_likelihood(coeffs):
            """Return the negative log likelihood of the observed local monthly
            residual temperatures as a function of lambda."""
            lambdas = lambda_function(coeffs[0], coeffs[1], local_yearly_T)
            # version with sklearn yeo johnson transform
            # x_trans = np.zeros_like(x)
            # for i, lmbda in enumerate(lambdas):
            #     x_trans[i] = self._yeo_johnson_transform(x[i], lmbda)

            # version with own power transform
            transformed_local_monthly_resids = self._yeo_johnson_transform(
                local_monthly_residuals, lambdas
            )

            n_samples = local_monthly_residuals.shape[0]
            loglikelihood = (
                -n_samples / 2 * np.log(transformed_local_monthly_resids.var())
            )
            loglikelihood += (
                (lambdas - 1)
                * np.sign(local_monthly_residuals)
                * np.log1p(np.abs(local_monthly_residuals))
            ).sum()

            return -loglikelihood

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        local_monthly_residuals = local_monthly_residuals[
            ~np.isnan(local_monthly_residuals)
        ]
        local_yearly_T = local_yearly_T[~np.isnan(local_yearly_T)]

        # first coefficient only positive for logistic function
        # second coefficient bounded to avoid very steep function
        bounds = np.array([[0, np.inf], [-0.1, 0.1]])
        # first guess is that data is already normal distributed
        first_guess = np.array([1, 0])

        return minimize(
            _neg_log_likelihood,
            first_guess,
            bounds=bounds,
            method="Nelder-Mead",
        ).x

    def _yeo_johnson_transform(self, local_monthly_residuals, lambdas):
        """Return transformed input local_monthly_residuals following Yeo-Johnson
        transform with parameter lambda. Input is for one month and gridcell but
        all years.
        """

        eps = np.finfo(np.float64).eps

        transformed = np.zeros_like(local_monthly_residuals)
        # get positions of four cases:
        # NOTE: this code is copied from sklearn's PowerTransformer, see
        # https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/preprocessing/_data.py#L3396
        # we acknowledge there is an inconsistency in the comparison of lambdas
        pos_a = (local_monthly_residuals >= 0) & (np.abs(lambdas) < eps)
        pos_b = (local_monthly_residuals >= 0) & (np.abs(lambdas) >= eps)
        pos_c = (local_monthly_residuals < 0) & (np.abs(lambdas - 2) > eps)
        pos_d = (local_monthly_residuals < 0) & (np.abs(lambdas - 2) <= eps)

        # assign values for the four cases
        transformed[pos_a] = np.log1p(local_monthly_residuals[pos_a])
        transformed[pos_b] = (
            np.power(local_monthly_residuals[pos_b] + 1, lambdas[pos_b]) - 1
        ) / lambdas[pos_b]
        transformed[pos_c] = -(
            np.power(-local_monthly_residuals[pos_c] + 1, 2 - lambdas[pos_c]) - 1
        ) / (2 - lambdas[pos_c])
        transformed[pos_d] = -np.log1p(-local_monthly_residuals[pos_d])

        return transformed

    def transform(self, monthly_residuals, yearly_T):
        """Apply the power transform to each feature using the fitted lambdas.
        Parameters
        ----------
        monthly_residuals : array-like, shape (n_years, n_gridcells)
            The monthly temperature data to be transformed using a power transformation
            with the fitted self.coeffs_. Contains the yearly values of one month
            for all gridcells, e.g. all January values.
        yearly_T: array-like, shape (n_years, n_gridcells)
            The yearly temperature values used as predictors for the lambdas using the
            fitted self.coeffs_.
        Returns
        -------
        transformed_monthly_resids : array-like, shape (n_years, n_gridcells)
            The transformed monthly residuals.
        """
        lambdas = self._get_yeo_johnson_lambdas(yearly_T)

        transformed_monthly_resids = np.zeros_like(monthly_residuals)

        for gridcell, lmbda in enumerate(lambdas.T):
            transformed_monthly_resids[:, gridcell] = self._yeo_johnson_transform(
                monthly_residuals[:, gridcell], lmbda
            )

        if self.standardize:
            transformed_monthly_resids = self._scaler.transform(
                transformed_monthly_resids
            )

        return transformed_monthly_resids

    def _get_yeo_johnson_lambdas(self, yearly_T):

        lambdas = np.zeros_like(yearly_T)
        gridcell = 0
        # TODO: sure yearly_T.T gives local yearly T?
        for coeffs, local_yearly_T in zip(self.coeffs_, yearly_T.T):
            lambdas[:, gridcell] = lambda_function(coeffs[0], coeffs[1], local_yearly_T)
            gridcell += 1

        lambdas = np.where(lambdas < 0, 0, lambdas)
        lambdas = np.where(lambdas > 2, 2, lambdas)

        return lambdas

    def inverse_transform(self, transformed_monthly_T, yearly_T):
        """Apply the inverse power transformation using the fitted lambdas.
        The inverse of the Yeo-Johnson transformation is given by::
            if X >= 0 and lambda_ == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda_ != 0:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
            elif X < 0 and lambda_ != 2:
                X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
            elif X < 0 and lambda_ == 2:
                X = 1 - exp(-X_trans)
        Parameters
        ----------
        transformed_monthly_T : array-like, shape (n_years, n_gridcells)
            The transformed data.
        yearly_T: array-like, shape (n_years, n_gridcells)
            The yearly temperature values used as predictors for the lambdas using
            the fitted self.coeffs_.
        Returns
        -------
        inverted_monthly_T : array-like, shape (n_years, n_gridcells)
            The inverted, i.e. original, monthly temperature values.
        """
        # TODO: if we save the lambdas, we would not need to give yearly temperaure here
        if self.standardize:
            transformed_monthly_T = self._scaler.inverse_transform(
                transformed_monthly_T
            )

        inverted_monthly_T = np.zeros_like(transformed_monthly_T)

        lambdas = self._get_yeo_johnson_lambdas(yearly_T)

        for gridcell, lmbda in enumerate(lambdas.T):
            for year, y_lmbda in enumerate(lmbda):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    inverted_monthly_T[year, gridcell] = (
                        self._yeo_johnson_inverse_transform(
                            transformed_monthly_T[year, gridcell], y_lmbda
                        )
                    )

            # clip values to not exceed original range
            # apparently a relict from when lambda was not constrained to [0,2]
            inverted_monthly_T[:, gridcell] = np.where(
                inverted_monthly_T[:, gridcell] < self.mins_[gridcell],
                self.mins_[gridcell],
                inverted_monthly_T[:, gridcell],
            )
            inverted_monthly_T[:, gridcell] = np.where(
                inverted_monthly_T[:, gridcell] > self.maxs_[gridcell],
                self.maxs_[gridcell],
                inverted_monthly_T[:, gridcell],
            )

        return inverted_monthly_T


def _yeo_johnson_transform_np(data, lambdas):
    """transform data using Yeo-Johnson transformation with variable lambda

    Input is for one month and gridcell but all years. This function is adjusted
    from sklearn to accomodate variable lambdas for each residual.
    """

    eps = np.finfo(np.float64).eps

    transformed = np.zeros_like(data)
    # get positions of four cases:
    # NOTE: this code is copied from sklearn's PowerTransformer, see
    # https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/preprocessing/_data.py#L3396
    # we acknowledge there is an inconsistency in the comparison of lambdas
    sel_a = (data >= 0) & (np.abs(lambdas) < eps)
    sel_b = (data >= 0) & (np.abs(lambdas) >= eps)
    sel_c = (data < 0) & (np.abs(lambdas - 2) > eps)
    sel_d = (data < 0) & (np.abs(lambdas - 2) <= eps)

    # assign values for the four cases
    transformed[sel_a] = np.log1p(data[sel_a])
    transformed[sel_b] = (np.power(data[sel_b] + 1, lambdas[sel_b]) - 1) / lambdas[
        sel_b
    ]
    transformed[sel_c] = -(np.power(-data[sel_c] + 1, 2 - lambdas[sel_c]) - 1) / (
        2 - lambdas[sel_c]
    )
    transformed[sel_d] = -np.log1p(-data[sel_d])

    return transformed


def _yeo_johnson_inverse_transform_np(data, lambdas):
    """invert residuals using Yeo-Johnson transformation with variable lambda

    This function is adjusted from sklearn to accomodate variable lambdas for each
    residual.

    Notes
    -----
    if X >= 0 and lambda_ == 0:
        X = exp(X_trans) - 1
    elif X >= 0 and lambda_ != 0:
        X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
    elif X < 0 and lambda_ != 2:
        X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
    elif X < 0 and lambda_ == 2:
        X = 1 - exp(-X_trans)
    """

    eps = np.finfo(np.float64).eps

    transformed = np.zeros_like(data)
    # get positions of four cases:
    pos_a = (data >= 0) & (np.abs(lambdas) < eps)
    pos_b = (data >= 0) & (np.abs(lambdas) >= eps)
    pos_c = (data < 0) & (np.abs(lambdas - 2) > eps)
    pos_d = (data < 0) & (np.abs(lambdas - 2) <= eps)

    # assign values for the four cases
    transformed[pos_a] = np.exp(data[pos_a]) - 1
    transformed[pos_b] = (
        np.power(data[pos_b] * lambdas[pos_b] + 1, 1 / lambdas[pos_b]) - 1
    )
    transformed[pos_c] = 1 - np.power(
        -(2 - lambdas[pos_c]) * data[pos_c] + 1, 1 / (2 - lambdas[pos_c])
    )
    transformed[pos_d] = 1 - np.exp(-data[pos_d])

    return transformed


def _yeo_johnson_optimize_lambda_np(monthly_residuals, yearly_pred):

    # the computation of lambda is influenced by NaNs so we need to
    # get rid of them
    isnan = np.isnan(monthly_residuals) | np.isnan(yearly_pred)
    monthly_residuals = monthly_residuals[~isnan]
    yearly_pred = yearly_pred[~isnan]

    def _neg_log_likelihood(coeffs):
        """Return the negative log likelihood of the observed local monthly residual
        temperatures as a function of lambda.
        """
        lambdas = lambda_function(coeffs[0], coeffs[1], yearly_pred)

        # version with own power transform
        transformed_resids = _yeo_johnson_transform_np(monthly_residuals, lambdas)

        n_samples = monthly_residuals.shape[0]
        loglikelihood = -n_samples / 2 * np.log(transformed_resids.var())
        loglikelihood += (
            (lambdas - 1)
            * np.sign(monthly_residuals)
            * np.log1p(np.abs(monthly_residuals))
        ).sum()

        return -loglikelihood

    bounds = np.array([[0, np.inf], [-0.1, 0.1]])
    first_guess = np.array([1, 0])

    xi_0, xi_1 = minimize(
        _neg_log_likelihood,
        x0=first_guess,
        bounds=bounds,
        method="Nelder-Mead",
    ).x

    return xi_0, xi_1


def get_lambdas_from_covariates_xr(coeffs, yearly_pred):
    """function that relates fitted coefficients and the yearly predictor
    to the lambdas. We usee a logistic function between 0 and 2 to estimate
    the lambdas.

    Parameters
    ----------
    coeffs : xr.Dataset containing xi_0 and xi_1 of shape (months, n_gridcells)
        The parameters of the power transformation for each gridcell and month, calculated
        using fit_yeo_johnson_transform.
    yearly_pred : xr.DataArray of shape (n_years, n_gridcells)
        yearly values used as predictors for the lambdas.

    Returns
    -------
    lambdas : xr.DataArray of shape (months, n_gridcells, n_years)
        The parameters of the power transformation for each gridcell month and year

    """
    if not isinstance(coeffs, xr.Dataset):
        raise TypeError(f"Expected a `xr.Dataset`, got {type(coeffs)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    lambdas = xr.apply_ufunc(
        lambda_function,
        coeffs.xi_0,
        coeffs.xi_1,
        yearly_pred,
        input_core_dims=[[], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return lambdas.rename("lambdas")


def fit_yeo_johnson_transform(monthly_residuals, yearly_pred, time_dim="time"):
    """estimate the optimal coefficients for the parameters lambda for each gridcell,
    to normalize monthly residuals conditional on yearly predictor.
    The optimal coefficients for the lambda parameters for minimizing skewness are
    estimated on each gridcell independently using maximum likelihood.

    Parameters
    ----------
    monthly_residuals : xr.DataArray of shape (n_years*12, n_gridcells)
        Monthly residuals after removing harmonic model fits, used to fit for the optimal
        transformation parameters (lambdas).
    yearly_pred : xr.DataArray of shape (n_years, n_gridcells)
        yearly values used as predictors for the lambdas.
    time_dim : str, optional
        Name of the time dimension in the input data used to align monthly residuals and
        yearly predictor data (needs to be the same in both).

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the estimated coefficients xi_0 and xi_1 needed to estimate
        lambda with dimensions (months, n_gridcells) and the lambdas themselves with
        dimensions (months, n_gridcells, n_years).

    """
    if not isinstance(monthly_residuals, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_residuals)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    monthly_resids_grouped = monthly_residuals.groupby(time_dim + ".month")

    coeffs = []
    for month in range(1, 13):

        # align time dimension
        monthly_data = monthly_resids_grouped[month]
        monthly_data[time_dim] = yearly_pred[time_dim]

        xi_0, xi_1 = xr.apply_ufunc(
            _yeo_johnson_optimize_lambda_np,
            monthly_data,
            yearly_pred,
            input_core_dims=[[time_dim], [time_dim]],
            output_core_dims=[[], []],
            output_dtypes=[float, float],
            vectorize=True,
        )

        coeffs.append(xr.Dataset({"xi_0": xi_0, "xi_1": xi_1}))

    return xr.concat(coeffs, dim="month")


def yeo_johnson_transform(monthly_residuals, coeffs, yearly_pred):
    """
    transform monthly_residuals following Yeo-Johnson transformer
    with parameters lambda, fit with fit_power_transformer_xr.

    Parameters
    ----------
    monthly_residuals : xr.DataArray of shape (n_years*12, n_gridcells)
        Monthly residuals after removing harmonic model fits, used to fit for the
        optimal transformation parameters (lambdas).
    coeffs : xr.Dataset containing xi_0 and xi_1 of shape (months, n_gridcells)
        The parameters of the power transformation for each gridcell, calculated using
        lambda_function.
    yearly_pred : xr.DataArray of shape (n_years, n_gridcells)
        yearly temperature values used as predictors for the lambdas.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the transformed monthly residuals and the parameters of the
        power transformation for each gridcell, calculated using lambda_function.
    """
    # NOTE: this is equivalent to using pt.transform with
    # pt = PowerTransformerVariableLambda(standardize = False)

    if not isinstance(monthly_residuals, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_residuals)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    if not isinstance(coeffs, xr.Dataset):
        raise TypeError(f"Expected a `xr.Dataset`, got {type(monthly_residuals)}")

    lambdas = get_lambdas_from_covariates_xr(coeffs, yearly_pred).rename(
        {"time": "year"}
    )
    lambdas_stacked = lambdas.stack(stack=["year", "month"])

    transformed_resids = xr.apply_ufunc(
        _yeo_johnson_transform_np,
        monthly_residuals,
        lambdas_stacked,
        input_core_dims=[["time"], ["stack"]],
        output_core_dims=[["time"]],
        output_dtypes=[float],
        vectorize=True,
    ).rename("transformed")

    return xr.merge([transformed_resids, lambdas])


def inverse_yeo_johnson_transform(monthly_residuals, coeffs, yearly_pred):
    """apply the inverse power transformation using the fitted lambdas.

    Parameters
    ----------
    monthly_residuals : xr.DataArray of shape (n_years, n_gridcells)
        The transformed data.
    coeffs : xr.Dataset containing xi_0 and xi_1 of shape (months, n_gridcells)
        The parameters of the power transformation for each gridcell, calculated using
        lambda_function.
    yearly_pred : xr.DataArray of shape (n_years, n_gridcells)
        yearly temperature values used as predictors for the lambdas.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the inverted monthly residuals and the parameters of the
        power transformation for each gridcell, calculated using lambda_function.

    Notes
    -----
    The inverse of the Yeo-Johnson transformation is given by::
        if X >= 0 and lambda_ == 0:
            X = exp(X_trans) - 1
        elif X >= 0 and lambda_ != 0:
            X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
        elif X < 0 and lambda_ != 2:
            X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
        elif X < 0 and lambda_ == 2:
            X = 1 - exp(-X_trans)
    """
    if not isinstance(monthly_residuals, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_residuals)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    if not isinstance(coeffs, xr.Dataset):
        raise TypeError(f"Expected a `xr.Dataset`, got {type(monthly_residuals)}")

    lambdas = get_lambdas_from_covariates_xr(coeffs, yearly_pred).rename(
        {"time": "year"}
    )
    lambdas_stacked = lambdas.stack(stack=["year", "month"])

    inverted_resids = xr.apply_ufunc(
        _yeo_johnson_inverse_transform_np,
        monthly_residuals,
        lambdas_stacked,
        input_core_dims=[["time"], ["stack"]],
        output_core_dims=[["time"]],
        output_dtypes=[float],
        vectorize=True,
    ).rename("inverted")

    return xr.merge([inverted_resids, lambdas])
