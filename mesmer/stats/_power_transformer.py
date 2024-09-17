import numpy as np
import xarray as xr
from scipy.optimize import minimize


def lambda_function(coeffs, local_yearly_T):
    r"""Use logistic function to calculate lambda depending on the local yearly
    values. The function is defined as

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
            yearly values of one gridcell and month used as predictor
            for lambda.

    Returns
    -------
    lambdas : ndarray of float of shape (n_years,)
        The parameters of the power transformation for each gridcell and month
    """
    return 2 / (1 + coeffs[0] * np.exp(local_yearly_T * coeffs[1]))


def _yeo_johnson_transform_np(data, lambdas):
    """transform data using Yeo-Johnson transformation with variable lambda.

    Input is for one month and gridcell but all years. This function is adjusted
    from sklearn to accommodate variable lambdas for each value.

    Notes
    -----
    The Yeo-Johnson transformation is given by:

        if :math:`X \\leq 0` and :math:`\\lambda = 0`:
            :math:`X_{trans} = log(X + 1)`
        elif :math:`X \\leq 0` and :math:`\\lambda \\neq 0`:
            :math:`X_{trans} = \\frac{(X + 1)^{\\lambda} - 1}{\\lambda}`
        elif :math:`X < 0` and :math:`\\lambda \\neq 2`:
            :math:`X_{trans} = - \\frac{(-X + 1)^{2 - \\lambda} - 1}{2 - \\lambda}`
        elif :math:`X < 0` and :math:`\\lambda = 2`:
            :math:`X_{trans} = - log(-X + 1)`

    Note that :math:`X` and :math:`X_{trans}`have the same sign.
    Also see `sklearn's PowerTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html>`_
    """

    lambdas = lambdas.copy()

    return _yeo_johnson_transform_optimized(data)(lambdas)


def _yeo_johnson_transform_optimized(data):
    """performance-optimize yeo-johnson transformation - for the inner loop of minimize"""

    # pre-compute constant values

    eps = np.finfo(np.float64).eps

    transf = np.empty_like(data)

    data_log1p = np.log1p(np.abs(data))

    # pos = data >= 0
    # neg = ~pos

    neg = data < 0

    def _inner(lambdas):

        # NOTE: this code is copied from sklearn's PowerTransformer, see
        # https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/preprocessing/_data.py#L3396
        # we acknowledge there is an inconsistency in the comparison of lambdas

        # nonlocal transf

        # lambdas = lambdas.copy()
        # l = lambdas[neg]
        lambdas[neg] = 2 - lambdas[neg]

        lambda_eq_0 = np.abs(lambdas) <= eps
        lambda_ne_0 = ~lambda_eq_0
        # lambda_eq_2 = np.abs(lambdas - 2) <= eps

        # sel_a = pos & lambda_eq_0
        # sel_b = pos & ~lambda_eq_0
        # sel_c = neg & ~lambda_eq_2
        # sel_d = neg & lambda_eq_2

        transf[lambda_eq_0] = data_log1p[lambda_eq_0]

        lmbds = lambdas[lambda_ne_0]
        transf[lambda_ne_0] = np.expm1(data_log1p[lambda_ne_0] * lmbds) / lmbds

        lambdas[neg] = 2 - lambdas[neg]
        # lambdas[neg] = l

        # transf[sel_a] = data_log1p[sel_a]

        # transf[sel_a] = data_log1p[sel_a]

        # lmbds = lambdas[sel_b]
        # transf[sel_b] = np.expm1(data_log1p[sel_b] * lmbds) / lmbds

        # lmbds = 2 - lambdas[sel_c]
        # transf[sel_c] = -np.expm1(data_log1p[sel_c] * lmbds) / lmbds

        # transf[sel_d] = -data_log1p[sel_d]

        np.copysign(transf, data, out=transf)

        return transf

    return _inner


def _yeo_johnson_inverse_transform_np(data, lambdas):
    """inverse Yeo-Johnson transformation with variable lambda.

    This function is adjusted from sklearn to accommodate variable lambdas for each
    value.

    Notes
    -----
    The inverse of the Yeo-Johnson transformation is given by:

        if :math:`X_{trans} \\leq 0` and :math:`\\lambda = 0`:
            :math:`X_{inv} = exp(X_{trans}) - 1`
        elif :math:`X_{trans} \\leq 0` and :math:`\\lambda \\neq 0`:
            :math:`X_{inv} = (X_{trans} \\cdot \\lambda + 1)^{\\frac{1}{\\lambda} - 1}`
        elif :math:`X_{trans} < 0` and :math:`\\lambda \\neq 2`:
            :math:`X_{inv} = 1 - ((\\lambda - 2) \\cdot X_{trans} + 1)^{\\frac{1}{2 - \\lambda}}`
        elif :math:`X_{trans} < 0` and :math:`\\lambda = 2`:
            :math:`X_{inv} = 1 - exp(-X_{trans})`

    Note that :math:`X_{inv}` and :math:`X_{trans}` have the same sign.
    """

    eps = np.finfo(np.float64).eps

    transf = np.empty_like(data)
    # get positions of four cases:
    sel_a = (data >= 0) & (np.abs(lambdas) < eps)
    sel_b = (data >= 0) & (np.abs(lambdas) >= eps)
    sel_c = (data < 0) & (np.abs(lambdas - 2) > eps)
    sel_d = (data < 0) & (np.abs(lambdas - 2) <= eps)

    # assign values for the four cases
    transf[sel_a] = np.expm1(data[sel_a])

    lmbds = lambdas[sel_b]
    transf[sel_b] = np.expm1(np.log1p(data[sel_b] * lmbds) / lmbds)

    lmbds = 2 - lambdas[sel_c]
    transf[sel_c] = -np.expm1(np.log1p(-lmbds * data[sel_c]) / lmbds)
    transf[sel_d] = -np.expm1(-data[sel_d])

    return transf


def _yeo_johnson_optimize_lambda_np(monthly_residuals, yearly_pred):

    # the computation of lambda is influenced by NaNs so we need to
    # get rid of them
    isnan = np.isnan(monthly_residuals) | np.isnan(yearly_pred)
    monthly_residuals = monthly_residuals[~isnan]
    yearly_pred = yearly_pred[~isnan]

    # pos = monthly_residuals >= 0
    # neg = ~pos
    # n_pos = pos.sum()

    # def _sort_by_sign(data):
    #     out = np.empty_like(data)
    #     out[:n_pos] = data[pos]
    #     out[n_pos:] = data[neg]
    #     return out

    # resid_ordered = _sort_by_sign(monthly_residuals)
    # pred_ordered = _sort_by_sign(yearly_pred)

    # initialize constant arrays
    # _yeo_johnson_transform = _yeo_johnson_transform_optimized(resid_ordered)
    _yeo_johnson_transform = _yeo_johnson_transform_optimized(monthly_residuals)

    data_log1p = np.sign(monthly_residuals) * np.log1p(np.abs(monthly_residuals))
    # data_log1p = np.sign(resid_ordered) * np.log1p(np.abs(resid_ordered))

    data_log1p_sum = data_log1p.sum()

    def _neg_log_likelihood(coeffs):
        """Return the negative log likelihood of the observed local monthly residuals
        as a function of lambda.
        """
        lambdas = lambda_function(coeffs, yearly_pred)
        # lambdas = lambda_function(coeffs, pred_ordered)

        # version with own power transform
        transformed_resids = _yeo_johnson_transform(lambdas)

        n_samples = monthly_residuals.shape[0]
        loglikelihood = -n_samples / 2 * np.log(transformed_resids.var())
        # equal ((lambdas - 1) * data_log1p).sum()
        # loglikelihood += ((lambdas - 1) * data_log1p).sum()

        # loglikelihood += (lambdas * data_log1p).sum() - data_log1p_sum

        # l = lambdas @ data_log1p - data_log1p_sum
        # print(l.shape)

        loglikelihood += lambdas @ data_log1p - data_log1p_sum

        return -loglikelihood

    bounds = np.array([[0, np.inf], [-0.1, 0.1]])
    first_guess = np.array([1, 0])

    res = minimize(
        _neg_log_likelihood,
        x0=first_guess,
        bounds=bounds,
        # args=(data_log1p, data_log1p_sum),
        method="Nelder-Mead",
    )

    # print()
    # print(res)

    return res.x


def get_lambdas_from_covariates(coeffs, yearly_pred):
    """function that relates fitted coefficients and the yearly predictor
    to the lambdas. We usee a logistic function between 0 and 2 to estimate
    the lambdas, see :func:`lambda_function <mesmer.stats.lambda_function>`.

    Parameters
    ----------
    lambda_coeffs : ``xr.Dataset``
        The parameters of the power transformation for each gridcell and month,
        containing ``lambda_coeffs`` with dims (months, n_gridcells, coeff) calculated
        using ``fit_yeo_johnson_transform``.
    yearly_pred : ``xr.DataArray``
        yearly values used as predictors for the lambdas. Must have shape shape
        (n_years, n_gridcells).

    Returns
    -------
    lambdas : ``xr.DataArray`` of shape (months, n_gridcells, n_years)
        The parameters of the power transformation for each gridcell month and year

    """
    if not isinstance(coeffs, xr.Dataset):
        raise TypeError(f"Expected a `xr.Dataset`, got {type(coeffs)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    lambdas = xr.apply_ufunc(
        lambda_function,
        coeffs.lambda_coeffs,
        yearly_pred,
        input_core_dims=[("coeff",), []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return lambdas.rename("lambdas")


def fit_yeo_johnson_transform(monthly_residuals, yearly_pred, time_dim="time"):
    """
    estimate the optimal coefficients for the parameters :math:`\\lambda` for each gridcell,
    to normalize monthly residuals conditional on yearly predictor. Here, :math:`\\lambda`
    depends on the yearly predictor according to :func:`lambda_function <mesmer.stats.lambda_function>`.
    The optimal coefficients for the lambda parameters for minimizing skewness are
    estimated on each gridcell independently using maximum likelihood.

    Parameters
    ----------
    monthly_residuals : ``xr.DataArray`` of shape (n_years*12, n_gridcells)
        Monthly residuals after removing harmonic model fits, used to fit for the optimal
        transformation parameters (lambdas).
    yearly_pred : ``xr.DataArray`` of shape (n_years, n_gridcells)
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
    # TODO allow passing func instead of our fixed lambda_function?
    if not isinstance(monthly_residuals, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_residuals)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    coeffs = []
    for month in range(12):

        monthly_data = monthly_residuals.isel({time_dim: slice(month, None, 12)})

        res = xr.apply_ufunc(
            _yeo_johnson_optimize_lambda_np,
            monthly_data,
            yearly_pred,
            input_core_dims=[[time_dim], [time_dim]],
            exclude_dims={time_dim},
            output_core_dims=[["coeff"]],
            output_dtypes=[float],
            vectorize=True,
        )

        coeffs.append(xr.Dataset({"lambda_coeffs": res}))

    return xr.concat(coeffs, dim="month")


def yeo_johnson_transform(monthly_residuals, coeffs, yearly_pred):
    """
    transform `monthly_residuals` following Yeo-Johnson transformer
    with parameters :math:`\\lambda`, fit with :func:`fit_yeo_johnson_transform <mesmer.stats.fit_yeo_johnson_transform>`.

    Parameters
    ----------
    monthly_residuals : ``xr.DataArray`` of shape (n_years*12, n_gridcells)
        Monthly residuals after removing harmonic model fits, used to fit for the
        optimal transformation parameters (lambdas).
    coeffs : ``xr.Dataset``
        The parameters of the power transformation containing ``lambda_coeffs`` of shape
        (months, n_gridcells, coeff) for each gridcell, calculated using
        :func:`lambda_function <mesmer.stats.lambda_function>`.
    yearly_pred : ``xr.DataArray`` of shape (n_years, n_gridcells)
        yearly values used as predictors for the lambdas.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the transformed monthly residuals and the parameters of the
        power transformation for each gridcell.

    Notes
    -----
    The Yeo-Johnson transformation is given by:

        if :math:`X \\leq 0` and :math:`\\lambda = 0`:
            :math:`X_{trans} = log(X + 1)`
        elif :math:`X \\leq 0` and :math:`\\lambda \\neq 0`:
            :math:`X_{trans} = \\frac{(X + 1)^{\\lambda} - 1}{\\lambda}`
        elif :math:`X < 0` and :math:`\\lambda \\neq 2`:
            :math:`X_{trans} = - \\frac{(-X + 1)^{2 - \\lambda} - 1}{2 - \\lambda}`
        elif :math:`X < 0` and :math:`\\lambda = 2`:
            :math:`X_{trans} = - log(-X + 1)`

    Note that :math:`X` and :math:`X_{trans}` have the same sign.
    Also see `sklearn's PowerTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html>`_.
    """
    if not isinstance(monthly_residuals, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_residuals)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    if not isinstance(coeffs, xr.Dataset):
        raise TypeError(f"Expected a `xr.Dataset`, got {type(monthly_residuals)}")

    lambdas = get_lambdas_from_covariates(coeffs, yearly_pred).rename({"time": "year"})
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
    monthly_residuals : ``xr.DataArray`` of shape (n_years, n_gridcells)
        The data to be transformed back to the original scale.
    coeffs : ``xr.Dataset``
        The parameters of the power transformation containing ``lambda_coeffs`` of shape
        (months, n_gridcells, coeff) for each gridcell, calculated using
        :func:`lambda_function <mesmer.stats.lambda_function>`.
    yearly_pred : ``xr.DataArray`` of shape (n_years, n_gridcells)
        yearly values used as predictors for the lambdas.

    Returns
    -------
    :obj:`xr.Dataset`
        Dataset containing the inverted monthly residuals and the parameters of the
        power transformation for each gridcell.

    Notes
    -----
    The inverse of the Yeo-Johnson transformation is given by:

        if :math:`X_{trans} \\leq 0` and :math:`\\lambda = 0`:
            :math:`X_{inv} = exp(X_{trans}) - 1`
        elif :math:`X_{trans} \\leq 0` and :math:`\\lambda \\neq 0`:
            :math:`X_{inv} = (X_{trans} \\cdot \\lambda + 1)^{\\frac{1}{\\lambda} - 1}`
        elif :math:`X_{trans} < 0` and :math:`\\lambda \\neq 2`:
            :math:`X_{inv} = 1 - ((\\lambda - 2) \\cdot X_{trans} + 1)^{\\frac{1}{2 - \\lambda}}`
        elif :math:`X_{trans} < 0` and :math:`\\lambda = 2`:
            :math:`X_{inv} = 1 - exp(-X_{trans})`

    Note that :math:`X_{inv}` and :math:`X_{trans}` have the same sign.
    """
    if not isinstance(monthly_residuals, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(monthly_residuals)}")

    if not isinstance(yearly_pred, xr.DataArray):
        raise TypeError(f"Expected a `xr.DataArray`, got {type(yearly_pred)}")

    if not isinstance(coeffs, xr.Dataset):
        raise TypeError(f"Expected a `xr.Dataset`, got {type(monthly_residuals)}")

    lambdas = get_lambdas_from_covariates(coeffs, yearly_pred).rename({"time": "year"})
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
