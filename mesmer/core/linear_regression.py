import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression(predictors, target, weights=None):
    """
    Perform a linear regression

    Parameters
    ----------
    predictors : array-like of shape (n_samples, n_predictors)
        Array of predictors

    target : array-like of shape (n_samples,)
        Array of targets

    weights : array-like of shape (n_samples,)
        Weights for each sample

    Returns
    -------
    :obj:`np.ndarray`
        Intercept of regression followed by the intercept for each predictor (
        in the same order as the columns of ``predictors``)
    """
    reg = LinearRegression()
    reg.fit(X=predictors, y=target, sample_weight=weights)

    return np.hstack([reg.intercept_, reg.coef_])
