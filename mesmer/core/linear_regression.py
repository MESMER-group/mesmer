import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression(predictors, target, weights=None):
    """
    Perform a linear regression

    Parameters
    ----------
    predictors : array-like of shape (n_samples, n_predictors)
        Array of predictors

    target : array-like of shape (n_samples, n_targets)
        Array of targets where each row is a sample and each column is a
        different target i.e. variable to be predicted

    weights : array-like of shape (n_samples,)
        Weights for each sample

    Returns
    -------
    :obj:`np.ndarray` of shape (n_targets, n_predictors + 1)
        Array of intercepts and coefficients. Each row is the intercept and
        coefficients for a different target (rows are in same order as the
        columns of ``target``). In each row, the intercept of the regression is
        followed by the intercept for each predictor (in the same order as the
        columns of ``predictors``).
    """
    reg = LinearRegression()
    reg.fit(X=predictors, y=target, sample_weight=weights)

    intercepts = np.atleast_2d(reg.intercept_).T
    coefficients = np.atleast_2d(reg.coef_)

    return np.hstack([intercepts, coefficients])
