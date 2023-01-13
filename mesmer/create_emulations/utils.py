import xarray as xr


def _gather_predictors(preds_dict, predictor_names, scen, dims):
    """gather predictors for linear regression from legacy data structures

    Parameters
    ----------
    preds_dict : dict
        Dictonary containg all predictors.
    predictor_names : list of str
        List of all predictors to gather from ``preds_dict``.
    scen : str
        Scenario for which to read the predictors.
    dims : str, tuple of str
        Name of string for DataArray

    Returns
    -------
    predictors : dict
        Dictonary of gathered predictors.

    Notes
    -----
    This function should become obsolete once switching to the newer data structures.

    """
    predictors = {}
    for pred in predictor_names:
        predictors[pred] = xr.DataArray(preds_dict[pred][scen], dims=dims)

    return predictors


def _gather_params(params_dict, targ, dims):
    """gather parameters for linear regression from legacy data structures

    Parameters
    ----------
    params_dict : dict
        Dictonary containg all parameters.
    targ : str
        Name of target variable for which to read the parameters.
    dims : str, tuple of str
        Name of string for DataArray

    Returns
    -------
    params : xr.Dataset
        Dataset of gathered parameters.

    Notes
    -----
    This function should become obsolete once switching to the newer data structures.


    """

    params = {}
    for pred in params_dict["preds"]:

        params[pred] = xr.DataArray(params_dict[f"coef_{pred}"][targ], dims=dims)

    if "intercept" in params_dict:
        intercept = xr.DataArray(params_dict["intercept"][targ], dims=dims)
        fit_intercept = True
    else:
        intercept = 0
        fit_intercept = False

    params["intercept"] = intercept
    params["fit_intercept"] = fit_intercept

    return xr.Dataset(data_vars=params)
