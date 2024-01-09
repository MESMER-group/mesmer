import numpy as np
import xarray as xr


def concatenate_hist_future(data):
    """concatenate historical and future data

    Parameters
    ----------
    data : dict
        Possibly nested dictionary containing arrays to concatenate. The keys of data
        must correspond to the scenarios to use. The values can either be numpy arrays
        or dicts of numpy arrays.

    Returns
    -------
    concatenated : dict
        Possibly nested dictionary with concatenated arrays.
    """

    scens_in = list(data.keys())

    if "hist" not in scens_in:
        raise ValueError("data does not contain 'hist' scenario")

    scens_in.remove("hist")
    scens_out = [f"h-{scen}" for scen in scens_in]

    concatenated = {}

    hist = data.get("hist")

    # data is not a nested dict
    if not isinstance(hist, dict):

        for scen_out, scen_in in zip(scens_out, scens_in):
            concatenated[scen_out] = np.concatenate([hist, data[scen_in]])

        return concatenated

    # data is a nested dict
    for scen_out, scen_in in zip(scens_out, scens_in):

        concatenated[scen_out] = {}

        for targ in data[scen_in].keys():
            concatenated[scen_out][targ] = np.concatenate(
                [hist[targ], data[scen_in][targ]]
            )

    return concatenated


def _gather_lr_preds(preds_dict, predictor_names, scen, dims):
    """gather predictors for linear regression from legacy data structures

    Parameters
    ----------
    preds_dict : dict
        Dictionary containing all predictors.
    predictor_names : list of str
        List of all predictors to gather from ``preds_dict``.
    scen : str
        Scenario for which to read the predictors.
    dims : str, tuple of str
        Name of string for DataArray

    Returns
    -------
    predictors : dict
        Dictionary of gathered predictors.

    Notes
    -----
    This function should become obsolete once switching to the newer data structures.

    """
    predictors = {}
    for pred in predictor_names:
        predictors[pred] = xr.DataArray(preds_dict[pred][scen], dims=dims)

    return predictors


def _gather_lr_params(params_dict, targ, dims):
    """gather parameters for linear regression from legacy data structures

    Parameters
    ----------
    params_dict : dict
        Dictionary containing all parameters.
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
