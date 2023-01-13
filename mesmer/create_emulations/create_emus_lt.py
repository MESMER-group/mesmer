# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to create local trends emulations with MESMER.
"""


import numpy as np

import mesmer.stats
from mesmer.create_emulations.utils import _gather_params, _gather_predictors
from mesmer.io.save_mesmer_bundle import save_mesmer_data


def create_emus_lt(params_lt, preds_lt, cfg, concat_h_f=False, save_emus=True):
    """
    Create local trends (i.e., forced response) emulations for given parameter set and
    predictors.

    Parameters
    ----------
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] (additional params depend on method employed)
        - ["full_model_contains_lv"] (whether the full model contains part of the local
          variability module, bool)

    preds_lt : dict
        nested dictionary of predictors for local trends with keys

        - [pred][scen] (1d/2d arrays (time)/(run, time) of predictor for specific
          scenario)

    cfg : module
        config file containing metadata

    concat_h_f : bool, optional
        determines if historical and future time period is concatenated into a single
        emulation or not, default = False (must be set to False if no historical data
        provided)

    save_emus : bool, optional
        determines if parameters are saved or not, default = True

    Returns
    -------
    emus_lt : dict
        local trend emulations nested dictionary with keys

        - [scen]["targ"] (2d array (time, gp) of local trend emulations)

    Notes
    -----
    - Assumptions:
        - same predictors for each target
        - if historical time period is included in predictors, it has its own dictionary
          key
        - if historical time period was included in training, it has its own scenario
        - either historical period is included for every scenario or for no scenario
    - Potential TODO:
        - evaluate if really need / want concat_h_f or if I want output to be determined
          by shape predictors

    """

    # derive necessary scenario names
    pred_names = list(preds_lt.keys())
    scenarios_emus = list(preds_lt[pred_names[0]].keys())

    if concat_h_f:
        if scenarios_emus[0] == "hist":
            scens_out_f = scenarios_emus[1:]
            scens_out = ["h-" + s for s in scens_out_f]
        else:
            raise ValueError("This combination is not supported.")
    else:
        if "h-" in scenarios_emus[0]:
            scens_out_f = list(map(lambda x: x.replace("h-", ""), scenarios_emus))
            scens_out = ["hist"] + scens_out_f
        else:
            scens_out = scens_out_f = scenarios_emus

    # check if correct predictors
    if pred_names != params_lt["preds"]:
        raise ValueError(
            "Wrong predictors were passed. The emulations cannot be created."
        )

    # select the method from a dict of fcts
    create_emus_method_func_mapping = {"OLS_each_gp_sep": create_emus_OLS_each_gp_sep}
    # extend dict if add more methods

    # dict could be extended to contain other methods (even actual fcts that I write)
    method_lt = params_lt["method"]
    if params_lt["method_each_gp_sep"]:
        method_lt = method_lt + "_each_gp_sep"
    else:
        meth = params_lt["method_each_gp_sep"]
        raise ValueError(f"No such method ({meth}) is currently implemented.")

    create_emus_method_lt = create_emus_method_func_mapping[method_lt]

    # print(f"{scens_out=}")
    # print(f"{scens_out_f=}")

    # create emulations
    emus_lt = {}
    if concat_h_f:
        lt_hist = create_emus_method_lt(params_lt, preds_lt, "hist")
        for scen_out, scen_out_f in zip(scens_out, scens_out_f):
            lt_scen_f = create_emus_method_lt(params_lt, preds_lt, scen_out_f)
            emus_lt[scen_out] = {}
            for targ in params_lt["targs"]:
                emus_lt[scen_out][targ] = np.concatenate(
                    [lt_hist[targ], lt_scen_f[targ]]
                )
    else:
        for scen_out in scens_out:
            emus_lt[scen_out] = create_emus_method_lt(params_lt, preds_lt, scen_out)

    # save the local trends emulation if requested
    if save_emus:
        save_mesmer_data(
            emus_lt,
            cfg.dir_mesmer_emus,
            "local",
            "local_trends",
            filename_parts=[
                "emus_lt",
                params_lt["method"],
                *params_lt["preds"],
                *params_lt["targs"],
                params_lt["esm"],
                *scens_out,
            ],
        )

    return emus_lt


def create_emus_OLS_each_gp_sep(params_lt, preds_lt, scen):
    """Create local trends with OLS with grid-point-specific predictors

    Parameters
    ----------
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] (additional params depend on method employed)
        - ["full_model_contains_lv"] (whether the full model contains part of the local
          variability module, bool)

    preds_lt : dict
        nested dictionary of predictors for local trends with keys

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific
          scenario)

    scen : str
        emulated scenario

    Returns
    -------
    emus_lt : dict
        local trend emulations dictionary with keys

        - ["targ"] (2d array (time, gp) of local trend emulations)

    Notes
    -----
    - Assumptions:
        - Coefficients are the same for every scenario

    """

    emus_lt = {}
    for targ in params_lt["targs"]:

        params = _gather_params(params_lt, targ, dims="cell")
        predictors = _gather_predictors(preds_lt, params_lt["preds"], scen, dims="time")

        lr = mesmer.stats.linear_regression.LinearRegression()
        lr.params = params

        prediction = lr.predict(predictors=predictors)

        emus_lt[targ] = prediction.values.T

    return emus_lt
