# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to create global trend emulations with MESMER.
"""
import warnings

from mesmer.create_emulations.utils import concatenate_hist_future
from mesmer.io.save_mesmer_bundle import save_mesmer_data


def create_emus_gt(params_gt, preds_gt, cfg, concat_h_f=False, save_emus=True):
    """see docstring of `gather_gt_data`"""

    warnings.warn(
        "'create_emus_gt' has been renamed to `gather_gt_data`", FutureWarning
    )

    return gather_gt_data(
        params_gt, preds_gt, cfg, concat_h_f=concat_h_f, save_emus=save_emus
    )


def gather_gt_data(params_gt, preds_gt, cfg, concat_h_f=False, save_emus=True):
    """
    Create global trend (emissions + volcanoes) emulations for specified ensemble type
    and method.

    Parameters
    ----------
    params_gt : dict
        Parameters dictionary

        - ["targ"] (emulated variable, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - ["time"] (1d array of years, np.ndarray)
        - [xx] (additional keys depend on employed method and are listed in
          train_gt_T_method() function)

    preds_gt : dict
        nested dictionary of predictors for global trend with keys

        - [pred][scen]  (1d/2d arrays (time)/(run, time) of predictor for specific
          scenario)

    cfg : module
        config file containing metadata

    concat_h_f : bool, optional
        determines if historical and future time period is concatenated into a single
        emulation or not, default = False (must be set to false if no historical data
        provided)

    save_emus : bool, optional
        determines if emulation is saved or not, default = True

    Returns
    -------
    emus_gt : dict
        global trend emulations dictionary with keys

        - [scen] (1d array of global trend emulation time series)

    Notes
    -----
    - Assumptions:
        - if no preds_gt needed, pass time as predictor instead to know which scenarios
          to emulate

    """
    # derive necessary scenario names
    pred_names = list(preds_gt.keys())
    scenarios_emus = list(preds_gt[pred_names[0]].keys())

    if "h-" in scenarios_emus[0]:
        scenarios_emus = ["hist"] + [scen.replace("h-", "") for scen in scenarios_emus]

    emus_gt = {}

    # gather data
    if "LOWESS" in params_gt["method"]:
        for scen in scenarios_emus:
            emus_gt[scen] = params_gt[scen]
    else:
        raise ValueError("The chosen method is currently not implemented.")

    if concat_h_f:
        emus_gt = concatenate_hist_future(emus_gt)
        scenarios_emus = list(emus_gt.keys())

    # save the global trend emulation if requested
    if save_emus:
        save_mesmer_data(
            emus_gt,
            cfg.dir_mesmer_emus,
            "global",
            "global_trend",
            filename_parts=[
                "emus_gt",
                params_gt["method"],
                *params_gt["preds"],
                params_gt["targ"],
                params_gt["esm"],
                *scenarios_emus,
            ],
        )

    return emus_gt
