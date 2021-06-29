# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to load in mesmer output.
"""


import glob
import os.path
import warnings

import joblib


def load_mesmer_output(
    name,
    cfg,
    method_str="_",
    preds_str="_",
    targs_str="_",
    esm_str="_",
    scen_str="",
    mid_path=None,
):
    """Load saved MESMER output (parameters or emulations).

    Parameters
    ----------
    name : str
        saved MESMER output to load (e.g., "params_lt", "emus_lv", "emus_g")
    cfg : module
        config file containing metadata
    method_str : str, optional
        method (e.g., "OLS")
    preds_str : st, optional
        predictos (e.g., "gttas", "gttas_gttas2")
    targs_str : str, optional
        target variables (e.g., "tas")
    esm_str : str, optional
        Earth System Model (e.g., "CanESM2", "CanESM5")
    scen_str : str, otional
        scenario (e.g., "rcp85", "ssp585", "h-ssp585")
    mid_path : str, optional
        middle part of pathway depending on what exactly want to load (e.g.,
        "local/local_trends")

    Returns
    -------
    dict_out : dict
        loaded MESMER output dictionary

        - [xx] (depending on the loaded output)

    Notes
    -----
    - This function can only load a single .pkl file at every call
    - If multiple files exist for the given input strings, an empty dictionary is
      returned
    - If no file exists for the given input strings, an empty dictionary is returned
    - Also partial strings are accepted (with the exception of esm_str, where the full
      ESM name is needed): e.g.:

        - scen_str="h-" for joint historical + ssp scenarios
        - scen_str="hist" for separated historical + ssp scenarios
    - If no mid_path is provided the default MESMER structure for saved params and emus
      is assumed

    """

    # choose directory depending on whether want to load params or emus
    if "params" in name:
        dir_mesmer = cfg.dir_mesmer_params
    elif "emu" in name:
        dir_mesmer = cfg.dir_mesmer_emus

    # identify middle part of pathway depending on what exactly want to load if not passed as keyword argument
    if mid_path is None:
        if "gt" in name:
            mid_path = os.path.join("global", "global_trend")
        elif "gv" in name:
            mid_path = os.path.join("global", "global_variability")
        elif "g" in name:
            mid_path = "global"
        elif "lt" in name:
            mid_path = os.path.join("local", "local_trends")
        elif "lv" in name:
            mid_path = os.path.join("local", "local_variability")
        elif "l" in name:
            mid_path = "local"

    # to ensure that don't end up with multiple files if multiple ESMs share same beginning of name
    if esm_str != "_":
        esm_str = esm_str + "_"

    path_list = glob.glob(
        os.path.join(
            dir_mesmer,
            mid_path,
            f"{name}*{method_str}*{preds_str}*{targs_str}*{esm_str}*{scen_str}*.pkl",
        )
    )

    # load the requested output dictionary
    if len(path_list) == 1:
        dict_out = joblib.load(path_list[0])
    elif len(path_list) == 0:
        warnings.warn("No such file exists. An empty dictionary will be returned.")
        dict_out = {}
    elif len(path_list) > 1:
        warnings.warn(
            "More than 1 file exists for these critera. "
            "Please be more concrete in your selection."
            " An empty dictionary will be returned.",
        )
        dict_out = {}

    return dict_out
