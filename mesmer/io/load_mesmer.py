"""
mesmer.io.load_mesmer
===================
Functions to load in mesmer output.


Functions:
    load_mesmer_output()

"""


import glob

import joblib


def load_mesmer_output(
    name,
    cfg,
    ens_type_str="_",
    method_str="_",
    preds_str="_",
    targs_str="_",
    esm_str="_",
    scen_str="",
):
    """Load in saved MESMER output (parameters or emulations).

    Args:
    - name (str): saved MESMER output to load (e.g., 'params_lt', 'emus_lv', 'emus_g')
    - cfg (module): config file containnig metadata
    - ens_type_str (str): ensemble type (e.g., 'msic')
    - method_str (str): method (e.g., 'OLS')
    - preds_str (st): predictos (e.g., 'gttas', 'gttas_gttas2')
    - targs_str (str): target variables (e.g., 'tas')
    - esm_str (str): Earth System Model (e.g., 'CanESM2', 'CanESM5')
    - scen_str (str): scenario (e.g., 'rcp85', 'ssp585', 'h-ssp585')

    Returns:
    - dict_out (dict): loaded MESMER output dictionary
        [xx] depending on the loaded output

    General remarks:
    - This function can only load a single .pkl file at every call
    - If multiple files exist for the given input strings, an empty dictionary is returned
    - If no file exists for the given input strings, an empty dictionary is returned
    - Also partial strings are accepted (with the exception of esm_str, where the full ESM name is needed): e.g.,
            scen_str='h-' for joint historical + ssp scenarios
            scen_str='hist' for separated historical + ssp scenarios
    - TODO: generalize writing of mid_path strings such that no longer require "/" (issues with other systems)

    """

    # choose directory depending on whether want to load params or emus
    if "params" in name:
        dir_mesmer = cfg.dir_mesmer_params
    elif "emu" in name:
        dir_mesmer = cfg.dir_mesmer_emus

    # choose middle part of pathway depending on what exactly want to load
    if "gt" in name:
        mid_path = "global/global_trend/"
    elif "gv" in name:
        mid_path = "global/global_variability/"
    elif "g" in name:
        mid_path = "global/"
    elif "lt" in name:
        mid_path = "local/local_trends/"
    elif "lv" in name:
        mid_path = "local/local_variability/"
    elif "l" in name:
        mid_path = "local/"

    # to ensure that don't end up with multiple files if multiple ESMs share same beginning of name
    if esm_str != "_":
        esm_str = esm_str + "_"

    path_list = glob.glob(
        dir_mesmer
        + mid_path
        + name
        + "*"
        + ens_type_str
        + "*"
        + method_str
        + "*"
        + preds_str
        + "*"
        + targs_str
        + "*"
        + esm_str
        + "*"
        + scen_str
        + "*.pkl"
    )

    # load the requested output dictionary
    if len(path_list) == 1:
        dict_out = joblib.load(path_list[0])
    elif len(path_list) == 0:
        print("No such file exists. An empty dictionary will be returned.")
        dict_out = {}
    elif len(path_list) > 1:
        print(
            "More than 1 file exists for these critera. Please be more concrete in your selection. An empty dictionary will be returned"
        )
        dict_out = {}

    return dict_out
