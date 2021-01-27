"""
mesmer.create_emulations.merge_emus
===================
Functions to merge emulations of different MESMER modules.


Functions:
    create_emus_g_T()

"""


import os

import joblib


def create_emus_g_T(
    emus_gt_T, emus_gv_T, params_gt_T, params_gv_T, cfg, save_emus=True
):
    """ Merge global trend and global variability temperature emulations of the same scenarios.
    
    Args:
    - emus_gt_T (dict): global trend emulations dictionary with keys
        [scen] (1d array of global trend emulation time series)
    - emus_gv_T (dict): global variability emulations dictionary with keys
        [scen] (2d array  (emus x time) of global variability emulation time series)
    - params_gt_T (dict): dictionary containing the calibrated parameters for the global trend emulations, keys relevant here
        ['targ'] (emulated variable, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
    - params_gv_T (dict): dictionary containing the calibrated parameters for the global variability emulations, keys relevant here
        ['targ'] (variable which is emulated, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (type of ensemble which is emulated, str)
        ['method'] (applied method, str)
    - cfg (module): config file containnig metadata
    - save_emus (bool,optional): determines if emulation is saved or not, default = True

    Returns:
    - emus_g_T (dict): global emulations dictionary with keys
        [scen] (2d array  (emus x time) of global emulation time series)
        
    """

    # specify necessary variables from config file
    dir_mesmer_emus = cfg.dir_mesmer_emus
    scen_name_emus = cfg.scen_name_emus

    scenarios_gt_T = list(emus_gt_T.keys())
    scenarios_gv_T = list(emus_gv_T.keys())

    if scenarios_gt_T == scenarios_gv_T:
        emus_g_T = {}
        for scen in scenarios_gt_T:
            emus_g_T[scen] = emus_gt_T[scen] + emus_gv_T[scen]

    else:
        print(
            "The global trend and the global variabilty emulations are not from the same scenario, no global emulation is created"
        )
        emus_g_T = []

    if params_gt_T["targ"] == params_gv_T["targ"]:
        targ = params_gt_T["targ"]
    else:
        print("The target variables do not match. No global emulation is created")
        emus_g_T = []

    if params_gt_T["esm"] == params_gv_T["esm"]:
        esm = params_gt_T["esm"]
    else:
        print("The Earth System Models do not match. No global emulation is created")
        emus_g_T = []

    if params_gt_T["ens_type"] == params_gv_T["ens_type"]:
        ens_type = params_gt_T["ens_type"]
    else:
        print("The ensemble types do not match. No global emulation is created")
        emus_g_T = []

    ## save the global emus if requested
    if save_emus == True:
        dir_mesmer_emus_g = dir_mesmer_emus + "global/"
        # check if folder to save emus in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_g):
            os.makedirs(dir_mesmer_emus_g)
            print("created dir:", dir_mesmer_emus_g)
        joblib.dump(
            emus_g_T,
            dir_mesmer_emus_g
            + "emus_g_"
            + ens_type
            + "_gt_"
            + params_gt_T["method"]
            + "_"
            + "_".join(params_gt_T["preds"])
            + "_gv_"
            + params_gv_T["method"]
            + "_"
            + "_".join(params_gv_T["preds"])
            + "_"
            + targ
            + "_"
            + esm
            + "_"
            + scen_name_emus
            + ".pkl",
        )

    return emus_g_T


def create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=True):
    """ Merge local trends and local variability temperature emulations of the same scenarios and targets.
    
    Args:
    - emus_lt (dict): local trend emulations dictionary with keys
        [scen][targ] (2d array (time x grid points) of local trends emulation time series)
    - emus_lv (dict): local variability emulations dictionary with keys
        [scen][targ] (3d array  (emus x time x grid points) of local varaibility emulation time series)
    - params_lt (dict): dictionary containing the calibrated parameters for the local trends emulations, keys relevant here
        ['targs'] (list of emulated variables, list of strs)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
    - params_lv_T (dict): dictionary containing the calibrated parameters for the local variability emulations, keys relevant here
        ['targs'] (list of variables which are emulated, list of strs)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (type of ensemble which is emulated, str)
        ['method'] (applied method, str)
    - cfg (module): config file containnig metadata
    - save_emus (bool,optional): determines if emulation is saved or not, default = True

    Returns:
    - emus_l (dict): local emulations dictionary with keys
        [scen][targ] (3d array  (emus x time x grid points) of local emulation time series)
        
    """

    # specify necessary variables from config file
    dir_mesmer_emus = cfg.dir_mesmer_emus
    scen_name_emus = cfg.scen_name_emus

    scenarios_lt = list(emus_lt.keys())
    scenarios_lv = list(emus_lv.keys())

    targs_lt = list(emus_lt[scenarios_lt[0]].keys())
    targs_lv = list(emus_lv[scenarios_lv[0]].keys())

    if scenarios_lt == scenarios_lv and targs_lt == targs_lv:
        emus_l = {}
        for scen in scenarios_lt:
            emus_l[scen] = {}
            for targ in targs_lt:
                emus_l[scen][targ] = emus_lt[scen][targ] + emus_lv[scen][targ]

    elif scenarios_lt != scenarios_lv:
        print("The scenarios do not match. No local emulation is created.")
        emus_l = []

    elif targs_lt != targs_lv:
        print("The target variables do not match. No local emulation is created.")
        emus_l = []

    if params_lt["esm"] == params_lv["esm"]:
        esm = params_lt["esm"]
    else:
        print("The Earth System Models do not match. No global emulation is created.")
        emus_l = []

    if params_lt["ens_type"] == params_lv["ens_type"]:
        ens_type = params_lt["ens_type"]
    else:
        print("The ensemble types do not match. No global emulation is created.")
        emus_l = []

    ## save the global emus if requested
    if save_emus == True:
        dir_mesmer_emus_l = dir_mesmer_emus + "local/"
        # check if folder to save emus in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_l):
            os.makedirs(dir_mesmer_emus_l)
            print("created dir:", dir_mesmer_emus_l)
        joblib.dump(
            emus_l,
            dir_mesmer_emus_l
            + "emus_l_"
            + ens_type
            + "_lt_"
            + params_lt["method"]
            + "_"
            + "_".join(params_lt["preds"])
            + "_lv_"
            + params_lv["method"]
            + "_"
            + "_".join(params_lv["preds"])
            + "_"
            + "_".join(params_lt["targs"])
            + "_"
            + esm
            + "_"
            + scen_name_emus
            + ".pkl",
        )

    return emus_l
