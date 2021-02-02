"""
mesmer.create_emulations.merge_emus
===================
Functions to merge emulations of different MESMER modules.


Functions:
    create_emus_g()
    create_emus_l()

"""


import os

import joblib


def create_emus_g(emus_gt, emus_gv, params_gt, params_gv, cfg, save_emus=True):
    """Merge global trend and global variability emulations of the same scenarios.

    Args:
    - emus_gt (dict): global trend emulations dictionary with keys
        [scen] (1d array of global trend emulation time series)
    - emus_gv (dict): global variability emulations dictionary with keys
        [scen] (2d array  (emus x time) of global variability emulation time series)
    - params_gt (dict): dictionary containing the calibrated parameters for the global trend emulations, keys relevant here
        ['targ'] (emulated variable, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (ensemble type, str)
        ['method'] (applied method, str)
    - params_gv (dict): dictionary containing the calibrated parameters for the global variability emulations, keys relevant here
        ['targ'] (variable which is emulated, str)
        ['esm'] (Earth System Model, str)
        ['ens_type'] (type of ensemble which is emulated, str)
        ['method'] (applied method, str)
    - cfg (module): config file containnig metadata
    - save_emus (bool,optional): determines if emulation is saved or not, default = True

    Returns:
    - emus_g (dict): global emulations dictionary with keys
        [scen] (2d array  (emus x time) of global emulation time series)

    """

    # specify necessary variables from config file
    dir_mesmer_emus = cfg.dir_mesmer_emus
    scenarios_emus = cfg.scenarios_emus
    scen_name_emus = "_".join(scenarios_emus)

    scenarios_gt = list(emus_gt.keys())
    scenarios_gv = list(emus_gv.keys())
    if scenarios_gv == ['all']:
        scenarios_gv = scenarios_emus

    if scenarios_gt == scenarios_gv:
        emus_g = {}
        for scen in scenarios_gt:
            emus_g[scen] = emus_gt[scen] + emus_gv[scen]

    else:
        print(
            "The global trend and the global variabilty emulations are not from the same scenario, no global emulation is created"
        )
        emus_g = []

    if params_gt["targ"] == params_gv["targ"]:
        targ = params_gt["targ"]
    else:
        print("The target variables do not match. No global emulation is created")
        emus_g = []

    if params_gt["esm"] == params_gv["esm"]:
        esm = params_gt["esm"]
    else:
        print("The Earth System Models do not match. No global emulation is created")
        emus_g = []

    if params_gt["ens_type"] == params_gv["ens_type"]:
        ens_type = params_gt["ens_type"]
    else:
        print("The ensemble types do not match. No global emulation is created")
        emus_g = []

    # save the global emus if requested
    if save_emus:
        dir_mesmer_emus_g = dir_mesmer_emus + "global/"
        # check if folder to save emus in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_g):
            os.makedirs(dir_mesmer_emus_g)
            print("created dir:", dir_mesmer_emus_g)

        filename_parts = [
            "emus_g",
            ens_type,
            "gt",
            params_gt["method"],
            *params_gt["preds"],
            "gv",
            params_gv["method"],
            *params_gv["preds"],
            targ,
            esm,
            scen_name_emus,
        ]
        filename_emus_g = dir_mesmer_emus_g + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_g, filename_emus_g)

    return emus_g


def create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=True):
    """Merge local trends and local variability temperature emulations of the same scenarios and targets.

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
    - params_lv (dict): dictionary containing the calibrated parameters for the local variability emulations, keys relevant here
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

    if targs_lt == targs_lv:
        targs = params_lt["targs"]
    else:
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

    # save the global emus if requested
    if save_emus:
        dir_mesmer_emus_l = dir_mesmer_emus + "local/"
        # check if folder to save emus in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_l):
            os.makedirs(dir_mesmer_emus_l)
            print("created dir:", dir_mesmer_emus_l)
        filename_parts = [
            "emus_l",
            ens_type,
            "lt",
            params_lt["method"],
            *params_lt["preds"],
            "lv",
            params_lv["method"],
            *params_lv["preds"],
            *targs,
            esm,
            scen_name_emus,
        ]
        filename_emus_l = dir_mesmer_emus_l + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_l, filename_emus_l)

    return emus_l
