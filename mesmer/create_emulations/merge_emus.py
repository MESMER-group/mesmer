# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to merge emulations of different MESMER modules.
"""


import os
import warnings

import joblib


def create_emus_g(emus_gt, emus_gv, params_gt, params_gv, cfg, save_emus=True):
    """Merge global trend and global variability emulations of the same scenarios.

    Parameters
    ----------
    emus_gt : dict
        global trend emulations dictionary with keys

        - [scen] (1d array of global trend emulation time series)
    emus_gv : dict
        global variability emulations dictionary with keys

        - [scen] (2d array  (emus, time) of global variability emulation time series)
    params_gt : dict
        dictionary containing the calibrated parameters for the global trend emulations,
        keys relevant here:

        - ["targ"] (emulated variable, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
    params_gv : dict
        dictionary containing the calibrated parameters for the global variability
        emulations, keys relevant here:

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
    cfg : module
        config file containing metadata
    save_emus : bool, optional
        determines if emulation is saved or not, default = True

    Returns
    -------
    emus_g : dict
        global emulations dictionary with keys

        - [scen] (2d array (emus, time) of global emulation time series)
    """
    scenarios_gt = list(emus_gt.keys())
    scenarios_gv = list(emus_gv.keys())

    emus_g = {}
    if scenarios_gt == scenarios_gv:
        for scen in scenarios_gt:
            emus_g[scen] = emus_gt[scen] + emus_gv[scen]
    elif scenarios_gv == ["all"]:
        for scen in scenarios_gt:
            emus_g[scen] = emus_gt[scen] + emus_gv["all"]
    else:
        warnings.warn(
            "The global trend and the global variabilty emulations are not from the"
            " same scenario, no global emulation is created."
        )
        emus_g = []

    if params_gt["targ"] == params_gv["targ"]:
        targ = params_gt["targ"]
    else:
        warnings.warn(
            "The target variables do not match. No global emulation is created."
        )
        emus_g = []

    if params_gt["esm"] == params_gv["esm"]:
        esm = params_gt["esm"]
    else:
        warnings.warn(
            "The Earth System Models do not match. No global emulation is created."
        )
        emus_g = []

    # save the global emus if requested
    if save_emus:
        dir_mesmer_emus = cfg.dir_mesmer_emus
        dir_mesmer_emus_g = dir_mesmer_emus + "global/"
        # check if folder to save emus in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_g):
            os.makedirs(dir_mesmer_emus_g)
            print("created dir:", dir_mesmer_emus_g)

        filename_parts = [
            "emus_g",
            "gt",
            params_gt["method"],
            *params_gt["preds"],
            "gv",
            params_gv["method"],
            *params_gv["preds"],
            targ,
            esm,
            *scenarios_gt,
        ]
        filename_emus_g = dir_mesmer_emus_g + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_g, filename_emus_g)

    return emus_g


def create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=True):
    """
    Merge local trends and local variability temperature emulations of the same
    scenarios and targets.

    Parameters
    ----------
    emus_lt : dict
        local trend emulations dictionary with keys

        - [scen][targ] (2d array (time x grid points) of local trends emulation time
          series)
    emus_lv : dict
        local variability emulations dictionary with keys

        - [scen][targ] (3d array  (emus x time x grid points) of local varaibility
          emulation time series)
    params_lt : dict
        dictionary containing the calibrated parameters for the local trends emulations,
        keys relevant here

        - ["targs"] (list of emulated variables, list of strs)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
    params_lv : dict
        dictionary containing the calibrated parameters for the local variability
        emulations, keys relevant here

        - ["targs"] (list of variables which are emulated, list of strs)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
    cfg : module
        config file containing metadata
    save_emus : bool, optional
        determines if emulation is saved or not, default = True

    Returns
    -------
    emus_l : dict
        local emulations dictionary with keys

        - [scen][targ] (3d array  (emus, time, grid points) of local emulation time
          series)

    """

    # specify necessary variables from config file
    if save_emus:
        dir_mesmer_emus = cfg.dir_mesmer_emus

    scenarios_lt = list(emus_lt.keys())
    scenarios_lv = list(emus_lv.keys())

    targs_lt = list(emus_lt[scenarios_lt[0]].keys())
    targs_lv = list(emus_lv[scenarios_lv[0]].keys())

    emus_l = {}
    if scenarios_lt == scenarios_lv and targs_lt == targs_lv:
        for scen in scenarios_lt:
            emus_l[scen] = {}
            for targ in targs_lt:
                emus_l[scen][targ] = emus_lt[scen][targ] + emus_lv[scen][targ]
    elif scenarios_lv == ["all"] and targs_lt == targs_lv:
        for scen in scenarios_lt:
            emus_l[scen] = {}
            for targ in targs_lt:
                emus_l[scen][targ] = emus_lt[scen][targ] + emus_lv["all"][targ]
    else:
        warnings.warn("The scenarios do not match. No local emulation is created.")
        emus_l = []

    if targs_lt == targs_lv:
        targs = params_lt["targs"]
    else:
        warnings.warn(
            "The target variables do not match. No local emulation is created."
        )
        emus_l = []

    if params_lt["esm"] == params_lv["esm"]:
        esm = params_lt["esm"]
    else:
        warnings.warn(
            "The Earth System Models do not match. No local emulation is created."
        )
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
            "lt",
            params_lt["method"],
            *params_lt["preds"],
            "lv",
            params_lv["method"],
            *params_lv["preds"],
            *targs,
            esm,
            *scenarios_lt,
        ]
        filename_emus_l = dir_mesmer_emus_l + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_l, filename_emus_l)

    return emus_l
