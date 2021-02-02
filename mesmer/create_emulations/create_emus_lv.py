"""
mesmer.create_emulations.create_emus_lv
===================
Functions to create local variability emulations with MESMER.


Functions:
    create_emus_lv()

"""


import os

import joblib
import numpy as np


def create_emus_lv(
    params_lv, preds_list, cfg, scenarios="emus", save_emus=True, submethod=""
):

    # TODO: change method with submethod in saving

    if (
        len(submethod) == 0
    ):  # if no submethod is specified, the actual method contained in params_lv is employed
        submethod = params_lv["method"]

    # specify necessary variables from config file
    if scenarios == "emus":
        scenarios_emus = cfg.scenarios_emus_v
        scen_name_emus = cfg.scen_name_emus_v
    elif scenarios == "tr":  # not sure if even needed?!
        scenarios_emus = params_lv["scenarios"]
        if cfg.hist_tr:  # check whether historical data was used in training
            scen_name_emus = "hist_" + "_".join(scenarios_emus)
        else:
            scen_name_emus = "_".join(scenarios_emus)

    dir_mesmer_emus = cfg.dir_mesmer_emus
    nr_emus_all_scens = cfg.nr_emus
    nr_ts_emus_v_all_scens = cfg.nr_ts_emus_v
    seed_all_scens = cfg.seed[params_lv["esm"]]

    # format the predictors to fulfill requirements
    if len(preds_list) == len(
        params_lv["preds"]
    ):  # basic check if correct list of predictors: just number, not content
        preds_lv = {}
        for scen in scenarios_emus:
            preds_lv[scen] = {}
            i = 0
            for pred in params_lv["preds"]:
                preds_lv[scen][pred] = preds_list[i][scen]
                # assumption: same predictors for all targets, each pred = arr shape (nr_runs,nr_ts)
                i += 1
    else:
        print("Wrong list of predictors was passed. The emulations cannot be created.")

    # carry out emulations
    emus_lv = {}
    for scen in scenarios_emus:
        emus_lv[scen] = {}
        for targ in params_lv["targs"]:

            if (
                "OLS" in submethod and params_lv["method_lt_each_gp_sep"]
            ):  # assumption: coefs are the same across scens
                print("Start with OLS")
                nr_emus, nr_ts_emus_v = preds_lv[scen][pred].shape
                nr_gps = len(params_lv["coef_" + params_lv["preds"][0]][targ])
                emus_lv[scen][targ] = np.zeros(
                    [nr_emus, nr_ts_emus_v, nr_gps]
                )  # nr ts x nr gp
                for run in np.arange(nr_emus):
                    for gp in np.arange(nr_gps):
                        emus_lv[scen][targ][run, :, gp] = sum(
                            [
                                params_lv["coef_" + pred][targ][gp]
                                * preds_lv[scen][pred][run]
                                for pred in params_lv["preds"]
                            ]
                        )

            # HERE ALTERNATIVE METHODS COULD BE PLACED WHICH ARE EXECUTED BEFORE AR1_sci
            # e.g., my idea to do hybrid between Link et al and my things
            # if add new method to replace OLS: do not forget to initialize emus_lv[scen]!

            if "AR1_sci" in submethod:
                # assumption: do for each target variable independently.
                # Once I add precip, I likely need to extend approach to account for different possibility

                print("Start with AR(1) with spatially correlated innovations.")
                seed = seed_all_scens[scen]["lv"]

                # in case no emus_lv exist yet, initialize it. Otherwise add to it
                if len(emus_lv[scen]) == 0:
                    nr_emus = nr_emus_all_scens[
                        scen
                    ]  # have to get values from config file because no predictors here
                    nr_ts_emus_v = nr_ts_emus_v_all_scens[scen]
                    nr_gps = len(params_lv["AR1_int"][targ])
                    emus_lv[scen][targ] = np.zeros(
                        nr_emus, nr_ts_emus_v, nr_gps
                    )  # initialize if doesn't exist yet, otherwise just build up on exisiting one

                # ensure reproducibility
                np.random.seed(seed)

                # buffer so that initial start at 0 does not influence overall result
                buffer = 20

                print("Draw the innovations")
                # draw the innovations
                innovs = np.random.multivariate_normal(
                    np.zeros(nr_gps),
                    params_lv["loc_ecov_AR1_innovs"][targ],
                    size=[nr_emus, nr_ts_emus_v + buffer],
                )

                print(
                    "Compute the contribution to emus_lv by the AR(1) process with the spatially correlated innovations"
                )
                emus_lv_tmp = np.zeros([nr_emus, nr_ts_emus_v + buffer, nr_gps])
                for t in np.arange(1, nr_ts_emus_v + buffer):
                    emus_lv_tmp[:, t, :] = (
                        params_lv["AR1_int"][targ]
                        + params_lv["AR1_coef"][targ] * emus_lv_tmp[:, t - 1, :]
                        + innovs[:, t, :]
                    )
                emus_lv_tmp = emus_lv_tmp[:, buffer:, :]

                print("Create the full local variability emulations")
                emus_lv[scen][targ] += emus_lv_tmp

    # save the local trends emulation if requested
    if save_emus:
        dir_mesmer_emus_lv = dir_mesmer_emus + "local/local_variability/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_emus_lv):
            os.makedirs(dir_mesmer_emus_lv)
            print("created dir:", dir_mesmer_emus_lv)
        filename_parts = [
            "emus_lv",
            params_lv["ens_type"],
            submethod,
            *params_lv["preds"],
            *params_lv["targs"],
            params_lv["esm"],
            scen_name_emus,
        ]
        filename_emus_lv = dir_mesmer_emus_lv + "_".join(filename_parts) + ".pkl"
        joblib.dump(emus_lv, filename_emus_lv)

    return emus_lv
