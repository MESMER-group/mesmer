"""
mesmer.calibrate_mesmer.train_lv
===================
Functions to train local variability module of MESMER.


Functions:
    train_lv()

"""


import os

import joblib
import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.tsa.ar_model import AutoReg


def train_lv(preds_list, targs_list, targ_names, esm, cfg, save_params=True, aux={}):
    # remarks: assumption: lv does not depend on scenario
    # remark: currently no method with preds implemented. But already have in there for consistency reason

    # specify necessary variables from config file
    targ_name = targ_names[0]
    ens_type_tr = cfg.ens_type_tr
    hist_tr = cfg.hist_tr
    preds_lv = cfg.preds[targ_name]["lv"]
    method_lv = cfg.methods[targ_name]["lv"]
    method_lt = cfg.methods[targ_name]["lt"]
    dir_mesmer_params = cfg.dir_mesmer_params

    scenarios_tr = list(targs_list[0].keys())
    if hist_tr:  # check whether historical data is used in training
        scen_name_tr = "hist_" + "_".join(scenarios_tr)
    else:
        scen_name_tr = "_".join(scenarios_tr)

    # prepare the y (can also add prepare the preds_list data if I will actually use preds once)
    # keep individ targs split at this point but will need to revisit this choice in future
    # if make changes: remember: current AR1 implementation requires the separation -> would need to rewrite
    y = {}
    for i in np.arange(len(targ_names)):
        targ = targs_list[i]
        targ_name = targ_names[i]

        # assumption: nr_runs per scen and nr_ts for these runs can vary
        nr_samples = 0
        for scen in scenarios_tr:
            nr_runs, nr_ts, nr_gps = targ[scen].shape
            nr_samples += nr_runs * nr_ts

        y[targ_name] = np.zeros([nr_samples, nr_gps])
        j = 0
        for scen in scenarios_tr:
            k = (
                targ[scen].shape[0] * targ[scen].shape[1]
            )  # nr_runs*nr_ts for this specific scenario
            y[targ_name][j : j + k] = targ[scen].reshape(-1, nr_gps)
            j += k

    if "OLS" in method_lv and "OLS" in method_lt:
        dir_mesmer_params_lv = dir_mesmer_params + "local/local_variability/"
        filename_parts = [
            "params_lv",
            ens_type_tr,
            method_lv,
            *preds_lv,
            *targ_names,
            esm,
            scen_name_tr,
        ]
        filename_params_lv = dir_mesmer_params_lv + "_".join(filename_parts) + ".pkl"
        print("Load existing params_lv dictionary")
        if os.path.exists(filename_params_lv):
            params_lv = joblib.load(filename_params_lv)
        else:
            print(
                "An error occurred. The OLS parameters of the lv method have not been saved when applying train_lt()."
            )
    else:  # assumption: only if OLS in method, I already derive part of lv model in lt training. Could be rewritten in more general way if becomes required
        print("Initialize params_lv dictionary")
        params_lv = {}
        params_lv["targs"] = targ_names
        params_lv["esm"] = esm
        params_lv["ens_type"] = ens_type_tr
        params_lv["method"] = method_lv
        params_lv["preds"] = preds_lv
        params_lv["scenarios"] = scenarios_tr
        params_lv["part_model_in_lt"] = False

    if (
        "AR1_sci" in method_lv
    ):  # assumption: target value I get at this point is already ready for AR1_sci method
        # this is important wrt fact that I want to extend lv methods to include a hybrid
        # Link et al + my stuff approach -> there will need to execute Link et al first

        # assumption: do for each target variable independently.
        # Once I add precip, I need to add other arguments for this
        print(
            "Derive parameters for AR(1) processes with spatially correlated innovations"
        )

        # AR(1)
        params_lv["AR1_int"] = {}
        params_lv["AR1_coef"] = {}
        params_lv["AR1_std_innovs"] = {}
        params_lv["L"] = {}  # localisation radius
        params_lv[
            "ecov"
        ] = {}  # empirical cov matrix of the local variability trained on here
        params_lv["loc_ecov"] = {}  # localized empirical cov matrix
        params_lv[
            "loc_ecov_AR1_innovs"
        ] = {}  # localized empirical cov matrix of the innovations of the AR(1) process

        for targ_name in targ_names:
            nr_samples, nr_gps = y[targ_name].shape  # because dim (nr_samples,nr_gps)

            # AR(1)
            params_lv["AR1_int"][targ_name] = np.zeros(nr_gps)
            params_lv["AR1_coef"][targ_name] = np.zeros(nr_gps)
            params_lv["AR1_std_innovs"][targ_name] = np.zeros(nr_gps)

            for gp in np.arange(nr_gps):
                AR1_model = AutoReg(y[targ_name][:, gp], lags=1).fit()
                params_lv["AR1_int"][targ_name][gp] = AR1_model.params[0]
                params_lv["AR1_coef"][targ_name][gp] = AR1_model.params[1]
                params_lv["AR1_std_innovs"][targ_name][gp] = np.sqrt(
                    AR1_model.sigma2
                )  # sqrt of variance = standard deviation

            # spatial cross-correlations with leave-one-out cross val (= bottelneck for speed)
            L_set = np.sort(list(aux["phi_gc"].keys()))  # the Ls to loop through

            llh_max = -10000
            llh_cv_sum = {}
            idx_L = 0
            L_sel = L_set[idx_L]
            idx_break = False

            while (idx_break == False) and (L_sel < L_set[-1]):
                # experience tells: once stop selecting larger loc radii, will not start again
                # better to stop once max is reached (to avoid singular matrices + limit computational effort)
                L = L_set[idx_L]
                llh_cv_sum[L] = 0

                for sample in np.arange(nr_samples):
                    y_est = np.delete(
                        y[targ_name], sample, axis=0
                    )  # y used to estimate params
                    y_cv = y[targ_name][sample]  # y used to crossvalidate the estimate

                    ecov = np.cov(y_est, rowvar=False)
                    loc_ecov = aux["phi_gc"][L] * ecov
                    mean_0 = np.zeros(
                        aux["phi_gc"][L].shape[0]
                    )  # we want the mean of the res to be 0

                    llh_cv = multivariate_normal.logpdf(
                        y_cv, mean=mean_0, cov=loc_ecov, allow_singular=True
                    )
                    # in case have issues with singular matrices (expected for CMIP6 CanESM5 and MCM-UA-1-0), can add
                    # argument: allow_singular=True in multivariate_normal.logpdf()
                    # alternatively, I could try to just take the last valid entry?
                    # added now because at L=5750 CanESM2 crashed because of singular matrix. I highly suspect this is a resolution problem.
                    # although added, final selected L=5500 -> re-assuring
                    # IPSL-CM5A-LR also had a singular matrix

                    # if I would evaluate several samples at once, e.g., y_est instead of y_cv, would get
                    # 1 number per sample -> could just sum them up:
                    # np.sum(multivariate_normal.logpdf(y_est,mean=mean_0, cov=loc_ecov))

                    llh_cv_sum[L] += llh_cv

                idx_L += 1

                if llh_cv_sum[L] > llh_max:
                    L_sel = L
                    llh_max = llh_cv_sum[L]
                    print("Newly selected L=", L_sel)
                else:
                    print("Final selected L=", L_sel)
                    idx_break = True

            ecov = np.cov(y[targ_name], rowvar=False)
            loc_ecov = aux["phi_gc"][L_sel] * ecov

            # ATTENTION: STILL NEED TO CHECK IF THIS IS TRUE. I UNFORTUNATELY LEARNED THAT I WROTE THIS FORMULA DIFFERENTLY
            # IN THE ESD PAPER!!!!!!! (But I am pretty sure that code is correct and the error is in the paper)
            loc_ecov_AR1_innovs = np.zeros(loc_ecov.shape)
            for i in np.arange(nr_gps):
                for j in np.arange(nr_gps):
                    loc_ecov_AR1_innovs[i, j] = (
                        np.sqrt(1 - params_lv["AR1_coef"][targ_name][i] ** 2)
                        * np.sqrt(1 - params_lv["AR1_coef"][targ_name][j] ** 2)
                        * loc_ecov[i, j]
                    )

            params_lv["L"][targ_name] = L_sel
            params_lv["ecov"][targ_name] = ecov
            params_lv["loc_ecov"][targ_name] = loc_ecov
            params_lv["loc_ecov_AR1_innovs"][targ_name] = loc_ecov_AR1_innovs

    if (
        save_params
    ):  # overwrites lv module if already exists, i.e., assumption: always lt before lv
        dir_mesmer_params_lv = dir_mesmer_params + "local/local_variability/"
        # check if folder to save params in exists, if not: make it
        if not os.path.exists(dir_mesmer_params_lv):
            os.makedirs(dir_mesmer_params_lv)
            print("created dir:", dir_mesmer_params_lv)
        filename_parts = [
            "params_lv",
            ens_type_tr,
            method_lv,
            *preds_lv,
            *targ_names,
            esm,
            scen_name_tr,
        ]
        filename_params_lv = dir_mesmer_params_lv + "_".join(filename_parts) + ".pkl"
        joblib.dump(params_lv, filename_params_lv)

    return params_lv
