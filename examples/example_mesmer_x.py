# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import os
import sys
import joblib

sys.path.append("../")

# additional packages for this script
import numpy as np
import xarray as xr

# import MESMER tools
# from mesmer.calibrate_mesmer import train_lv
# from mesmer.create_emulations import create_emus_lv
# from mesmer.mesmer_x import *
import mesmer.mesmer_x.train_l_distrib_mesmerx as mesmer_x_train
import mesmer.mesmer_x.train_utils_mesmerx as mesmer_x_train_utils

# load in MESMER-X configurations used in this script
from mesmer.mesmer_x.temporary_config_all import ConfigMesmerX
from mesmer.mesmer_x.temporary_support import load_inputs_MESMERx
from mesmer.utils import separate_hist_future

# load in MESMER scripts for treatment of data
# from mesmer.io import (
#     load_cmip,
#     load_phi_gc,
#     load_regs_ls_wgt_lon_lat,
#     test_combination_vars,
# )


def main():

    # ==============================================================
    # 0. OPTIONS FOR THE SCRIPT
    # ==============================================================
    # variables to represent
    targ = "mrso"  # txx, mrso, fwils, fwisa, fwixd, fwixx, mrso_minmon mrsomean??!
    pred = "tas"
    sub_pred = None  # 'hfds' hfds | (pr)

    # options for server
    # To control if run everything or using a bunch of processes
    run_on_exo = False
    # ==============================================================
    # ==============================================================

    # ==============================================================
    # 1. PREPARATION OF MESMER-X
    # ==============================================================
    # short preparation
    # Priority of this script on the server. Value in [-20,19], default at 0, higher is nicer to others
    if run_on_exo:
        os.nice(19)
    subindex_csl = int(sys.argv[1]) if run_on_exo else None
    runs_per_process = 3

    # paths
    path_save_figures = "/home/yquilcaille/mesmer-x_figures/" + targ
    path_save_results = "/net/exo/landclim/yquilcaille/mesmer-x_results/" + targ

    # configuration
    gen = 6
    dir_cmipng = "/net/atmos/data/cmip" + str(gen) + "-ng/"
    # /net/ch4/data/cmip6-Next_Generation/mrso/ann/g025

    dir_cmip_X = {
        "txx": "/net/cfc/landclim1/mathause/projects/IPCC_AR6_CH11/IPCC_AR6_CH11/data/cmip6/tasmax/txx_regrid",
        "mrso": "/net/ch4/data/cmip6-Next_Generation/mrso/ann/g025",
        "mrsomean": "/landclim/yquilcaille/annual_indicators/mrsomean",
        "mrso_minmon": "/landclim/yquilcaille/annual_indicators/mrso_minmon/ann/g025",
        "fwixx": "/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwixx/ann/g025",
        "fwisa": "/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwisa/ann/g025",
        "fwixd": "/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwixd/ann/g025",
        "fwils": "/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwils/ann/g025",
    }[targ]
    # '/net/cfc/landclim1/mathause/projects/IPCC_AR6_CH11/IPCC_AR6_CH11/data/cmip6/mrso/sm_annmean'

    # observations
    dir_obs = "/net/exo/landclim/yquilcaille/mesmer-x/data/observations/"
    # auxiliary data
    dir_aux = "/net/exo/landclim/yquilcaille/mesmer-x/data/auxiliary/"
    dir_mesmer_params = "/net/exo/landclim/yquilcaille/mesmer-x/calibrated_parameters/"
    dir_mesmer_emus = "/net/exo/landclim/yquilcaille/mesmer-x/emulations/"
    # emulation statistics
    dir_stats = "/net/exo/landclim/yquilcaille/mesmer-x/statistics/"
    # plots
    dir_plots = "/net/exo/landclim/yquilcaille/mesmer-x/plots/"

    cfg = ConfigMesmerX(
        gen=gen,
        paths={
            "dir_cmipng": dir_cmipng,
            "dir_cmip_X": dir_cmip_X,
            "dir_obs": dir_obs,
            "dir_aux": dir_aux,
            "dir_mesmer_params": dir_mesmer_params,
            "dir_mesmer_emus": dir_mesmer_emus,
            "dir_stats": dir_stats,
            "dir_plots": dir_plots,
        },
        esms="all",
    )

    # make paths if not existing
    for path in [path_save_results, path_save_figures]:
        if not os.path.exists(path):
            os.makedirs(path)

    # ==============================================================
    # ==============================================================

    # ==============================================================
    # 2. PREPARATION: based on data structures of the former version of MESMER
    # ==============================================================
    print("-------------------------------------------")
    print("Objective: " + targ)
    print("Predictor for the objective: " + pred)
    if sub_pred is not None:
        print("Sub-predictor for the predictor: " + sub_pred)
    print("-------------------------------------------")
    print(" ")

    # load in the ESM runs
    # just taking 1 esm (CanESM5) with several ensemble members for refactoring
    esms = [cfg.esms[3]]  # cfg.esms
    if run_on_exo:
        esms = esms[
            subindex_csl * runs_per_process : (subindex_csl + 1) * runs_per_process
        ]

    # preparing data
    (
        time,
        PRED,
        SUB_PRED,
        reg_dict,
        ls,
        wgt_g,
        lon,
        lat,
        land_targ,
        land_pred,
        phi_gc,
        ind,
        gp2reg,
        ww_reg,
        used_esms,
        dico_gps_nan,
    ) = load_inputs_MESMERx(cfg, [targ, pred, sub_pred], esms)

    # just for readibility
    print(" ")
    # ==============================================================
    # ==============================================================

    # ==============================================================
    # 3. EXAMPLE TO RUN MESMER-X
    # ==============================================================
    # --------------------------------------------------------------
    # 3.1. PREPARING DATA FOR EXAMPLE: temporary reformating of data: based on list of xarrays, one per scenario
    # --------------------------------------------------------------
    # esm test
    esm = "CanESM5"

    # variable
    land_targ_s, time_s = separate_hist_future(land_targ[esm], time[esm], cfg)

    # predictor GMT at t
    pred_s, time_s = separate_hist_future(PRED[esm], time[esm], cfg)

    # predictor GMT at t-1
    tmp_pred = {
        scen: np.hstack([PRED[esm][scen][:, 0, np.newaxis], PRED[esm][scen][:, :-1]])
        for scen in PRED[esm].keys()
    }
    tmp_pred_s, time_s = separate_hist_future(tmp_pred, time[esm], cfg)

    # preparing predictors
    predictors = []
    for scen in pred_s.keys():
        predictors.append((xr.Dataset(), scen))
        predictors[-1][0]["GMT_t"] = xr.DataArray(
            pred_s[scen],
            coords={"member": np.arange(pred_s[scen].shape[0]), "time": time_s[scen]},
            dims=(
                "member",
                "time",
            ),
        )
        predictors[-1][0]["GMT_tm1"] = xr.DataArray(
            tmp_pred_s[scen],
            coords={"member": np.arange(pred_s[scen].shape[0]), "time": time_s[scen]},
            dims=(
                "member",
                "time",
            ),
        )

    # preparing target
    target = []
    for scen in land_targ_s.keys():
        target.append((xr.Dataset(), scen))
        target[-1][0][targ] = xr.DataArray(
            land_targ_s[scen],
            coords={
                "member": np.arange(land_targ_s[scen].shape[0]),
                "time": time_s[scen],
                "gridpoint": np.arange(land_targ_s[scen].shape[2]),
            },
            dims=(
                "member",
                "time",
                "gridpoint",
            ),
        )
    # --------------------------------------------------------------
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # 3.2. EXAMPLE OF TRAINING
    # --------------------------------------------------------------
    expr_name = "cfgA"
    expr = "norm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __GMT_t__ + c4 * __GMT_tm1__ - c5) ), scale=c6)"
    # potential solution for extreme precipitations
    # expr = "genextreme(loc=c1 + c2 * __GMT_t__, scale=c3*(c1 + c2 * __GMT_t__), c=c7)"
    # testing
    # expr = "skewnorm(loc=c1 + (c2 - c1) / ( 1 + np.exp(c3 * __GMT_t__ + c4 * __GMT_tm1__ - c5) ), scale=c6, a=c7)"
    # testing
    # expr = "norm(loc=c1 + c3 * __GMT_t__ + c4 * __GMT_tm1__, scale=c6)"

    # training conditional distributions following 'expr' in all grid points
    xr_coeffs_distrib, xr_qual = mesmer_x_train.xr_train_distrib(
        predictors=predictors,
        target=target,
        target_name=targ,
        expr=expr,
        expr_name=expr_name,
        option_2ndfit=False,
        r_gasparicohn_2ndfit=500,
        scores_fit=["func_optim", "NLL", "BIC"],
    )

    # probability integral transform: projection of the data on a standard normal distribution
    transf_target = mesmer_x_train_utils.probability_integral_transform(
        data=target,
        expr_start=expr,
        coeffs_start=xr_coeffs_distrib,
        preds_start=predictors,
        expr_end="norm(loc=0, scale=1)",
    )

    # training of auto-regression with spatially correlated innovations
    # NEW CODE OF MESMER: not applied on residuals, but on 'transf_target'
    # TODO this below is just here to satisfy the linter but eventually
    # we should use transf_target to train the tempospatial corellated innovations here
    transf_target = transf_target * 1 # dummy use of transf_target

    # --------------------------------------------------------------
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # 3.3. EXAMPLE OF EMULATION
    # --------------------------------------------------------------
    # new scenario:
    # NEW CODE OF MESMER, same structure as 'predictors' used for training, output=preds_newscen

    preds_newscen = []  # TODO

    # generate realizations based on the auto-regression with spatially correlated innovations
    # NEW CODE OF MESMER, output = 'transf_emus'
    transf_emus = []  # TODO

    # probability integral transform: projection of the transformed data on the knwon distributions
    emus = mesmer_x_train_utils.probability_integral_transform(
        data=transf_emus,
        expr_start="norm(loc=0, scale=1)",
        expr_end=expr,
        coeffs_end=xr_coeffs_distrib,
        preds_end=preds_newscen,
    )

    joblib.dump(emus, cfg.paths["dir_mesmer_emus"] + "emus_" + targ + ".pkl")
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # ==============================================================
    # ==============================================================


if __name__ == "__main__":
    main()
