"""
Collection of functions to calibrate all modules of MESMER.
"""
# flake8: noqa
import logging
import warnings

from .train_gt import *
from .train_gv import *
from .train_lt import *
from .train_lv import *
from .train_utils import *
from .create_emulations import (
    create_emus_g,
    create_emus_gt,
    create_emus_gv,
    create_emus_l,
    create_emus_lt,
    create_emus_lv,
)
from .io import (
    load_cmipng,
    load_phi_gc,
    load_regs_ls_wgt_lon_lat,
    save_mesmer_bundle,
)
from .utils import convert_dict_to_arr, extract_land, separate_hist_future

LOGGER = logging.getLogger(__name__)


def calibrate_mesmer(
    esms,
    scenarios_to_train,
    target_variable,
    reg_type,
    threshold_land,
    output_file,
):
    tas_g_dict = {}  # tas with global coverage
    GSAT_dict = {}  # global mean tas
    GHFDS_dict = {}  # global mean hfds (needed as predictor)
    tas_g = {}
    GSAT = {}
    GHFDS = {}
    time = {}

    for esm in esms:
        LOGGER.info("Loading data for %s", esm)
        tas_g_dict[esm] = {}
        GSAT_dict[esm] = {}
        GHFDS_dict[esm] = {}
        time[esm] = {}

        for scen in scenarios_to_train:
            # TODO: rename tas_g_tmp to target_variable_g_tmp or simply
            #       hard-code tas as always being the target variable
            tas_g_tmp, GSAT_tmp, lon_tmp, lat_tmp, time_tmp = load_cmipng(
                target_variable, esm, scen, cfg
            )

            if tas_g_tmp is None:
                warnings.warn(f"Scenario {scen} does not exist for tas for ESM {esm}")
            else:  # if scen exists: save fields + load hfds fields for it too
                tas_g_dict[esm][scen], GSAT_dict[esm][scen], lon, lat, time[esm][scen] = (
                    tas_g_tmp,
                    GSAT_tmp,
                    lon_tmp,
                    lat_tmp,
                    time_tmp,
                )
                _, GHFDS_dict[esm][scen], _, _, _ = load_cmipng("hfds", esm, scen, cfg)

        tas_g[esm] = convert_dict_to_arr(tas_g_dict[esm])
        GSAT[esm] = convert_dict_to_arr(GSAT_dict[esm])
        GHFDS[esm] = convert_dict_to_arr(GHFDS_dict[esm])

    # load in the constant files
    reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(reg_type, lon, lat)

    # extract land
    tas, reg_dict, ls = extract_land(
        tas_g, reg_dict, wgt_g, ls, threshold_land=threshold_land
    )

    for esm in esms:
        LOGGER.info("Calibrating %s", esm)

        LOGGER.info("Calibrating global trend module")
        params_gt_T = train_gt(GSAT[esm], targ, esm, time[esm], cfg, save_params=True)
        params_gt_hfds = train_gt(GHFDS[esm], "hfds", esm, time[esm], cfg, save_params=True)

        # Do you need to create emulations in order to calibrate the global
        # variability and local trends modules? Or do I misunderstand what is
        # going on here?
        LOGGER.info("Creating global-trend emulations")
        preds_gt = {"time": time[esm]}
        emus_gt_T = create_emus_gt(
            params_gt_T, preds_gt, cfg, concat_h_f=True, save_emus=True
        )
        gt_T_s = create_emus_gt(
            params_gt_T, preds_gt, cfg, concat_h_f=False, save_emus=False
        )

        LOGGER.info(
            "Preparing predictors for global variability, local trends, and "
            "local variability"
        )
        gt_T2_s = {}
        for scen in gt_T_s.keys():
            gt_T2_s[scen] = gt_T_s[scen] ** 2

        gt_hfds_s = create_emus_gt(
            params_gt_hfds, preds_gt, cfg, concat_h_f=False, save_emus=False
        )

        gv_novolc_T = {}
        for scen in emus_gt_T.keys():
            gv_novolc_T[scen] = GSAT[esm][scen] - emus_gt_T[scen]

        gv_novolc_T_s, time_s = separate_hist_future(gv_novolc_T, time[esm], cfg)

        tas_s, time_s = separate_hist_future(tas[esm], time[esm], cfg)

        LOGGER.info("Calibrating global variability module")
        params_gv_T = train_gv(gv_novolc_T_s, target_variable, esm, cfg, save_params=True)

        time_v = {}
        time_v["all"] = time[esm][scen]

        # Are these necessary for calibration?
        LOGGER.info("Creating global variability emulations")
        preds_gv = {"time": time_v}
        emus_gv_T = create_emus_gv(params_gv_T, preds_gv, cfg, save_emus=True)

        LOGGER.info("Joining global trend and global variability emulations")
        emus_g_T = create_emus_g(
            emus_gt_T, emus_gv_T, params_gt_T, params_gv_T, cfg, save_emus=True
        )

        LOGGER.info("Calibrating local trends module")
        preds = {
            "gttas": gt_T_s,
            "gttas2": gt_T2_s,
            "gthfds": gt_hfds_s,
            "gvtas": gv_novolc_T_s,
        }
        targs = {"tas": tas_s}
        params_lt, params_lv = train_lt(preds, targs, esm, cfg, save_params=True)

        # Are these necessary for calibration?
        LOGGER.info("Creating local trends emulations")
        preds_lt = {"gttas": gt_T_s, "gttas2": gt_T2_s, "gthfds": gt_hfds_s}
        lt_s = create_emus_lt(params_lt, preds_lt, cfg, concat_h_f=False, save_emus=True)
        emus_lt = create_emus_lt(params_lt, preds_lt, cfg, concat_h_f=True, save_emus=True)

        LOGGER.info("Calibrating local variability module")
        # derive variability part induced by gv
        preds_lv = {"gvtas": gv_novolc_T_s}  # predictors_list

        # we need to create emulations to train local variability?
        lv_gv_s = create_emus_lv(params_lv, preds_lv, cfg, save_emus=False, submethod="OLS")

        # tas essentially hard-coded here too
        LOGGER.debug(
            "Calculating residual variability i.e. what remains of tas once "
            "lt + gv removed"
        )
        res_lv_s = {}  # residual local variability
        for scen in tas_s.keys():
            res_lv_s[scen] = tas_s[scen] - lt_s[scen]["tas"] - lv_gv_s[scen]["tas"]

        LOGGER.debug("Loading auxiliary files")
        aux = {}
        aux["phi_gc"] = load_phi_gc(
            lon, lat, ls, cfg, L_start=1750, L_end=2000, L_interval=250
        )  # better results with default values L, but like this much faster + less space needed

        LOGGER.debug("Training local variability module on derived data")
        targs_res_lv = {"tas": res_lv_s}
        params_lv = train_lv(
            {}, targs_res_lv, esm, cfg, save_params=True, aux=aux, params_lv=params_lv
        )

        # are seeds needed here?
        save_mesmer_bundle(
            output_file,
            params_lt,
            params_lv,
            params_gv_T,
            seeds=cfg.seed,
            land_fractions=ls["grid_l_m"],
            lat=lat["c"],
            lon=lon["c"],
            time=time_s,
        )
