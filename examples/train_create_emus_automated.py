# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import sys

# load in configurations used in this script
import configs.config_cmip5ng_tas_rcp85_obs as cfg

# import MESMER tools
from mesmer.calibrate_mesmer import train_gt, train_gv, train_lt, train_lv
from mesmer.create_emulations import (
    create_emus_g,
    create_emus_gt,
    create_emus_gv,
    create_emus_l,
    create_emus_lt,
    create_emus_lv,
)
from mesmer.io import load_cmipng, load_phi_gc, load_regs_ls_wgt_lon_lat
from mesmer.utils import convert_dict_to_arr, extract_land

sys.path.append("../")


# specify the target variable
targ = cfg.targs[0]
print(targ)

# load in the ESM runs

# load in tas with global coverage
tas_g = {}  # tas with global coverage
GSAT = {}  # global mean tas

for esm in cfg.esms:
    print(esm)
    tas_g[esm] = {}
    GSAT[esm] = {}
    for scen in cfg.scenarios_tr:
        tas_g[esm][scen], GSAT[esm][scen], lon, lat, time = load_cmipng(
            targ, esm, scen, cfg
        )

# load in the constant files
reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(cfg.reg_type, lon, lat)

# extract land
tas, reg_dict, ls = extract_land(
    tas_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
)

for esm in cfg.esms:
    print(esm)

    print("Convert dictionaries to arrays for better handling")
    tas_arr = convert_dict_to_arr(tas[esm])
    GSAT_arr = convert_dict_to_arr(GSAT[esm])

    print(esm, "Start with global mean temperature modules")
    print(esm, "Start with the global trend module.")
    params_gt_T = train_gt(GSAT_arr, targ, esm, time, cfg, save_params=True)

    # create emulation of global trend from training scenarios to be used as predictors for lt training
    gt_T = create_emus_gt(params_gt_T, cfg, scenarios="tr", save_emus=False)

    # create emulation of global trend for emulation scenarios to be used as predictors for lt emulation
    # (same in this ic setup but done for completeness)
    emus_gt_T = create_emus_gt(params_gt_T, cfg, scenarios="emus", save_emus=True)

    # extract global variability left after removing smooth trend with volcanic spikes
    gv_novolc_T = {}
    for scen in cfg.scenarios_tr:
        gv_novolc_T[scen] = GSAT_arr[scen] - gt_T[scen]

    params_gv_T = train_gv(gv_novolc_T, targ, esm, cfg, save_params=True)

    emus_gv_T = create_emus_gv(params_gv_T, cfg, save_emus=True)

    print(esm, "Merge the global trend and the global variability.")
    emus_g_T = create_emus_g(
        emus_gt_T, emus_gv_T, params_gt_T, params_gv_T, cfg, save_emus=True
    )

    print(esm, "Start with the local modules.")

    print(esm, "Start with the local trends module.")
    preds_list = [gt_T, gv_novolc_T]  # predictors_list
    targs_list = [tas_arr]  # targets list
    targ_names = ["tas"]
    params_lt, params_lv = train_lt(
        preds_list, targs_list, targ_names, esm, cfg, save_params=True
    )

    preds_list_emus_lt = [emus_gt_T]

    lt = create_emus_lt(
        params_lt, preds_list_emus_lt, cfg, scenarios="tr", save_emus=False
    )
    emus_lt = create_emus_lt(
        params_lt, preds_list_emus_lt, cfg, scenarios="emus", save_emus=True
    )

    tas_lv = {}
    for scen in cfg.scenarios_tr:
        tas_lv[scen] = tas_arr[scen] - lt[scen]["tas"]

    print(esm, "Start with the local variability module.")

    # extract the global influence on the variability
    preds_list = [gv_novolc_T]
    lv_g = create_emus_lv(
        params_lv, preds_list, cfg, scenarios="tr", save_emus=False, submethod="OLS"
    )

    tas_rlv = {}  # tas residual local variability in array
    for scen in cfg.scenarios_tr:
        tas_rlv[scen] = tas_lv[scen] - lv_g[scen]["tas"]

    # load in the auxiliary files
    aux = {}
    aux["phi_gc"] = load_phi_gc(lon, lat, ls, cfg)

    # train on the residual local variability
    preds_list = []
    targs_list = [tas_rlv]
    targ_names = ["tas"]
    params_lv = train_lv(
        preds_list, targs_list, targ_names, esm, cfg, save_params=True, aux=aux
    )

    # create local variability emulations
    preds_list = [emus_gv_T]
    emus_lv = create_emus_lv(
        params_lv, preds_list, cfg, scenarios="emus", save_emus=True, submethod=""
    )

    print(esm, "Merge the local trends and the local variability.")
    emus_l = create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=True)
