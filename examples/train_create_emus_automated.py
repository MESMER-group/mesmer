# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import sys

sys.path.append("../")


# load in configurations used in this script
import configs.config_across_scen_T_cmip6ng_test as cfg

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
from mesmer.utils import (
    convert_dict_to_arr,
    extract_land,
    extract_time_period,
    separate_hist_future,
)

# specify the target variable
targ = cfg.targs[0]
print(targ)

# load in the ESM runs
esms = cfg.esms

# load in tas with global coverage
tas_g_dict = {}  # tas with global coverage
GSAT_dict = {}  # global mean tas
GHFDS_dict = {}  # global mean hfds (needed as predictor)
tas_g = {}
GSAT = {}
GHFDS = {}
time = {}

for esm in esms:
    print(esm)
    tas_g_dict[esm] = {}
    GSAT_dict[esm] = {}
    GHFDS_dict[esm] = {}
    time[esm] = {}

    for scen in cfg.scenarios_tr:

        tas_g_tmp, GSAT_tmp, lon_tmp, lat_tmp, time_tmp = load_cmipng(
            targ, esm, scen, cfg
        )

        if tas_g_tmp is None:
            print("Scenario " + scen + " does not exist for tas for ESM " + esm)
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
reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(cfg.reg_type, lon, lat)

# extract land
tas, reg_dict, ls = extract_land(
    tas_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
)


for esm in esms:
    print(esm)

    print(esm, "Start with global trend module")
    params_gt_T = train_gt(GSAT[esm], targ, esm, time[esm], cfg, save_params=True)
    params_gt_hfds = train_gt(GHFDS[esm], "hfds", esm, time[esm], cfg, save_params=True)

    emus_gt_T = create_emus_gt(
        params_gt_T, cfg, scenarios="tr", concat_h_f=True, save_emus=True
    )
    gt_T_s = create_emus_gt(
        params_gt_T, cfg, scenarios="tr", concat_h_f=False, save_emus=False
    )

    print(
        esm,
        "Start preparing predictors for global variability, local trends, and local variability",
    )
    gt_T2_s = {}
    for scen in gt_T_s.keys():
        gt_T2_s[scen] = gt_T_s[scen] ** 2

    gt_hfds_s = create_emus_gt(
        params_gt_hfds, cfg, scenarios="tr", concat_h_f=False, save_emus=False
    )

    gv_novolc_T = {}
    for scen in emus_gt_T.keys():
        gv_novolc_T[scen] = GSAT[esm][scen] - emus_gt_T[scen]
    gv_novolc_T_s, time_s = separate_hist_future(gv_novolc_T, time[esm], cfg)

    tas_s, time_s = separate_hist_future(tas[esm], time[esm], cfg)

    print(esm, "Start with global variability module")
    params_gv_T = train_gv(gv_novolc_T_s, targ, esm, cfg, save_params=True)
    emus_gv_T = create_emus_gv(params_gv_T, cfg, save_emus=True)

    print(esm, "Merge the global trend and the global variability.")
    emus_g_T = create_emus_g(
        emus_gt_T, emus_gv_T, params_gt_T, params_gv_T, cfg, save_emus=True
    )

    print(esm, "Start with local trends module")
    preds = {
        "gttas": gt_T_s,
        "gttas2": gt_T2_s,
        "gthfds": gt_hfds_s,
        "gvtas": gv_novolc_T_s,
    }  # predictors_list
    targs = {"tas": tas_s}  # targets list
    params_lt, params_lv = train_lt(preds, targs, esm, cfg, save_params=True)

    preds_lt = {"gttas": gt_T_s, "gttas2": gt_T2_s, "gthfds": gt_hfds_s}
    lt_s = create_emus_lt(
        params_lt,
        preds_lt,
        cfg,
        scenarios="tr",
        concat_h_f=False,
        save_emus=True,
    )
    emus_lt = create_emus_lt(
        params_lt,
        preds_lt,
        cfg,
        scenarios="tr",
        concat_h_f=True,
        save_emus=True,
    )

    print(esm, "Start with local variability module")
    print("ATTENTION: CURRENTLY NOT WORKING. Will (hopefully) reintroduce soon.")

    # tas_lv = {}
    # for scen in cfg.scenarios_tr:
    #   tas_lv[scen] = tas_arr[scen] - lt[scen]["tas"]

    # extract the global influence on the variability
    # preds_list = [gv_novolc_T]
    # lv_g = create_emus_lv(
    #   params_lv, preds_list, cfg, scenarios="tr", save_emus=False, submethod="OLS"
    # )

    # tas_rlv = {}  # tas residual local variability in array
    # for scen in cfg.scenarios_tr:
    #   tas_rlv[scen] = tas_lv[scen] - lv_g[scen]["tas"]

    # load in the auxiliary files
    # aux = {}
    # aux["phi_gc"] = load_phi_gc(lon, lat, ls, cfg)

    # train on the residual local variability
    # preds_list = []
    # targs_list = [tas_rlv]
    # targ_names = ["tas"]
    # params_lv = train_lv(
    #   preds_list, targs_list, targ_names, esm, cfg, save_params=True, aux=aux
    # )

    # create local variability emulations
    # preds_list = [emus_gv_T]
    # emus_lv = create_emus_lv(
    #   params_lv, preds_list, cfg, scenarios="emus", save_emus=True, submethod=""
    # )

    # print(esm, "Merge the local trends and the local variability.")
    # emus_l = create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=True)
