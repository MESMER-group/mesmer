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
from mesmer.utils import convert_dict_to_arr, extract_land, extract_time_period

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
scen_max_runs = {}

for esm in esms:
    print(esm)
    tas_g_dict[esm] = {}
    GSAT_dict[esm] = {}
    GHFDS_dict[esm] = {}
    time[esm] = {}
    max_runs = 0

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
            nr_runs = len(GSAT_dict[esm][scen])
            if max_runs < nr_runs:
                max_runs = nr_runs
                scen_max_runs[esm] = scen

    tas_g[esm] = convert_dict_to_arr(tas_g_dict[esm])
    GSAT[esm] = convert_dict_to_arr(GSAT_dict[esm])
    GHFDS[esm] = convert_dict_to_arr(GHFDS_dict[esm])

# load in the constant files
reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(cfg.reg_type, lon, lat)

# extract land
tas, reg_dict, ls = extract_land(
    tas_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
)


for esm in cfg.esms:
    print(esm)

    print(esm, "Start with the global trend module.")
    params_gt_T = train_gt(GSAT[esm], targ, esm, time[esm], cfg, save_params=True)
    params_gt_hfds = train_gt(GHFDS[esm], "hfds", esm, time[esm], cfg, save_params=True)

    # create emulation of global trend T from training scenarios to be used as predictor for lt training
    gt_T = create_emus_gt(params_gt_T, cfg, scenarios="tr", save_emus=False)

    # create emulation of global trend T for emulation scenarios to be used as predictor for lt emulation
    # (same in this ic setup but done for completeness)
    emus_gt_T = create_emus_gt(params_gt_T, cfg, scenarios="emus", save_emus=True)

    # create emulation of global trend hfds from training scenarios to be used as predictor for lt emulation
    gt_hfds = create_emus_gt(params_gt_hfds, cfg, scenarios="tr", save_emus=True)

    # create emulation of global trend hfds from emulation scenarios to be used as predictor for lt emulation
    # (same in this ic setup but done for completeness)
    emus_gt_hfds = create_emus_gt(params_gt_hfds, cfg, scenarios="emus", save_emus=True)

    print(
        esm,
        "Prepare predictors for global variability, local trend, and local variability module.",
    )
    # extract global variability left after removing smooth trend with volcanic spikes
    gv_novolc_T = {}
    gt_T2 = {}
    for scen in gt_T.keys():
        gt_T2[scen] = gt_T[scen] ** 2
        gv_novolc_T[scen] = GSAT[esm][scen] - gt_T[scen]

    # turn hist into a "scenario" -> useful for gv, lt, lv (but still not very clean with my naming conventions)
    tas_tmp = {}
    gt_T_tmp = {}
    gt_T2_tmp = {}
    gt_hfds_tmp = {}
    gv_novolc_T_tmp = {}

    tas_tmp["hist"], time_tmp = extract_time_period(
        tas[esm][scen_max_runs[esm]], time[esm][scen_max_runs[esm]], 1850, 2014
    )
    gt_T_tmp["hist"], time_tmp = extract_time_period(
        gt_T[scen_max_runs[esm]], time[esm][scen_max_runs[esm]], 1850, 2014
    )
    gt_T2_tmp["hist"], time_tmp = extract_time_period(
        gt_T2[scen_max_runs[esm]], time[esm][scen_max_runs[esm]], 1850, 2014
    )
    gt_hfds_tmp["hist"], time_tmp = extract_time_period(
        gt_hfds[scen_max_runs[esm]], time[esm][scen_max_runs[esm]], 1850, 2014
    )
    gv_novolc_T_tmp["hist"], time_tmp = extract_time_period(
        gv_novolc_T[scen_max_runs[esm]], time[esm][scen_max_runs[esm]], 1850, 2014
    )

    for scen in GSAT[esm].keys():
        tas_tmp[scen], time_tmp = extract_time_period(
            tas[esm][scen], time[esm][scen], 2015, 2100
        )
        gt_T_tmp[scen], time_tmp = extract_time_period(
            gt_T[scen], time[esm][scen], 2015, 2100
        )
        gt_T2_tmp[scen], time_tmp = extract_time_period(
            gt_T2[scen], time[esm][scen], 2015, 2100
        )
        gt_hfds_tmp[scen], time_tmp = extract_time_period(
            gt_hfds[scen], time[esm][scen], 2015, 2100
        )
        gv_novolc_T_tmp[scen], time_tmp = extract_time_period(
            gv_novolc_T[scen], time[esm][scen], 2015, 2100
        )

    print(esm, "Start with global variability module.")
    params_gv_T = train_gv(gv_novolc_T_tmp, targ, esm, cfg, save_params=True)
    emus_gv_T = create_emus_gv(params_gv_T, cfg, save_emus=True)

    print(esm, "Merge the global trend and the global variability.")
    emus_g_T = create_emus_g(
        emus_gt_T, emus_gv_T, params_gt_T, params_gv_T, cfg, save_emus=True
    )

    print(esm, "Start with local modules.")

    print(esm, "Start with local trends module.")
    preds = {
        "gttas": gt_T_tmp,
        "gttas2": gt_T2_tmp,
        "gthfds": gt_hfds_tmp,
        "gvtas": gv_novolc_T_tmp,
    }  # predictors_list including gv
    targs = {"tas": tas_tmp}  # targets list
    params_lt, params_lv = train_lt(preds, targs, esm, cfg, save_params=True)

    preds = {
        "gttas": gt_T_tmp,
        "gttas2": gt_T2_tmp,
        "gthfds": gt_hfds_tmp,
    }  # predictors_list just trends
    lt = create_emus_lt(params_lt, preds, cfg, scenarios="tr", save_emus=False)

    preds_emus_lt = {"gttas": gt_T, "gttas2": gt_T2, "gthfds": gt_hfds}
    emus_lt = create_emus_lt(
        params_lt, preds_emus_lt, cfg, scenarios="emus", save_emus=True
    )
    preds_list_emus_lt = [emus_gt_T]

    print(esm, "Start with local variability module.")
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
