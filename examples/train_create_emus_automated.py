import sys
import warnings

# add pathway to configs folder
sys.path.append("../")

# load in configurations used in this script
import configs.config_across_scen_T_cmip6ng_test as cfg

# import MESMER tools
from mesmer.calibrate_mesmer import train_gt, train_gv, train_lt, train_lv
from mesmer.create_emulations import (
    create_emus_gt,
    create_emus_gv,
    create_emus_l,
    create_emus_lt,
    create_emus_lv,
)
from mesmer.io import load_cmipng, load_phi_gc, load_regs_ls_wgt_lon_lat
from mesmer.utils import convert_dict_to_arr, extract_land, separate_hist_future

# specify the target variable
targ = cfg.targs[0]
print(targ)

# load in the ESM runs
esms = cfg.esms
print(esms)
print(len(esms))

# load in tas with global coverage
tas_g_dict = {}  # tas with global coverage
GSAT_dict = {}  # global mean tas
tas_g = {}
GSAT = {}
time = {}

for esm in esms:
    print(esm)
    tas_g_dict[esm] = {}
    GSAT_dict[esm] = {}
    time[esm] = {}

    for scen in cfg.scenarios:

        tas_g_tmp, GSAT_tmp, lon_tmp, lat_tmp, time_tmp = load_cmipng(
            targ, esm, scen, cfg
        )

        if tas_g_tmp is None:
            warnings.warn(f"Scenario {scen} does not exist for tas for ESM {esm}")
        else:  # if scen exists: save fields
            tas_g_dict[esm][scen], GSAT_dict[esm][scen], lon, lat, time[esm][scen] = (
                tas_g_tmp,
                GSAT_tmp,
                lon_tmp,
                lat_tmp,
                time_tmp,
            )

    tas_g[esm] = convert_dict_to_arr(tas_g_dict[esm])
    GSAT[esm] = convert_dict_to_arr(GSAT_dict[esm])

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

    preds_gt = {"time": time[esm]}
    gt_T_s = create_emus_gt(
        params_gt_T, preds_gt, cfg, concat_h_f=False, save_emus=False
    )
    emus_gt_T = create_emus_gt(
        params_gt_T, preds_gt, cfg, concat_h_f=True, save_emus=True
    )

    print(
        esm,
        "Start preparing predictors for global variability, local trends, and local variability",
    )

    GSAT_s, time_s = separate_hist_future(GSAT[esm], time[esm], cfg)
    gv_novolc_T_s = {}
    for scen in gt_T_s.keys():
        gv_novolc_T_s[scen] = GSAT_s[scen] - gt_T_s[scen]

    tas_s, time_s = separate_hist_future(tas[esm], time[esm], cfg)

    print(esm, "Start with global variability module")

    params_gv_T = train_gv(gv_novolc_T_s, targ, esm, cfg, save_params=True)

    time_v = {}
    scen = list(emus_gt_T.keys())[0]
    time_v["all"] = time[esm][scen]
    preds_gv = {"time": time_v}
    emus_gv_T = create_emus_gv(params_gv_T, preds_gv, cfg, save_emus=True)

    print(esm, "Start with local trends module")

    preds = {
        "gttas": gt_T_s,
        "gvtas": gv_novolc_T_s,
    }
    targs = {"tas": tas_s}
    params_lt, params_lv = train_lt(preds, targs, esm, cfg, save_params=True)

    preds_lt = {"gttas": gt_T_s}
    lt_s = create_emus_lt(params_lt, preds_lt, cfg, concat_h_f=False, save_emus=True)
    emus_lt = create_emus_lt(params_lt, preds_lt, cfg, concat_h_f=True, save_emus=True)

    print(esm, "Start with local variability module")

    # derive variability part induced by gv
    preds_lv = {"gvtas": gv_novolc_T_s}  # predictors_list
    lv_gv_s = create_emus_lv(params_lv, preds_lv, cfg, save_emus=False, submethod="OLS")

    # derive residual variability i.e. what remains of tas once lt + gv removed
    res_lv_s = {}  # residual local variability
    for scen in tas_s.keys():
        res_lv_s[scen] = tas_s[scen] - lt_s[scen]["tas"] - lv_gv_s[scen]["tas"]

    # load in the auxiliary files
    aux = {}
    aux["phi_gc"] = load_phi_gc(lon, lat, ls, cfg)

    # train lv AR1_sci on residual variability
    targs_res_lv = {"tas": res_lv_s}
    params_lv = train_lv(
        {}, targs_res_lv, esm, cfg, save_params=True, aux=aux, params_lv=params_lv
    )

    # create full lv emulations
    preds_lv = {"gvtas": emus_gv_T}  # predictors_list
    emus_lv = create_emus_lv(params_lv, preds_lv, cfg, save_emus=True)

    # create full emulations
    print(esm, "Merge the local trends and the local variability.")
    emus_l = create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=True)
