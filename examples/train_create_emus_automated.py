# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import sys

sys.path.append("../")

import os.path

import joblib
import xarray as xr

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
from mesmer.utils import convert_dict_to_arr, extract_land, separate_hist_future


def save_mesmer_bundle(bundle_file, params_lt, params_lv, params_gv_T, seeds, land_fractions, lat, lon, time):
    """
    Save all the information required to draw MESMER emulations to disk

    TODO: parameters

    TODO: return info

    TODO: move this function into the mesmer package
    """
    assert land_fractions.shape[0] == lat.shape[0]
    assert land_fractions.shape[1] == lon.shape[0]

    # hopefully right way around
    land_fractions = xr.DataArray(land_fractions, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    mesmer_bundle = {
        "params_lt": params_lt,
        "params_lv": params_lv,
        "params_gv_T": params_gv_T,
        "seeds": seeds,
        "land_fractions": land_fractions,
        "time": time,
    }
    joblib.dump(mesmer_bundle, bundle_file)


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

    for scen in cfg.scenarios:

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

    preds_gt = {"time": time[esm]}
    emus_gt_T = create_emus_gt(
        params_gt_T, preds_gt, cfg, concat_h_f=True, save_emus=True
    )
    gt_T_s = create_emus_gt(
        params_gt_T, preds_gt, cfg, concat_h_f=False, save_emus=False
    )

    print(
        esm,
        "Start preparing predictors for global variability, local trends, and local variability",
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

    print(esm, "Start with global variability module")
    params_gv_T = train_gv(gv_novolc_T_s, targ, esm, cfg, save_params=True)

    time_v = {}
    time_v["all"] = time[esm][scen]
    # remember: scen comes from emus_gt_T.keys() here
    # (= necessary to derive compatible emus_gt & emus_gv)
    preds_gv = {"time": time_v}
    emus_gv_T = create_emus_gv(params_gv_T, preds_gv, cfg, save_emus=True)

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
    aux["phi_gc"] = load_phi_gc(
        lon, lat, ls, cfg, L_start=1750, L_end=2000, L_interval=250
    )  # better results with default values L, but like this much faster + less space needed

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

    save_mesmer_bundle(
        os.path.join("tests", "test-data", "test-mesmer-bundle.pkl"),
        params_lt,
        params_lv,
        params_gv_T,
        seeds=cfg.seed,
        land_fractions=ls["grid_l_m"],
        lat=lat["c"],
        lon=lon["c"],
        time=time_v["all"],
    )

