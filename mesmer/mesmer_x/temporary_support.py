# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import sys

sys.path.append("../")

# additional packages for this script
import numpy as np

# load in MESMER scripts for treatment of data
# TODO: write the function test_combination_vars
from mesmer.io import load_cmipng, load_phi_gc, load_regs_ls_wgt_lon_lat
from mesmer.utils import convert_dict_to_arr, extract_land


def load_inputs_MESMERx(cfg, variables, esms):
    targ, pred, sub_pred = variables

    # initiate TEMPORARY dictionaries
    # target with global coverage (dict[esm][scen][run]: array Time x Lat x Lon)
    targ_g_dict = {esm: {} for esm in esms}

    # predictor with global coverage (dict[esm][scen][run]: array Time x Lat x Lon)
    pred_g_dict = {esm: {} for esm in esms}

    # global mean predictor (dict[esm][scen][run]: array Time)
    PRED_dict = {esm: {} for esm in esms}

    # global mean hfds (needed as predictor) (dict[esm][scen][run]: array Time)
    if sub_pred is not None:
        SUB_PRED_dict = {esm: {} for esm in esms}

    # initiate dictionaries
    # time axis (dict[esm][scen]: array Time)
    time = {esm: {} for esm in esms}

    # target with global coverage (dict[esm][scen]: array Run x Time x Lat x Lon)
    targ_g = {}
    # predictor with global coverage (dict[esm][scen]: array Run x Time x Lat x Lon)
    pred_g = {}

    # global mean tas (dict[esm][scen]: array Run x Time x Lat x Lon)
    PRED = {}

    # global mean hfds (dict[esm][scen]: array Run x Time x Lat x Lon)
    if sub_pred is not None:
        SUB_PRED = {}

    for esm in esms:
        print(esm)

        for scen in cfg.scenarios:

            # TODO: checking if this (esm,scen) combination has compatible runs.
            # if sub_pred is not None:
            #     available_runs, _ = test_combination_vars(
            #         [targ, pred, sub_pred], esm, scen, cfg
            #     )
            # else:
            #     available_runs, _ = test_combination_vars([targ, pred], esm, scen, cfg)

            available_runs = ["all"]
            if len(available_runs) > 0:
                targ_g_dict[esm][scen], _, lon, lat, time[esm][scen] = load_cmipng(
                    targ, esm, scen, cfg
                )
                pred_g_dict[esm][scen], PRED_dict[esm][scen], _, _, _ = load_cmipng(
                    pred, esm, scen, cfg
                )
                if sub_pred is not None:
                    _, SUB_PRED_dict[esm][scen], _, _, _ = load_cmipng(
                        sub_pred, esm, scen, cfg
                    )

        # grouping the level [run] of dict[esm][scen][run] into a single array
        targ_g[esm] = convert_dict_to_arr(targ_g_dict[esm])
        pred_g[esm] = convert_dict_to_arr(pred_g_dict[esm])
        PRED[esm] = convert_dict_to_arr(PRED_dict[esm])
        if sub_pred is not None:
            SUB_PRED[esm] = convert_dict_to_arr(SUB_PRED_dict[esm])

    # clean temporary files
    del targ_g_dict, pred_g_dict, PRED_dict
    if sub_pred is not None:
        del SUB_PRED_dict

    # stops here if nothing in there to do:
    if len(PRED) == 0:
        raise Exception("No common runs found.")

    # load in the constant files
    reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(cfg.reg_type, lon, lat)

    # extract land
    land_targ, reg_dict, ls = extract_land(
        targ_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
    )
    land_pred, reg_dict, ls = extract_land(
        pred_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
    )

    # prepare the auxiliary files. better results with default values L, but like this
    # much faster + less space needed
    phi_gc = load_phi_gc(lon, lat, ls, cfg, L_start=1750, L_end=10000, L_interval=250)

    lon_mesh, lat_mesh = np.meshgrid(lon["c"], lat["c"])

    # adding few lines for future regional calculations (used for tests)
    ind = np.where(ls["idx_grid_l"])
    gp2reg = reg_dict["grids"][:, ind[0], ind[1]]  # grid points to regions
    ww_reg = np.nansum((ls["wgt_gp_l"] * gp2reg).T, axis=0)

    # Just checking what ESMs are actually used. Some are removed because not having all
    # drivers
    used_esms = [esm for esm in esms if len(PRED[esm].keys()) > 0]

    # adding few lines for SMA, because some values in NaN  or inf!
    dico_gps_nan = {}
    for esm in used_esms:

        # creating common set:
        to_exclude = set()
        for scen in land_targ[esm].keys():
            # full list of gridpoints that have NaN or inf for this esm
            gps_nan = np.where(
                np.isnan(land_targ[esm][scen]) | np.isinf(land_targ[esm][scen])
            )[2]
            to_exclude = to_exclude.union(set(gps_nan))

        # for each one of these gridpoints, all timeseries along scenarios are excluded.
        for scen in land_targ[esm].keys():
            land_targ[esm][scen][..., list(to_exclude)] = np.nan

    if sub_pred is not None:
        return (
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
        )
    else:
        return (
            time,
            PRED,
            None,
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
        )
