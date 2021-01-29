# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import copy
import sys

import joblib
import numpy as np

# load in configurations used in this script
import configs.config_attrib_emu_T_obs as cfg
from mesmer.io import load_cmipng, load_mesmer_output, load_regs_ls_wgt_lon_lat
from mesmer.utils import convert_dict_to_arr, extract_land

sys.path.append("../")


# import MESMER tools


# specify the target variable
targ = cfg.targs[0]
print(targ)

# load in the ESM runs

# load in tas with global coverage
tas_g = {}  # tas with global coverage
GSAT = {}  # global mean tas

for esm in ["CanESM2"]:
    print(esm)
    tas_g[esm] = {}
    GSAT[esm] = {}
    for scen in cfg.scenarios_tr:
        tas_g[esm][scen], GSAT[esm][scen], lon, lat, _ = load_cmipng(
            targ, esm, scen, cfg
        )

# load in the constant files
reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(cfg.reg_type, lon, lat)

# extract land
tas, reg_dict, ls = extract_land(
    tas_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
)


# define the correct time periods
time = np.arange(1850, 2031)
idx_time = {}
idx_time["ref30"] = 0
idx_time["top5_2016"] = np.where(time == 2016)[0][0]
idx_time["top5_1991"] = np.where(time == 1991)[0][0]


# load in Tglob
dir_data_Tglob = "/net/cfc/landclim1/beuschl/attrib_emu_T/magicc/NDC_minustop5/"
scenarios = ["ref30", "top5_2016", "top5_1991"]

Tglob_gt = joblib.load(
    dir_data_Tglob + "Tglob_ref_1850-1900/Tglob_NDC_minustop5_2016_1991.pkl"
)

for scen in scenarios:
    Tglob_gt[scen] = Tglob_gt[scen].values[:, idx_time[scen] :]

# find the top5 indices
eu28_names = [
    "Belgium",
    "Bulgaria",
    "Czechia",
    "Denmark",
    "Germany",
    "Estonia",
    "Ireland",
    "Greece",
    "Spain",
    "France",
    "Croatia",
    "Italy",
    "Cyprus",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Hungary",
    "Netherlands",
    "Austria",
    "Poland",
    "Portugal",
    "Romania",
    "Slovenia",
    "Slovakia",
    "Finland",
    "Sweden",
    "United Kingdom",
]  # Malta (somehow not in the list of countries)
idxs_eu28 = np.zeros(len(eu28_names), dtype=int)
for i in np.arange(len(eu28_names)):
    idxs_eu28[i] = np.int(reg_dict["full"].map_keys(eu28_names[i]))


idx_top5_rest = {}
idx_top5_rest["CHN"] = np.int(reg_dict["full"].map_keys("China"))
idx_top5_rest["USA"] = np.int(reg_dict["full"].map_keys("US"))
idx_top5_rest["IND"] = np.int(reg_dict["full"].map_keys("IND"))
idx_top5_rest["RUS"] = np.int(reg_dict["full"].map_keys("RUS"))


# find the top5 weights
nr_gps = reg_dict["wgt_gps_l"].shape[1]

wgt_gps_l = {}
wgt_gps_l["EU28"] = np.zeros(nr_gps)


for idx_eu28 in idxs_eu28:
    wgt_gps_l["EU28"] += reg_dict["wgt_gps_l"][idx_eu28]


for c in ["CHN", "USA", "IND", "RUS"]:
    wgt_gps_l[c] = reg_dict["wgt_gps_l"][idx_top5_rest[c]]

# top5 names
top5_names = ["CHN", "USA", "EU28", "IND", "RUS"]

# load in the regional features by loading in a single cmip5 model


targ_names = ["tas"]

for esm in cfg.esms:

    print("Start with ESM", esm)

    print("Load in local variability emulations")
    emus_lv = load_mesmer_output("emus_lv", targ_names, esm, cfg, scen_type="emus")

    print("Load local trends parameters")
    params_lt = load_mesmer_output("params_lt", targ_names, esm, cfg, scen_type="tr")

    print("Create local trend emulations")
    nr_emus_lv, _, nr_gps = emus_lv["all"]["tas"].shape

    # my create_emus_lt() fct did not work here. Should write fct that does work with input like this as this is pretty much
    # the default way I want to combine MAGICC data with MESMER data

    emus_reg = {}
    for c in top5_names:
        emus_reg[c] = {}
        for scen in scenarios:
            nr_Tglob_gts, nr_ts = Tglob_gt[scen].shape
            emus_reg[c][scen] = np.zeros([nr_emus_lv * nr_Tglob_gts, nr_ts])

    for scen in scenarios:
        print("start with scenario", scen)
        nr_Tglob_gts, nr_ts = Tglob_gt[scen].shape

        emus_lt_tmp = np.zeros([nr_Tglob_gts, nr_ts, nr_gps])
        for nr_gp in np.arange(nr_gps):
            emus_lt_tmp[:, :, nr_gp] = (
                params_lt["coef_gt"]["tas"][nr_gp] * Tglob_gt[scen]
                + params_lt["intercept"]["tas"][nr_gp]
            )

        i = 0
        for run_gt in np.arange(nr_Tglob_gts):
            emus_l_tmp = (
                emus_lv["all"]["tas"][:, idx_time[scen] :, :] + emus_lt_tmp[run_gt]
            )

            for c in top5_names:
                emus_reg[c][scen][i : i + nr_emus_lv] = np.average(
                    emus_l_tmp, weights=wgt_gps_l[c], axis=2
                )

            i += nr_emus_lv
            if run_gt % 50 == 0:
                print("done with global mean trajectory", run_gt)

    print("save the regional averages")
    for c in top5_names:
        joblib.dump(
            emus_reg[c],
            cfg.dir_stats
            + "reg_avgs/"
            + c
            + "_"
            + esm
            + "_NDC_minustop5_2016_1991.pkl",
        )
