# TODO: write various fcts for the things carried out here. Just tmp script. In .py file to carry out via screen so that does not crash if have internet issues.

# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import sys

import joblib
import numpy as np
from scipy.stats import percentileofscore

# load in configurations used in this script
import configs.config_attrib_emu_T_obs as cfg

# import MESMER tools
from mesmer.io import load_mesmer_output

sys.path.append("../")


# to determine considered time periods
time = np.arange(1850, 2031)
idx_time = {}
idx_time["ref30"] = 0
idx_time["top5_2016"] = np.where(time == 2016)[0][0]
idx_time["top5_1991"] = np.where(time == 1991)[0][0]

# TODO: write fct to rebaseline Tglobs on pre-industrial period (each run on its own) (not in this script because already carried out in create_esm-specific_statistics.ipynb
# TODO: write fct for reading in Tglob from MAGICC (as done below)
dir_data_Tglob = "/net/cfc/landclim1/beuschl/attrib_emu_T/magicc/NDC_minustop5/"
scenarios = ["ref30", "top5_2016", "top5_1991"]

Tglob_gt = joblib.load(
    dir_data_Tglob + "Tglob_ref_1850-1900/Tglob_NDC_minustop5_2016_1991.pkl"
)

for scen in scenarios:
    Tglob_gt[scen] = Tglob_gt[scen].values[:, idx_time[scen] :]


targ_names = ["tas"]

for esm in cfg.esms:

    print("Start with ESM", esm)

    print("Load local trends parameters")
    params_lt = load_mesmer_output("params_lt", targ_names, esm, cfg, scen_type="tr")

    print("Create local trend emulations")
    nr_gps = params_lt["intercept"]["tas"].shape[0]

    # my create_emus_lt() fct did not work here. Should write fct that does work with input like this as this is pretty much
    # the default way I want to combine MAGICC data with MESMER data

    emus_lt = {}
    for scen in scenarios:
        nr_Tglob_gts, nr_ts = Tglob_gt[scen].shape
        emus_lt[scen] = np.zeros([nr_Tglob_gts, nr_ts, nr_gps])
        for nr_gp in np.arange(nr_gps):
            emus_lt[scen][:, :, nr_gp] = (
                params_lt["coef_gt"]["tas"][nr_gp] * Tglob_gt[scen]
                + params_lt["intercept"]["tas"][nr_gp]
            )

    print(
        "Derive medians (ie save emus_lt as median due to way emulator is constructed)"
    )
    median = {}  # median

    for scen in scenarios:
        median[scen] = emus_lt[scen]  # by definition: the true median is emus_lt

    joblib.dump(
        median, cfg.dir_stats + "median/median_" + esm + "_NDC_minustop5_2016_1991.pkl"
    )


for esm in cfg.esms:

    print("Start with ESM", esm)

    print("Load in local trend emulations")
    emus_lt = joblib.load(
        cfg.dir_stats + "median/median_" + esm + "_NDC_minustop5_2016_1991.pkl"
    )

    print("Load local variability emulations")
    emus_lv = load_mesmer_output("emus_lv", targ_names, esm, cfg, scen_type="emus")

    print("Derive probability multiplication factors")
    pmf_rare_pi = {}  # probability multiplication factor

    for scen in scenarios:
        nr_Tglob_gts, nr_ts = Tglob_gt[scen].shape
        pmf_rare_pi[scen] = np.zeros([nr_Tglob_gts, nr_ts, nr_gps])

    for run_gt in np.arange(nr_Tglob_gts):
        for scen in scenarios:
            nr_Tglob_gts, nr_ts = Tglob_gt[scen].shape
            emus_l_tmp = (
                emus_lt[scen][run_gt] + emus_lv["all"]["tas"][:, idx_time[scen] :, :]
            )
            if scen == "ref30":
                # compute the magnitude of a 1 in 100 years hot year in pre-industrial time period
                # ASSUMPTION: the first 51 years represent 1851 - 1900 (ie the pre-industrial ref period)
                rare_pi = np.percentile(
                    emus_l_tmp[:, :51, :].reshape(-1, nr_gps), 99, axis=0
                )

            # for each time step and each grid point compute the pmf as 100 - percentileofscore
            # (ie what is the percentile of an event of the same magnitude as the 99th percentile event in pi times)
            # a bit strangely computed here but went through it several time. Convinced the math is right.
            # Will just need to find a clear way to write it down.
            for ts in np.arange(nr_ts):
                for gp in np.arange(nr_gps):
                    pmf_rare_pi[scen][run_gt, ts, gp] = 100 - percentileofscore(
                        emus_l_tmp[:, ts, gp], rare_pi[gp]
                    )

        if run_gt % 50 == 0:
            print("Done with pmf for global trend trajectory", run_gt)

        del rare_pi  # to avoid using wrong baseline scenario

    joblib.dump(
        pmf_rare_pi,
        cfg.dir_stats + "pmf/pmf_rare_pi_" + esm + "_NDC_minustop5_2016_1991.pkl",
    )
