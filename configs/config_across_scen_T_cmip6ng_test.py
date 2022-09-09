"""
Configuration file for tests

"""
import os.path

# path to mesmer root directory can be found in a slightly sneaky way like this
MESMER_ROOT = os.path.join(os.path.dirname(__file__), "..")

# using os.path makes the paths platform-portable i.e. this will
# still work on windows (hard-coding "/" as file separators does not
# work on windows)
TEST_DATA_ROOT = os.path.join(
    MESMER_ROOT, "tests", "test-data", "calibrate-coarse-grid"
)

# ---------------------------------------------------------------------------------

# directories:

# cmip-ng
gen = 6  # generation
dir_cmipng = os.path.join(TEST_DATA_ROOT, f"cmip{gen}-ng")

# observations
dir_obs = os.path.join(TEST_DATA_ROOT, "observations")

# auxiliary data
dir_aux = "auxillary"

# mesmer
dir_mesmer_params = os.path.join(MESMER_ROOT, "calibrated_parameters")
dir_mesmer_emus = os.path.join(MESMER_ROOT, "emulations")


# configs that can be set for every run:

# esms
esms = ["IPSL-CM6A-LR"]


targs = ["tas"]  # emulated variables

reg_type = "srex"

ref = {}
ref["type"] = "individ"  # alternatives: "first","all"
ref["start"] = "1850"  # first included year
ref["end"] = "1900"  # last included year

threshold_land = 1 / 3

wgt_scen_tr_eq = True  # if True weigh each scenario equally (ie less weight to individ runs of scens with more ic members)

nr_emus_v = 5  # tmp made smaller for testing purposes. Normally larger.
scen_seed_offset_v = 0  # 0 meaning same emulations drawn for each scen, if put a number will have different ones for each scen
max_iter_cv = 15  # max. nr of iterations in cross validation, small for testing purpose

# predictors (for global module)
preds = {}
preds["tas"] = {}  # predictors for the target variable tas
preds["hfds"] = {}
preds["tas"]["gt"] = ["saod"]
preds["hfds"]["gt"] = []
preds["tas"]["gv"] = []
preds["tas"]["g_all"] = preds["tas"]["gt"] + preds["tas"]["gv"]

# methods (for all modules)
methods = {}
methods["tas"] = {}  # methods for the target variable tas
methods["hfds"] = {}
methods["tas"]["gt"] = "LOWESS_OLSVOLC"  # global trend emulation method
methods["hfds"]["gt"] = "LOWESS"
methods["tas"]["gv"] = "AR"  # global variability emulation method
methods["tas"]["lt"] = "OLS"  # local trends emulation method
method_lt_each_gp_sep = True  # method local trends applied to each gp separately
methods["tas"]["lv"] = "OLS_AR1_sci"  # local variability emulation method

# ---------------------------------------------------------------------------------

# configs that should remain untouched:

# full list of esms (to have unique seed for each esm no matter how)
all_esms = [
    # "ACCESS-CM2",
    # "ACCESS-ESM1-5",
    # "AWI-CM-1-1-MR",
    # "CAMS-CSM1-0",
    # "CanESM5",
    # "CESM2",
    # "CESM2-WACCM",
    # "CIESM",
    # "CMCC-CM2-SR5",
    # "CNRM-CM6-1",
    # "CNRM-CM6-1-HR",
    # "CNRM-ESM2-1",
    # "E3SM-1-1",
    # "EC-Earth3",
    # "EC-Earth3-Veg",
    # "EC-Earth3-Veg-LR",
    # "FGOALS-f3-L",
    # "FGOALS-g3",
    # "FIO-ESM-2-0",
    # "GFDL-CM4",
    # "GFDL-ESM4",
    # "GISS-E2-1-G",
    # "HadGEM3-GC31-LL",
    # "HadGEM3-GC31-MM",
    "IPSL-CM6A-LR",
    # "MCM-UA-1-0",
    # "MPI-ESM1-2-HR",
    # "MPI-ESM1-2-LR",
    # "MRI-ESM2-0",
    # "NESM3",
    # "NorESM2-LM",
    # "NorESM2-MM",
    # "UKESM1-0-LL",
]

# full list of scenarios that could be considered
scenarios = [
    # "h-ssp585",
    # "h-ssp370",
    # "h-ssp460",
    # "h-ssp245",
    # "h-ssp534-over",
    # "h-ssp434",
    "h-ssp126",
    # "h-ssp119",
]

if scen_seed_offset_v == 0:
    scenarios_emus_v = ["all"]
else:
    scenarios_emus_v = scenarios

nr_emus = {}
nr_ts_emus_v = {}
seed = {}
i = 0
for esm in all_esms:
    seed[esm] = {}
    j = 0
    for scen in scenarios_emus_v:
        seed[esm][scen] = {}
        seed[esm][scen]["gv"] = i + j * scen_seed_offset_v
        seed[esm][scen]["lv"] = i + j * scen_seed_offset_v + 1000000
        j += 1
    i += 1


# ---------------------------------------------------------------------------------

# information about loaded data:

# cmip-ng
# Data downloaded from ESGF (https://esgf-node.llnl.gov/projects/esgf-llnl/) and pre-processed according to Brunner et al. 2020 (https://doi.org/10.5281/zenodo.3734128)
# assumes folder structure / file name as in cmip-ng archives at ETHZ -> see mesmer.io.load_cmipng.file_finder_cmipng() for details
# - global mean stratospheric AOD, monthly, 1850-"2020" (0 after 2012), downloaded from KNMI climate explorer in August 2020, no pre-processing
