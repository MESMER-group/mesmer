"""
Configuration file for tests

"""
import os.path

# path to mesmer root directory can be found in a slightly sneaky way
# like this
MESMER_ROOT = os.path.join(os.path.dirname(__file__), "..")

# using os.path makes the paths platform-portable i.e. this will
# still work on windows (hard-coding "/" as file separators does not
# work on windows)
TEST_DATA_ROOT = os.path.join(MESMER_ROOT, "tests", "test-data", "first-run-test")

# pathways to adapt depending on local setup:

# cmip-data
gen = 6  # generation
dir_cmipng = os.path.join(TEST_DATA_ROOT, "cmip{}-ng/".format(gen))  # TODO: remove need for trailing "/" here

# observations
dir_obs = os.path.join(TEST_DATA_ROOT, "observations/")  # TODO: remove need for trailing "/" here

# mesmer
dir_mesmer_params = os.path.join(MESMER_ROOT, "calibrated_parameters/")
dir_mesmer_emus = os.path.join(MESMER_ROOT, "emulations/")


# directories below not needed for parts that we test so far (20210210)
# auxiliary data
# dir_aux = "/net/cfc/landclim1/beuschl/mesmer/data/auxiliary/"

# emulation statistics
# dir_stats = "/net/cfc/landclim1/beuschl/across_scen_T/statistics/"

# plots
# dir_plots = "/net/cfc/landclim1/beuschl/across_scen_T/plots/"


# all other config information

# cmip-ng
# Data downloaded from ESGF (https://esgf-node.llnl.gov/projects/esgf-llnl/) and pre-processed according to Brunner et al. 2020 (https://doi.org/10.5281/zenodo.3734128)
# assumes folder structure / file name as in cmip-ng archives at ETHZ -> see mesmer.io.load_cmipng.file_finder_cmipng() for details

# esms we actually use
esms = ["IPSL-CM6A-LR"]

# all esms that are in the cmip-ng archive to have unique seed for each esm no matter which ones I currently emulate
all_esms = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "CAMS-CSM1-0",
    "CanESM5",
    "CESM2",
    "CESM2-WACCM",
    "CIESM",
    "CMCC-CM2-SR5",
    "CNRM-CM6-1",
    "CNRM-CM6-1-HR",
    "CNRM-ESM2-1",
    "E3SM-1-1",
    "EC-Earth3",
    "EC-Earth3-Veg",
    "EC-Earth3-Veg-LR",
    "FGOALS-f3-L",
    "FGOALS-g3",
    "FIO-ESM-2-0",
    "GFDL-CM4",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "IPSL-CM6A-LR",
    "MCM-UA-1-0",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NESM3",
    "NorESM2-LM",
    "NorESM2-MM",
    "UKESM1-0-LL",
]


targs = ["tas"]  # emulated variables
ens_type_tr = (
    "msic"  # initial-condition ensemble (ic), multiple-scenarios ensemble (ms)
)
scenarios_tr = [
    "h-ssp126",
]  # scenarios trained on, ATTENTION: full potential list. Not all ESMs have all ssps available.
scenarios_emus = [
    "h-ssp126",
]  # scenarios emulated


wgt_scen_tr_eq = True  # if True weigh each scenario equally (ie less weight to individ runs of scens with more ic members)


scen_seed_offset_v = 0  # 0 meaning same emulations drawn for each scen, if put a number will have different ones for each scen
if scen_seed_offset_v == 0:  # Potential TODO: integrate hist in this name too?
    scenarios_emus_v = ["all"]
else:
    scenarios_emus_v = scenarios_emus

reg_type = "srex"
ref = {}
ref["type"] = "individ"  # alternatives: 'first','all'
ref["start"] = "1850"  # first included year
ref["end"] = "1900"  # last included year
time = {}
time["start"] = "1850"  # first included year
time["end"] = "2100"  # last included year
threshold_land = 1 / 3

# observations
# - global mean stratospheric AOD, monthly, 1850-"2020" (0 after 2012), downloaded from KNMI climate explorer in August 2020, no pre-processing
# will probably add obs (Cowtan + Way) / (BEST) in here too (preferably land only as well).


nr_emus = {}
nr_ts_emus_v = {}
seed = {}
i = 0
for esm in all_esms:
    seed[esm] = {}
    nr_emus[esm] = {}
    nr_ts_emus_v[esm] = {}
    j = 0
    for scen in scenarios_emus_v:
        nr_emus[esm][scen] = 6000  # nr of emulation time series
        nr_ts_emus_v[esm][scen] = 251  # nr of emulated time steps
        seed[esm][scen] = {}
        seed[esm][scen]["gv"] = i + j * scen_seed_offset_v
        seed[esm][scen]["lv"] = i + j * scen_seed_offset_v + 1000000
        j += 1
    i += 1

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
