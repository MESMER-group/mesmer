"""
Example configuration file
"""
import os.path

from mesmer.create_emulations import create_seed_dict

# path to mesmer root directory
MESMER_ROOT = os.path.join(os.path.dirname(__file__), "..")

# test data for example
TEST_DATA_ROOT = os.path.join(
    MESMER_ROOT, "tests", "test-data", "calibrate-coarse-grid"
)

# folder for output of example
EXAMPLE_OUTPUT_ROOT = os.path.join(MESMER_ROOT, "examples", "output")

# ---------------------------------------------------------------------------------

# cmip-ng generation
gen = 6

# directories:

# cmip-ng
dir_cmipng = os.path.join(TEST_DATA_ROOT, f"cmip{gen}-ng")

# mesmer output
dir_aux = os.path.join(EXAMPLE_OUTPUT_ROOT, "auxiliary")
dir_mesmer_params = os.path.join(EXAMPLE_OUTPUT_ROOT, "calibrated_parameters")
dir_mesmer_emus = os.path.join(EXAMPLE_OUTPUT_ROOT, "emulations")

# configs that can be set for every run:

# analysed esms
esms = ["IPSL-CM6A-LR"]

# emulated variables
targs = ["tas"]

# reference period
ref = {}

# alternative: "all"
ref["type"] = "individ"
# first included year
ref["start"] = "1850"
# last included year
ref["end"] = "1900"

threshold_land = 1 / 3

# if True weigh each scenario equally (ie less weight to individ runs of scens with more
# members)
wgt_scen_tr_eq = True

# number of created emulations (small number for example)
nr_emus_v = 5

# seed offset for scenarios (0 meaning same emulations drawn for each scen, other
# numbers will have different ones for each scen)
scen_seed_offset_v = 0

# max. nr of iterations in cross validation (small for example)
max_iter_cv = 15

# predictors (for global module)
preds = {}

# predictors for the target variable tas
preds["tas"] = {}
preds["hfds"] = {}
preds["tas"]["gt"] = ["saod"]
preds["hfds"]["gt"] = []
preds["tas"]["gv"] = []
preds["tas"]["g_all"] = preds["tas"]["gt"] + preds["tas"]["gv"]

# methods (for all modules)
methods = {}

# methods for tas
methods["tas"] = {}

# global trend emulation method
methods["tas"]["gt"] = "LOWESS_OLSVOLC"
# global variability emulation method
methods["tas"]["gv"] = "AR"
# local trends emulation method
methods["tas"]["lt"] = "OLS"
# method local trends applied to each gp separately
method_lt_each_gp_sep = True
# local variability emulation method
methods["tas"]["lv"] = "OLS_AR1_sci"

# methods for hfds
methods["hfds"] = {}
methods["hfds"]["gt"] = "LOWESS"


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

seed = create_seed_dict(all_esms, scenarios, scen_seed_offset_v)

# ---------------------------------------------------------------------------------

# information about loaded data:

# cmip-ng
# Data downloaded from ESGF (https://esgf-node.llnl.gov/projects/esgf-llnl/) and pre-
# processed according to Brunner et al. 2020 (https://doi.org/10.5281/zenodo.3734128)
# assumes folder structure / file name as in cmip-ng archives at ETHZ
# -> see mesmer.io.load_cmipng.file_finder_cmipng() for details

# global mean stratospheric AOD
# downloaded from KNMI climate explorer in August 2020, no pre-processing, monthly data,
# 1850-"2020" (0 after 2012)
