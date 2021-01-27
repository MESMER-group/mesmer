"""
Configuration file for cmip5-ng, tas, rcp85, for the emulator attribution paper

"""

## cmip-ng
# Data downloaded from ESGF (https://esgf-node.llnl.gov/projects/esgf-llnl/) and pre-processed according to Brunner et al. 2020 (https://doi.org/10.5281/zenodo.3734128)
# assumes folder structure / file name as in cmip-ng archives at ETHZ -> see mesmer.io.load_cmipng.file_finder_cmipng() for details
gen = 5  # generations
# esms = ["CanESM2", "CNRM-CM5", "GISS-E2-H"]  # Earth System Models (used for code development)

## top5 reg averages:
# v1
# ch4 -> stopped in middle of CMCC-CMS (because already available)
# esms=['ACCESS1-0','ACCESS1-3','bcc-csm1-1-m','bcc-csm1-1','BNU-ESM','CanESM2','CCSM4','CESM1-BGC','CESM1-CAM5','CMCC-CESM','CMCC-CM','CMCC-CMS','CNRM-CM5','CSIRO-Mk3-6-0','EC-EARTH','FGOALS-g2','FIO-ESM','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M']
# cfc
# esms=['GISS-E2-H-CC','GISS-E2-H','GISS-E2-R-CC','GISS-E2-R', 'HadGEM2-AO','HadGEM2-CC','HadGEM2-ES','inmcm4','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR','MIROC5','MIROC-ESM-CHEM','MIROC-ESM','MPI-ESM-LR','MPI-ESM-MR','MRI-CGCM3','MRI-ESM1','NorESM1-ME','NorESM1-M']

# v2
# so4 1
# esms=['IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR','MIROC5','MIROC-ESM-CHEM','MIROC-ESM']
# so4 2
# esms=['EC-EARTH','FGOALS-g2','FIO-ESM','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M']
# ch4 1 -> ok
# esms=['MPI-ESM-LR','MPI-ESM-MR','MRI-CGCM3','MRI-ESM1','NorESM1-ME','NorESM1-M']
# ch4 2 -> ok
# esms=['CESM1-CAM5','CMCC-CESM','CMCC-CM','CMCC-CMS','CNRM-CM5','CSIRO-Mk3-6-0']

# v3 iono because cfc was shut down -> ok
# esms=['HadGEM2-CC','HadGEM2-ES','inmcm4']
# v3 cfc because so4 will be shut down: -> ok
# esms=['GFDL-ESM2G','GFDL-ESM2M']
# esms=['MIROC-ESM-CHEM','MIROC-ESM']


## pmf:
# ch4: -> ok
# esms=['ACCESS1-0','ACCESS1-3','bcc-csm1-1-m','bcc-csm1-1','BNU-ESM','CanESM2','CCSM4','CESM1-BGC','CESM1-CAM5','CMCC-CESM','CMCC-CM','CMCC-CMS']
# so4:
# esms=['CNRM-CM5','CSIRO-Mk3-6-0','EC-EARTH','FGOALS-g2','FIO-ESM','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','GISS-E2-H-CC','GISS-E2-H','GISS-E2-R-CC','GISS-E2-R']
# cfc:
# esms=['HadGEM2-AO','HadGEM2-CC','HadGEM2-ES','inmcm4','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR','MIROC5','MIROC-ESM-CHEM','MIROC-ESM']
# iono:
# esms=['MPI-ESM-LR','MPI-ESM-MR','MRI-CGCM3','MRI-ESM1','NorESM1-ME','NorESM1-M']
# iono again because Urs shuts down cfc: -> ok
# esms=['IPSL-CM5B-LR','MIROC5','MIROC-ESM-CHEM','MIROC-ESM']
# a few on cfc because know that so4 will be shut down (+ not finish anyways) -> ok
# esms = ['GISS-E2-H-CC','GISS-E2-H','GISS-E2-R-CC','GISS-E2-R']
# so4 again -> ok
# esms=['GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M']

# subgroup that has pmf for pmf test:
# esms=['ACCESS1-0','ACCESS1-3','bcc-csm1-1-m','bcc-csm1-1','BNU-ESM','CanESM2','CCSM4','CESM1-BGC','CNRM-CM5','CSIRO-Mk3-6-0','EC-EARTH','FGOALS-g2','FIO-ESM','HadGEM2-AO','HadGEM2-CC','HadGEM2-ES','inmcm4','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR','MIROC5','MPI-ESM-LR','MPI-ESM-MR','MRI-CGCM3','MRI-ESM1','NorESM1-ME','NorESM1-M']

# for actual paper:
esms = [
    "ACCESS1-0",
    "ACCESS1-3",
    "bcc-csm1-1-m",
    "bcc-csm1-1",
    "BNU-ESM",
    "CanESM2",
    "CCSM4",
    "CESM1-BGC",
    "CESM1-CAM5",
    "CMCC-CESM",
    "CMCC-CM",
    "CMCC-CMS",
    "CNRM-CM5",
    "CSIRO-Mk3-6-0",
    "EC-EARTH",
    "FGOALS-g2",
    "FIO-ESM",
    "GFDL-CM3",
    "GFDL-ESM2G",
    "GFDL-ESM2M",
    "GISS-E2-H-CC",
    "GISS-E2-H",
    "GISS-E2-R-CC",
    "GISS-E2-R",
    "HadGEM2-AO",
    "HadGEM2-CC",
    "HadGEM2-ES",
    "inmcm4",
    "IPSL-CM5A-LR",
    "IPSL-CM5A-MR",
    "IPSL-CM5B-LR",
    "MIROC5",
    "MIROC-ESM-CHEM",
    "MIROC-ESM",
    "MPI-ESM-LR",
    "MPI-ESM-MR",
    "MRI-CGCM3",
    "MRI-ESM1",
    "NorESM1-ME",
    "NorESM1-M",
]

# to have unique seed for each esm no matter which ones I currently emulate
all_esms = [
    "ACCESS1-0",
    "ACCESS1-3",
    "bcc-csm1-1-m",
    "bcc-csm1-1",
    "BNU-ESM",
    "CanESM2",
    "CCSM4",
    "CESM1-BGC",
    "CESM1-CAM5",
    "CMCC-CESM",
    "CMCC-CM",
    "CMCC-CMS",
    "CNRM-CM5",
    "CSIRO-Mk3-6-0",
    "EC-EARTH",
    "FGOALS-g2",
    "FIO-ESM",
    "GFDL-CM3",
    "GFDL-ESM2G",
    "GFDL-ESM2M",
    "GISS-E2-H-CC",
    "GISS-E2-H",
    "GISS-E2-R-CC",
    "GISS-E2-R",
    "HadGEM2-AO",
    "HadGEM2-CC",
    "HadGEM2-ES",
    "inmcm4",
    "IPSL-CM5A-LR",
    "IPSL-CM5A-MR",
    "IPSL-CM5B-LR",
    "MIROC5",
    "MIROC-ESM-CHEM",
    "MIROC-ESM",
    "MPI-ESM-LR",
    "MPI-ESM-MR",
    "MRI-CGCM3",
    "MRI-ESM1",
    "NorESM1-ME",
    "NorESM1-M",
]


targs = ["tas"]  # emulated variables
ens_type_tr = "ic"  # initial-condition ensemble
scenarios_tr = ["rcp85"]  # scenarios trained on
scenarios_emus = [
    "all"
]  # scenarios emulated <- for this study combine lt params with MAGICC for NDC (-top5) pathways
scen_name_tr = "hist_" + "_".join(scenarios_tr)
hist_tr = True  # if historical part of run is included in training (not yet implemented for False. Would need to write loading fct() accordingly)
wgt_scen_tr_eq= True # if True weigh each scenario equally (ie less weight to individ runs of scens with more ic members)
scen_name_emus = "all"
reg_type = "countries"
ref = {}
ref["type"] = "individ"  # alternatives: 'first','all'
ref["start"] = "1870"  # first included year
ref["end"] = "1900"  # last included year
time = {}
time["start"] = "1870"  # first included year
time["end"] = "2100"  # last included year
threshold_land = 1 / 3
dir_cmipng = "/net/atmos/data/cmip" + str(gen) + "-ng/"

## observations
dir_obs = "/net/cfc/landclim1/beuschl/mesmer/data/observations/"

# - global mean stratospheric AOD, monthly, 1850-"2020" (0 after 2012), downloaded from KNMI climate explorer in August 2020, no pre-processing
# will probably add obs (Cowtan + Way) / (BEST) in here too (preferably land only as well).

## auxiliary data
dir_aux = "/net/cfc/landclim1/beuschl/mesmer/data/auxiliary/"

## MESMER
dir_mesmer_params = (
    "/net/cfc/landclim1/beuschl/attrib_emu_T/mesmer/calibrated_parameters/"
)
dir_mesmer_emus = "/net/cfc/landclim1/beuschl/attrib_emu_T/mesmer/emulations/"

## emulation statistics
dir_stats = "/net/cfc/landclim1/beuschl/attrib_emu_T/statistics/"

nr_emus = {}
nr_emus[
    "all"
] = 6000  # per ESM and scen (to be seen if feasible or needs to be reduced)
nr_ts_emus_v = {}
nr_ts_emus_v["all"] = 181  # 1850-2030
seed = {}
scen_seed_offset = 0  # 0 meaning same emulations drawn for each scen, if put a number will have different ones for each scen
i = 0
for esm in all_esms:
    seed[esm] = {}
    j = 0
    for scen in scenarios_emus:
        seed[esm][scen] = {}
        seed[esm][scen]["gv"] = i + j * scen_seed_offset
        seed[esm][scen]["lv"] = i + j * scen_seed_offset + 1000000
        j += 1
    i += 1

# predictors
preds = {}
preds["tas"] = {}  # predictors for the target variable tas
preds["tas"]["gt"] = ["saod"]
preds["tas"]["gv"] = []
preds["tas"]["g_all"] = preds["tas"]["gt"] + preds["tas"]["gv"]
preds["tas"]["lt"] = ["gt"]
preds["tas"]["lv"] = ["gv"]
preds["tas"]["l_all"] = preds["tas"]["lt"] + preds["tas"]["lv"]

# methods
methods = {}
methods["tas"] = {}  # methods for the target variable tas
methods["tas"][
    "gt"
] = "LOWESS_OLS"  # "LOWESS_OLS_" + "_".join(preds_g["tas"]["gt"])  # global trend emulation method
methods["tas"][
    "gv"
] = "AR"  # "AR" + "_".join(preds_g["tas"]["gv"])  # global variability emulation method
methods["tas"]["lt"] = "OLS"  # "OLS_" + "_".join(preds_l["tas"]["lt"])
method_lt_each_gp_sep = True  # method local trends applied to each gp separately
methods["tas"][
    "lv"
] = "OLS_AR1_sci"  # ("OLS_" + "_".join(preds_l["tas"]["lv"]) + "_AR1_sci")  # local variability emulation OLS to global var and  AR(1) process with spatially correlated innovations


# plots
dir_plots = "/net/cfc/landclim1/beuschl/attrib_emu_T/plots/"
