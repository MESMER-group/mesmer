"""
Configuration file for cmip6-ng, tas, hist + all ssps

"""

import os.path


class ConfigMesmerX:
    """
    This class defines the full configuration of MESMER.

    Inputs:
     - paths: information on what paths to use.
        If nothing is provided in 'paths', default is to assume paths for tests.
        If any known keyword is provided in 'paths', they will be used instead of default.
        Unknown keywords in 'paths' cause an error.
     - gen: generation of CMIP data (default: 6). If some paths are provided in 'paths', it MUST be consistent with 'gen'.
     - esms: list of the ESMs used. The default is 'all', BUT if no paths are provided, it is assumed that only data for tests are used, then only ["IPSL-CM6A-LR"].
    """

    def __init__(self, paths={}, gen=6, esms="all"):

        # preparing some parameters
        self.paths = paths
        self.gen = gen
        self.esms = esms

        # Handling paths & directories
        self.paths_directories()

        # Handling flexible configuration
        self.flex_config()

        # Handling non-flexible configuration
        self.nonflex_config()

        return

    def paths_directories(self):

        # ---------------------------------------------------------------------------------
        # PATHS

        # path to mesmer root directory can be found in a slightly sneaky way
        # like this
        MESMER_ROOT = os.path.join(os.path.dirname(__file__), "..")

        # using os.path makes the paths platform-portable i.e. this will
        # still work on windows (hard-coding "/" as file separators does not
        # work on windows)
        TEST_DATA_ROOT = os.path.join(
            MESMER_ROOT, "tests", "test-data", "calibrate-coarse-grid"
        )

        # ---------------------------------------------------------------------------------
        # DIRECTORIES
        # checking if any is unknown:
        for key_path in self.paths:
            if key_path not in [
                "dir_cmipng",
                "dir_cmip_X",
                "dir_obs",
                "dir_aux",
                "dir_mesmer_params",
                "dir_mesmer_emus",
                "dir_stats",
                "dir_plots",
            ]:
                raise Exception(
                    'Unknown type of directory provided in "paths", please check available options.'
                )

        # cmip-ng
        gen = 6  # generation
        if "dir_cmipng" in self.paths:
            self.dir_cmipng = self.paths["dir_cmipng"]
        else:
            self.dir_cmipng = os.path.join(TEST_DATA_ROOT, "cmip{}-ng/".format(gen))
            # TODO: remove need for trailing "/" here

        # cmip-x: climate extremes
        if "dir_cmip_X" in self.paths:
            self.dir_cmip_X = self.paths["dir_cmip_X"]
        else:
            # For now, no test data is provided for climate extremes.
            self.dir_cmip_X = None

        # observations
        if "dir_obs" in self.paths:
            self.dir_obs = self.paths["dir_obs"]
        else:
            self.dir_obs = os.path.join(TEST_DATA_ROOT, "observations/")
            # TODO: remove need for trailing "/" here

        # auxiliary data
        if "dir_aux" in self.paths:
            self.dir_aux = self.paths["dir_aux"]
        else:
            self.dir_aux = "auxillary/"

        # mesmer params
        if "dir_mesmer_params" in self.paths:
            self.dir_mesmer_params = self.paths["dir_mesmer_params"]
        else:
            self.dir_mesmer_params = os.path.join(MESMER_ROOT, "calibrated_parameters/")

        # mesmer emus
        if "dir_mesmer_emus" in self.paths:
            self.dir_mesmer_emus = self.paths["dir_mesmer_emus"]
        else:
            self.dir_mesmer_emus = os.path.join(MESMER_ROOT, "emulations/")

        # emulation statistics
        if "dir_stats" in self.paths:
            self.dir_stats = self.paths["dir_stats"]
        else:
            self.dir_stats = "/net/exo/landclim/yquilcaille/across_scen_T/statistics/"

        # plots
        if "dir_plots" in self.paths:
            self.dir_plots = self.paths["dir_plots"]
        else:
            self.dir_plots = "/net/exo/landclim/yquilcaille/across_scen_T/plots/"

        # ---------------------------------------------------------------------------------
        return

    def flex_config(self):

        # ---------------------------------------------------------------------------------
        # ESMs

        if self.paths == {}:
            # running in test mode, only using this ESM
            self.esms = ["IPSL-CM6A-LR"]

        elif self.esms == "all":
            self.esms = [
                "ACCESS-CM2",
                "ACCESS-ESM1-5",
                "AWI-CM-1-1-MR",
                "CanESM5",
                "CESM2",
                "CESM2-WACCM",
                "CMCC-CM2-SR5",
                "CNRM-CM6-1",
                "CNRM-CM6-1-HR",
                "CNRM-ESM2-1",
                "E3SM-1-1",
                "FGOALS-f3-L",
                "FGOALS-g3",
                "FIO-ESM-2-0",
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
            # tmp removed (need to investigate stms soon how can get them in too!):
            # -CAMS-CSM1-0 (train_lt did not work: nans?!)
            # -CIESM (sth wrong with GHFDS)
            # -"EC-Earth3" (sth wrong when reading in files, index issue)
            # - "EC-Earth3-Veg" (probably sth wrong with GHFDS)
            # - "EC-Earth3-Veg-LR" (didn't even try. Just assumed same problem)
            # - "GFDL-CM4" (probably sth wrong with GHFDS)
            # - "GFDL-ESM4" (didn't even try. Just assumed same problem)
            # - "GISS-E2-1-G" (sth wrong when reading in files, index issue)
            # Check update on this aspect on slack Yann-Lea

        else:
            pass  # nothing to change, esms is used as provided.

        # ---------------------------------------------------------------------------------
        # Variables, ensembles, regions
        self.targs = ["tas"]  # emulated variables

        # initial-condition ensemble (ic), multiple-scenarios ensemble (ms)
        self.ens_type_tr = "msic"

        self.reg_type = "ar6.land"

        # ---------------------------------------------------------------------------------
        # Time
        self.ref = {}
        self.ref["type"] = "individ"  # alternatives: 'first','all'
        # first included year
        self.ref["start"] = "1850"
        # last included year
        self.ref["end"] = "1900"

        self.time = {}
        # first included year
        self.time["start"] = "1850"
        # last included year #TODO: check if even used anywhere??
        self.time["end"] = "2100"

        # ---------------------------------------------------------------------------------
        # Parameters
        self.threshold_land = 1 / 3

        self.wgt_scen_tr_eq = True
        # if True weigh each scenario equally (ie less weight to individ runs of scens with more ic members)

        self.nr_emus_v = 1000
        # tmp made smaller for testing purposes. Normally 6000.

        self.scen_seed_offset_v = 0
        # 0 meaning same emulations drawn for each scen, if put a number will have different ones for each scen

        # max. nr of iterations in cross validation, will increase later
        self.max_iter_cv = 15

        # ---------------------------------------------------------------------------------
        # predictors (for global module)
        self.preds = {}
        self.preds["tas"] = {}
        self.preds["hfds"] = {}
        self.preds["pr"] = {}
        self.preds["tas"]["gt"] = ["saod"]
        self.preds["hfds"]["gt"] = []
        self.preds["pr"]["gt"] = []
        self.preds["tas"]["gv"] = []
        self.preds["tas"]["g_all"] = self.preds["tas"]["gt"] + self.preds["tas"]["gv"]

        # ---------------------------------------------------------------------------------
        # methods (for all modules)
        self.methods = {}

        # method local trends applied to each gp separately
        self.method_lt_each_gp_sep = True

        # tas
        # methods for the target variable tas
        self.methods["tas"] = {}
        # global trend emulation method
        self.methods["tas"]["gt"] = "LOWESS_OLSVOLC"
        # local trends emulation method
        self.methods["tas"]["lt"] = "OLS"
        # global variability emulation method
        self.methods["tas"]["gv"] = "AR"
        # local variability emulation method
        self.methods["tas"]["lv"] = "OLS_AR1_sci"

        # hfds
        self.methods["hfds"] = {}
        self.methods["hfds"]["gt"] = "LOWESS"

        # pr
        self.methods["pr"] = {}
        self.methods["pr"]["gt"] = "LOWESS"

        # l_distrib ==> local variability distribution
        # lv ==> local variability emulation method

        # txx
        self.methods["txx"] = {}
        self.methods["txx"]["l_distrib"] = "GEV"
        self.methods["txx"]["lv"] = "AR1_sci"

        # mrso
        self.methods["mrso"] = {}
        self.methods["mrso"]["l_distrib"] = "gaussian"
        self.methods["mrso"]["lv"] = "AR1_sci"

        # mrsomean
        self.methods["mrsomean"] = {}
        self.methods["mrsomean"]["l_distrib"] = "gaussian"
        self.methods["mrsomean"]["lv"] = "AR1_sci"

        # mrso_minmon
        self.methods["mrso_minmon"] = {}
        self.methods["mrso_minmon"]["l_distrib"] = "gaussian"
        self.methods["mrso_minmon"]["lv"] = "AR1_sci"

        # fwixx
        self.methods["fwixx"] = {}
        self.methods["fwixx"]["l_distrib"] = "GEV"
        self.methods["fwixx"]["lv"] = "AR1_sci"

        # fwisa
        self.methods["fwisa"] = {}
        self.methods["fwisa"]["l_distrib"] = "gaussian"
        # GEV | gaussian
        self.methods["fwisa"]["lv"] = "AR1_sci"

        # fwixd
        self.methods["fwixd"] = {}
        self.methods["fwixd"]["l_distrib"] = "poisson"
        self.methods["fwixd"]["lv"] = "AR1_sci"

        # fwils
        self.methods["fwils"] = {}
        self.methods["fwils"]["l_distrib"] = "poisson"
        self.methods["fwils"]["lv"] = "AR1_sci"

        return

    def nonflex_config(self):

        # ---------------------------------------------------------------------------------
        # list of scenarios that could be considered. Right now, complying to previous scripts of configurations, passing scenarios as set configuration, but could be passed in flexible configurations.
        if self.paths == {}:
            # running in test mode, only using this ESM
            self.scenarios = ["h-ssp126"]

        else:
            self.scenarios = [
                "h-ssp585",
                "h-ssp370",
                "h-ssp460",
                "h-ssp245",
                "h-ssp534-over",
                "h-ssp434",
                "h-ssp126",
                "h-ssp119",
            ]

        # ---------------------------------------------------------------------------------
        # Emulations of scenarios and seeds
        if self.scen_seed_offset_v == 0:
            self.scenarios_emus_v = ["all"]
        else:
            self.scenarios_emus_v = self.scenarios

        self.nr_emus = {}
        self.nr_ts_emus_v = {}
        self.seed = {}
        i = 0
        for esm in self.esms:
            self.seed[esm] = {}
            j = 0
            for scen in self.scenarios_emus_v:
                self.seed[esm][scen] = {}
                self.seed[esm][scen]["gv"] = i + j * self.scen_seed_offset_v
                self.seed[esm][scen]["lv"] = i + j * self.scen_seed_offset_v + 1000000
                j += 1
            i += 1

        return


# ---------------------------------------------------------------------------------

# information about loaded data:

# cmip-ng
# Data downloaded from ESGF (https://esgf-node.llnl.gov/projects/esgf-llnl/) and pre-processed according to Brunner et al. 2020 (https://doi.org/10.5281/zenodo.3734128)
# assumes folder structure / file name as in cmip-ng archives at ETHZ -> see mesmer.io.load_cmipng.file_finder_cmipng() for details
# - global mean stratospheric AOD, monthly, 1850-"2020" (0 after 2012), downloaded from KNMI climate explorer in August 2020, no pre-processing
