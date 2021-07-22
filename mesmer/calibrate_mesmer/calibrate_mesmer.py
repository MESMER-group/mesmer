"""
Functions to calibrate all modules of MESMER
"""
import logging
import os
import warnings

from ..create_emulations import create_emus_gt, create_emus_lt, create_emus_lv
from ..io import load_cmipng, load_phi_gc, load_regs_ls_wgt_lon_lat, save_mesmer_bundle
from ..utils import convert_dict_to_arr, extract_land, separate_hist_future
from .train_gt import train_gt
from .train_gv import train_gv
from .train_lt import train_lt
from .train_lv import train_lv

LOGGER = logging.getLogger(__name__)


class _Config:
    """Workaround to mock the ``cfg`` interface used elsewhere"""

    def __init__(
        self,
        esms,
        scenarios,
        cmip_generation,
        cmip_data_root_dir,
        observations_root_dir,
        auxiliary_data_dir,
        reference_period_type,
        reference_period_start_year,
        reference_period_end_year,
        tas_global_trend_method,
        hfds_global_trend_method,
        tas_global_variability_method,
        tas_local_trend_method,
        tas_local_variability_method,
        method_lt_each_gp_sep,
        nr_emus_v,  # TODO: remove when we remove the emulation part
        seeds,  # TODO: remove when we remove the emulation part
        weight_scenarios_equally,
        threshold_land,
        cross_validation_max_iterations,
    ):
        self.esms = esms
        self.scenarios = scenarios
        self.gen = cmip_generation
        # TODO: remove need for trailing seperator eventually
        self.dir_cmipng = f"{cmip_data_root_dir}{os.sep}"
        self.dir_obs = f"{observations_root_dir}{os.sep}"
        self.dir_aux = f"{auxiliary_data_dir}{os.sep}"
        self.ref = {
            "type": reference_period_type,
            "start": reference_period_start_year,
            "end": reference_period_end_year,
        }
        self.methods = {
            "tas": {
                "gt": tas_global_trend_method,
                "gv": tas_global_variability_method,
                "lt": tas_local_trend_method,
                "lv": tas_local_variability_method,
            },
            "hfds": {"gt": hfds_global_trend_method},
        }
        self.method_lt_each_gp_sep = method_lt_each_gp_sep

        # Essentially metadata about predictors used. Maybe used for naming etc.
        # TODO: work out where and how this is used in future so it can be defined
        # more precisely.
        self.preds = {
            "tas": {
                "gt": ["saod"],
                "gv": [],
            },
            "hfds": {"gt": []},
        }
        self.preds["tas"]["g_all"] = self.preds["tas"]["gt"] + self.preds["tas"]["gv"]

        self.nr_emus_v = nr_emus_v
        self.seed = seeds
        self.wgt_scen_tr_eq = weight_scenarios_equally
        self.threshold_land = threshold_land
        self.max_iter_cv = cross_validation_max_iterations


# TODO: remove draw realisations functionality
def _calibrate_and_draw_realisations(
    esms,
    scenarios_to_train,
    target_variable,
    reg_type,
    threshold_land,
    output_file,
    scen_seed_offset_v,  # TODO: remove when we remove the emulation part
    cmip_data_root_dir,
    observations_root_dir,
    auxiliary_data_dir,
    cmip_generation=6,
    reference_period_type="individ",
    reference_period_start_year="1850",
    reference_period_end_year="1900",
    tas_global_trend_method="LOWESS_OLSVOLC",
    hfds_global_trend_method="LOWESS",
    tas_global_variability_method="AR",
    tas_local_trend_method="OLS",
    tas_local_variability_method="OLS_AR1_sci",
    # specify if the local trends method is applied to each grid point separately.
    # Currently it must be set to True
    method_lt_each_gp_sep=True,
    nr_emus_v=100,  # TODO: remove when we remove the emulation part
    weight_scenarios_equally=True,
    cross_validation_max_iterations=30,
):
    """calibrate mesmer - additional predictors configuration. used for end-to-end test"""

    tas_g_dict = {}  # tas with global coverage
    GSAT_dict = {}  # global mean tas
    GHFDS_dict = {}  # global mean hfds (needed as predictor)
    tas_g = {}
    GSAT = {}
    GHFDS = {}
    time = {}

    # TODO: decide if we want this functionality and, if we do, test what
    #       happens if scen_seed_offset_v != 0
    if scen_seed_offset_v == 0:
        scenarios_emus_v = ["all"]
    else:
        scenarios_emus_v = scenarios_to_train

    seeds = {}
    i = 0
    for esm in esms:
        seeds[esm] = {}
        j = 0
        for scen in scenarios_emus_v:
            seeds[esm][scen] = {}
            seeds[esm][scen]["gv"] = i + j * scen_seed_offset_v
            seeds[esm][scen]["lv"] = i + j * scen_seed_offset_v + 1000000
            j += 1
        i += 1

    cfg = _Config(
        esms,
        scenarios_to_train,
        cmip_generation,
        cmip_data_root_dir,
        observations_root_dir,
        auxiliary_data_dir,
        reference_period_type,
        reference_period_start_year,
        reference_period_end_year,
        tas_global_trend_method,
        hfds_global_trend_method,
        tas_global_variability_method,
        tas_local_trend_method,
        tas_local_variability_method,
        method_lt_each_gp_sep,
        nr_emus_v,
        seeds,
        weight_scenarios_equally,
        threshold_land,
        cross_validation_max_iterations,
    )

    for esm in esms:
        LOGGER.info("Loading data for %s", esm)
        tas_g_dict[esm] = {}
        GSAT_dict[esm] = {}
        GHFDS_dict[esm] = {}
        time[esm] = {}

        for scen in scenarios_to_train:
            # TODO: rename tas_g_tmp to target_variable_g_tmp or simply
            #       hard-code tas as always being the target variable
            tas_g_tmp, GSAT_tmp, lon_tmp, lat_tmp, time_tmp = load_cmipng(
                target_variable, esm, scen, cfg
            )

            if tas_g_tmp is None:
                # should this be an error?
                warnings.warn(f"Scenario {scen} does not exist for tas for ESM {esm}")
            else:  # if scen exists: save fields + load hfds fields for it too
                (
                    tas_g_dict[esm][scen],
                    GSAT_dict[esm][scen],
                    lon,
                    lat,
                    time[esm][scen],
                ) = (
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
    reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(reg_type, lon, lat)

    # extract land
    tas, reg_dict, ls = extract_land(
        tas_g, reg_dict, wgt_g, ls, threshold_land=threshold_land
    )

    for esm in esms:
        LOGGER.info("Calibrating %s", esm)

        LOGGER.info("Calibrating global trend module")
        # TODO: `target_variable` is used here but not elsewhere (where tas is
        #       basically hard-coded)
        params_gt_T = train_gt(
            GSAT[esm], target_variable, esm, time[esm], cfg, save_params=False
        )
        # TODO: remove hard-coded hfds
        params_gt_hfds = train_gt(
            GHFDS[esm], "hfds", esm, time[esm], cfg, save_params=False
        )

        # From params_gt_T, extract the global-trend so that the global
        # variability, local trends, and local variability modules can be
        # trained.
        # In this case we're not actually creating emulations
        LOGGER.info("Creating global-trend emulations")
        preds_gt = {"time": time[esm]}

        # TODO: remove use of emus_gt from this script.
        emus_gt_T = create_emus_gt(
            params_gt_T, preds_gt, cfg, concat_h_f=True, save_emus=False
        )
        gt_T_s = create_emus_gt(
            params_gt_T, preds_gt, cfg, concat_h_f=False, save_emus=False
        )

        LOGGER.info(
            "Preparing predictors for global variability, local trends, and "
            "local variability"
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

        LOGGER.info("Calibrating global variability module")
        params_gv_T = train_gv(
            gv_novolc_T_s, target_variable, esm, cfg, save_params=False
        )

        # TODO: remove because time_v is not needed for calibration
        time_v = {}
        time_v["all"] = time[esm][scen]

        LOGGER.info("Calibrating local trends module")
        preds = {
            "gttas": gt_T_s,
            "gttas2": gt_T2_s,
            "gthfds": gt_hfds_s,
            "gvtas": gv_novolc_T_s,
        }
        targs = {"tas": tas_s}
        params_lt, params_lv = train_lt(preds, targs, esm, cfg, save_params=False)

        # Create forced local warming samples used for training the local variability
        # module. Samples are cheap to create so not an issue to have here.
        LOGGER.info("Creating local trends emulations")
        preds_lt = {"gttas": gt_T_s, "gttas2": gt_T2_s, "gthfds": gt_hfds_s}
        lt_s = create_emus_lt(
            params_lt, preds_lt, cfg, concat_h_f=False, save_emus=False
        )

        LOGGER.info("Calibrating local variability module")
        # derive variability part induced by gv
        preds_lv = {"gvtas": gv_novolc_T_s}  # predictors_list

        # Create local variability due to global variability warming samples
        # used for training the local variability module. Samples are cheap to create so not an issue to have here.
        lv_gv_s = create_emus_lv(
            params_lv, preds_lv, cfg, save_emus=False, submethod="OLS"
        )

        # tas essentially hard-coded here too
        LOGGER.debug(
            "Calculating residual variability i.e. what remains of tas once "
            "lt + gv removed"
        )
        res_lv_s = {}  # residual local variability
        for scen in tas_s.keys():
            res_lv_s[scen] = tas_s[scen] - lt_s[scen]["tas"] - lv_gv_s[scen]["tas"]

        LOGGER.debug("Loading auxiliary files")
        aux = {}
        aux["phi_gc"] = load_phi_gc(
            lon, lat, ls, cfg, L_start=1750, L_end=2000, L_interval=250
        )  # better results with default values L, but like this much faster + less space needed

        LOGGER.debug("Finalising training of local variability module on derived data")
        targs_res_lv = {"tas": res_lv_s}
        params_lv = train_lv(
            {}, targs_res_lv, esm, cfg, save_params=False, aux=aux, params_lv=params_lv
        )

        save_mesmer_bundle(
            output_file,
            params_lt,
            params_lv,
            params_gv_T,
            land_fractions=ls["grid_l_m"],
            lat=lat["c"],
            lon=lon["c"],
        )
