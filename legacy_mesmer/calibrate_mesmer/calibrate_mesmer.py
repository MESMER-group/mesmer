"""
Functions to calibrate all modules of MESMER
"""
import logging
import warnings

from mesmer.create_emulations.utils import concatenate_hist_future

from ..create_emulations import create_emus_lt, create_emus_lv, gather_gt_data
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
        *args,
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
        weight_scenarios_equally,
        threshold_land,
        cross_validation_max_iterations,
        save_params=False,
        params_output_dir=None,
        **kwargs,
    ):

        if args:
            raise ValueError("All params are now keyword-only")

        for key in kwargs:
            warnings.warn(f"{key} has been deprecated and has no effect", FutureWarning)

        self.esms = esms
        self.scenarios = scenarios
        self.gen = cmip_generation

        self.dir_cmipng = cmip_data_root_dir
        self.dir_obs = observations_root_dir
        self.dir_aux = auxiliary_data_dir
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

        self.wgt_scen_tr_eq = weight_scenarios_equally
        self.threshold_land = threshold_land
        self.max_iter_cv = cross_validation_max_iterations

        if save_params and not params_output_dir:
            raise ValueError("`dir_mesmer_params` required if `save_params` is True")

        self.save_params = save_params
        self.dir_mesmer_params = params_output_dir


# TODO: remove draw realisations functionality
def _calibrate_and_draw_realisations(
    *args,
    esms,
    scenarios_to_train,
    threshold_land,
    output_file,
    cmip_data_root_dir=None,
    observations_root_dir=None,
    auxiliary_data_dir=None,
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
    weight_scenarios_equally=True,
    cross_validation_max_iterations=30,
    save_params=False,
    params_output_dir=None,
    **kwargs,
):
    """
    calibrate mesmer - additional predictors configuration. used for end-to-end test
    """

    if args:
        raise ValueError("All params are now keyword-only")

    for key in kwargs:
        warnings.warn(f"{key} has been deprecated and has no effect", FutureWarning)

    tas_g = {}  # tas with global coverage
    gsat = {}  # global mean tas
    ghfds = {}  # global mean hfds (needed as predictor)
    time = {}

    cfg = _Config(
        esms=esms,
        scenarios=scenarios_to_train,
        cmip_generation=cmip_generation,
        cmip_data_root_dir=cmip_data_root_dir,
        observations_root_dir=observations_root_dir,
        auxiliary_data_dir=auxiliary_data_dir,
        reference_period_type=reference_period_type,
        reference_period_start_year=reference_period_start_year,
        reference_period_end_year=reference_period_end_year,
        tas_global_trend_method=tas_global_trend_method,
        hfds_global_trend_method=hfds_global_trend_method,
        tas_global_variability_method=tas_global_variability_method,
        tas_local_trend_method=tas_local_trend_method,
        tas_local_variability_method=tas_local_variability_method,
        method_lt_each_gp_sep=method_lt_each_gp_sep,
        weight_scenarios_equally=weight_scenarios_equally,
        threshold_land=threshold_land,
        cross_validation_max_iterations=cross_validation_max_iterations,
        save_params=save_params,
        params_output_dir=params_output_dir,
    )

    for esm in esms:
        LOGGER.info("Loading data for %s", esm)

        time[esm] = {}

        # temporary dicts to gather data over scenarios
        tas_temp, gsat_temp, ghfds_temp = {}, {}, {}
        for scen in scenarios_to_train:
            out = load_cmipng("tas", esm, scen, cfg)

            if out[0] is None:
                warnings.warn(f"Scenario {scen} does not exist for tas for ESM {esm}")
                continue

            # unpack data
            tas_temp[scen], gsat_temp[scen], lon, lat, time[esm][scen] = out

            _, ghfds_temp[scen], _, _, _ = load_cmipng("hfds", esm, scen, cfg)

        tas_g[esm] = convert_dict_to_arr(tas_temp)
        gsat[esm] = convert_dict_to_arr(gsat_temp)
        ghfds[esm] = convert_dict_to_arr(ghfds_temp)

    # load in the constant files
    _, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(lon=lon, lat=lat)

    # extract land
    tas, _, ls = extract_land(tas_g, wgt=wgt_g, ls=ls, threshold_land=threshold_land)

    for esm in esms:
        LOGGER.info("Calibrating %s", esm)

        LOGGER.info("Calibrating global trend module")

        params_gt_tas = train_gt(
            gsat[esm], "tas", esm, time[esm], cfg, save_params=cfg.save_params
        )
        # TODO: remove hard-coded hfds
        params_gt_hfds = train_gt(
            ghfds[esm], "hfds", esm, time[esm], cfg, save_params=cfg.save_params
        )

        # From params_gt_T, extract the global-trend so that the global variability,
        # local trends, and local variability modules can be trained.
        # In this case we're not actually creating emulations
        LOGGER.info("Creating global-trend emulations")
        preds_gt = {"time": time[esm]}

        gt_tas_s = gather_gt_data(
            params_gt_tas, preds_gt, cfg, concat_h_f=False, save_emus=False
        )
        gt_tas = concatenate_hist_future(gt_tas_s)

        LOGGER.info(
            "Prepare predictors for global variability, local trends variability"
        )
        gt_tas2_s = {scen: tas**2 for scen, tas in gt_tas_s.items()}

        gt_hfds_s = gather_gt_data(
            params_gt_hfds, preds_gt, cfg, concat_h_f=False, save_emus=False
        )

        # calculate tas residuals
        gv_novolc_tas = {scen: gsat[esm][scen] - gt_tas[scen] for scen in gt_tas}

        gv_novolc_tas_s, _ = separate_hist_future(gv_novolc_tas, time[esm], cfg)

        tas_s, _ = separate_hist_future(tas[esm], time[esm], cfg)

        LOGGER.info("Calibrating global variability module")
        params_gv_tas = train_gv(
            gv_novolc_tas_s, "tas", esm, cfg, save_params=cfg.save_params
        )

        LOGGER.info("Calibrating local trends module")
        preds = {
            "gttas": gt_tas_s,
            "gttas2": gt_tas2_s,
            "gthfds": gt_hfds_s,
            "gvtas": gv_novolc_tas_s,
        }
        targs = {"tas": tas_s}
        params_lt, params_lv = train_lt(
            preds, targs, esm, cfg, save_params=cfg.save_params
        )

        # Create forced local warming samples used for training the local variability
        # module. Samples are cheap to create so not an issue to have here.
        LOGGER.info("Creating local trends emulations")
        preds_lt = {"gttas": gt_tas_s, "gttas2": gt_tas2_s, "gthfds": gt_hfds_s}
        lt_s = create_emus_lt(
            params_lt, preds_lt, cfg, concat_h_f=False, save_emus=False
        )

        LOGGER.info("Calibrating local variability module")
        # derive variability part induced by gv
        preds_lv = {"gvtas": gv_novolc_tas_s}  # predictors_list

        # Create local variability due to global variability warming samples
        # required to find the local residuals
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
        # better results with default values L, but faster + less space needed
        aux["phi_gc"] = load_phi_gc(
            lon, lat, ls, cfg, L_start=1750, L_end=2000, L_interval=250
        )

        LOGGER.debug("Finalising training of local variability module on derived data")
        targs_res_lv = {"tas": res_lv_s}
        params_lv = train_lv(
            {},
            targs_res_lv,
            esm,
            cfg,
            save_params=cfg.save_params,
            aux=aux,
            params_lv=params_lv,
        )

        save_mesmer_bundle(
            output_file,
            params_lt,
            params_lv,
            params_gv_tas,
            land_fractions=ls["grid_l_m"],
            lat=lat["c"],
            lon=lon["c"],
        )
