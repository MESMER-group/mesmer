import warnings

# import MESMER tools
from mesmer.calibrate_mesmer import train_gt, train_gv, train_lt, train_lv
from mesmer.create_emulations import (
    create_emus_g,
    create_emus_gv,
    create_emus_l,
    create_emus_lt,
    create_emus_lv,
    gather_gt_data,
)
from mesmer.create_emulations.utils import concatenate_hist_future
from mesmer.io import load_cmipng, load_phi_gc, load_regs_ls_wgt_lon_lat
from mesmer.utils import convert_dict_to_arr, extract_land, separate_hist_future


def main(cfg):

    # specify the target variable
    targ = cfg.targs[0]
    print(f"Target variables: {targ}")

    # load in the ESM runs
    esms = cfg.esms
    print(f"Analysed esms: {esms}")
    print()

    # load in tas with global coverage

    tas_g = {}  # tas with global coverage
    gsat = {}  # global mean tas
    time = {}

    print("Loading data")
    print("============")

    for esm in esms:
        print(f"- {esm}")

        time[esm] = {}

        # temporary dicts to gather data over scenarios
        tas_temp, gsat_temp = {}, {}
        for scen in cfg.scenarios:

            out = load_cmipng(targ, esm, scen, cfg)

            # skip if no data found
            if out[0] is None:
                warnings.warn(f"Scenario {scen} does not exist for tas for ESM {esm}")
                continue

            # unpack data
            tas_temp[scen], gsat_temp[scen], lon, lat, time[esm][scen] = out

        tas_g[esm] = convert_dict_to_arr(tas_temp)
        gsat[esm] = convert_dict_to_arr(gsat_temp)

    # load the constant files
    _, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(lon=lon, lat=lat)

    # extract land
    tas, _, ls = extract_land(
        tas_g, wgt=wgt_g, ls=ls, threshold_land=cfg.threshold_land
    )

    print()

    for esm in esms:
        print(f"{esm}")
        print("=" * len(esm))

        print("- Start with global trend module")

        params_gt_tas = train_gt(gsat[esm], targ, esm, time[esm], cfg, save_params=True)

        preds_gt = {"time": time[esm]}
        gt_tas_s = gather_gt_data(
            params_gt_tas, preds_gt, cfg, concat_h_f=False, save_emus=False
        )
        gt_tas = concatenate_hist_future(gt_tas_s)

        print(
            "- Prepare predictors for global variability, local trends and variability"
        )

        gsat_s, _ = separate_hist_future(gsat[esm], time[esm], cfg)
        gv_novolc_tas_s = {}
        for scen in gt_tas_s.keys():
            gv_novolc_tas_s[scen] = gsat_s[scen] - gt_tas_s[scen]

        tas_s, _ = separate_hist_future(tas[esm], time[esm], cfg)

        print("- Start with global variability module")

        params_gv_tas = train_gv(gv_novolc_tas_s, targ, esm, cfg, save_params=True)

        time_v = {}
        scen = list(gt_tas.keys())[0]
        time_v["all"] = time[esm][scen]
        preds_gv = {"time": time_v}
        emus_gv_tas = create_emus_gv(params_gv_tas, preds_gv, cfg, save_emus=True)

        # create and save full global emulations
        print("- Merge the global trend and the global variability.")
        create_emus_g(
            gt_tas, emus_gv_tas, params_gt_tas, params_gv_tas, cfg, save_emus=True
        )

        print("- Start with local trends module")

        preds = {
            "gttas": gt_tas_s,
            "gvtas": gv_novolc_tas_s,
        }
        targs = {"tas": tas_s}
        params_lt, params_lv = train_lt(preds, targs, esm, cfg, save_params=True)

        preds_lt = {"gttas": gt_tas_s}
        lt_s = create_emus_lt(
            params_lt, preds_lt, cfg, concat_h_f=False, save_emus=True
        )
        emus_lt = create_emus_lt(
            params_lt, preds_lt, cfg, concat_h_f=True, save_emus=True
        )

        print("- Start with local variability module")

        # derive variability part induced by gv
        preds_lv = {"gvtas": gv_novolc_tas_s}  # predictors_list
        lv_gv_s = create_emus_lv(
            params_lv, preds_lv, cfg, save_emus=False, submethod="OLS"
        )

        # derive residual variability i.e. what remains of tas once lt + gv removed
        res_lv_s = {}  # residual local variability
        for scen in tas_s.keys():
            res_lv_s[scen] = tas_s[scen] - lt_s[scen]["tas"] - lv_gv_s[scen]["tas"]

        # load in the auxiliary files
        aux = {}
        aux["phi_gc"] = load_phi_gc(
            lon, lat, ls, cfg, L_start=1500, L_end=2000, L_interval=250
        )

        # train lv AR1_sci on residual variability
        targs_res_lv = {"tas": res_lv_s}
        params_lv = train_lv(
            {}, targs_res_lv, esm, cfg, save_params=True, aux=aux, params_lv=params_lv
        )

        # create full lv emulations
        preds_lv = {"gvtas": emus_gv_tas}  # predictors_list
        emus_lv = create_emus_lv(params_lv, preds_lv, cfg, save_emus=True)

        # create and save full emulations
        print("- Merge the local trends and the local variability.")
        create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=True)


if __name__ == "__main__":

    # load in configurations used in this script
    import config_tas_cmip6ng_example as cfg

    main(cfg)
