"""
mesmer.io.load_mesmer
===================
Functions to load in mesmer output.


Functions:
    load_mesmer_output()

"""


import joblib


def load_mesmer_output(name, targ_names, esm, cfg, scen_type="emus"):

    # assumption: targets which are saved in single mesmer output entity share the applied methods and predictors

    # specify necessary variables from config file
    ens_type_tr = cfg.ens_type_tr

    targ = targ_names[0]  # same methods for all targets
    method_gt = cfg.methods[targ]["gt"]
    method_gv = cfg.methods[targ]["gv"]
    method_lt = cfg.methods[targ]["lt"]
    method_lv = cfg.methods[targ]["lv"]
    preds_gt = cfg.preds[targ]["gt"]
    preds_gv = cfg.preds[targ]["gv"]
    preds_lt = cfg.preds[targ]["lt"]
    preds_lv = cfg.preds[targ]["lv"]

    if "params" in name:
        dir_mesmer = cfg.dir_mesmer_params
    elif "emus" in name:
        dir_mesmer = cfg.dir_mesmer_emus

    if scen_type == "emus":
        # scenarios = cfg.scenarios_emus
        scen_name = cfg.scen_name_emus
    elif scen_type == "tr":
        # scenarios = cfg.scenarios_tr
        scen_name = cfg.scen_name_tr

    if "gt" in name:
        file = (
            "global/global_trend/"
            + name
            + "_"
            + ens_type_tr
            + "_"
            + method_gt
            + "_"
            + "_".join(preds_gt)
            + "_"
            + "_".join(targ_names)
            + "_"
            + esm
            + "_"
            + scen_name
            + ".pkl"
        )
    elif "gv" in name:
        file = (
            "global/global_variability/"
            + name
            + "_"
            + ens_type_tr
            + "_"
            + method_gv
            + "_"
            + "_".join(preds_gv)
            + "_"
            + "_".join(targ_names)
            + "_"
            + esm
            + "_"
            + scen_name
            + ".pkl"
        )

    elif "g" in name:
        file = (
            "global/"
            + name
            + "_"
            + ens_type_tr
            + "_gt_"
            + method_gt
            + "_"
            + "_".join(preds_gt)
            + "_gv_"
            + method_gv
            + "_"
            + "_".join(preds_gv)
            + "_"
            + "_".join(targ_names)
            + "_"
            + esm
            + "_"
            + scen_name
            + ".pkl"
        )

    elif "lt" in name:
        file = (
            "local/local_trends/"
            + name
            + "_"
            + ens_type_tr
            + "_"
            + method_lt
            + "_"
            + "_".join(preds_lt)
            + "_"
            + "_".join(targ_names)
            + "_"
            + esm
            + "_"
            + scen_name
            + ".pkl"
        )

    elif "lv" in name:
        file = (
            "local/local_variability/"
            + name
            + "_"
            + ens_type_tr
            + "_"
            + method_lv
            + "_"
            + "_".join(preds_lv)
            + "_"
            + "_".join(targ_names)
            + "_"
            + esm
            + "_"
            + scen_name
            + ".pkl"
        )

    elif "l" in name:
        file = (
            "local/"
            + name
            + "_"
            + ens_type_tr
            + "_lt_"
            + method_lt
            + "_"
            + "_".join(preds_lt)
            + "_lv_"
            + method_lv
            + "_"
            + "_".join(preds_lv)
            + "_"
            + "_".join(targ_names)
            + "_"
            + esm
            + "_"
            + scen_name
            + ".pkl"
        )

    else:
        print("No such file exists. No file is loaded.")

    mesmer_output = joblib.load(dir_mesmer + file)

    return mesmer_output
