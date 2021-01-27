# add pathway to folders 1 level higher (i.e., to mesmer and configs)
import sys

# load in configurations used in this script
import configs.config_attrib_emu_T_obs as cfg

# import MESMER tools
from mesmer.create_emulations import create_emus_gv_T, create_emus_lv
from mesmer.io import load_mesmer_output

sys.path.append("../")


# create new variability emulations
targ_names = ["tas"]
for esm in cfg.esms:
    params_gv_T = load_mesmer_output("params_gv", targ_names, esm, cfg, scen_type="tr")

    print(esm, "Create global variability emulations.")
    emus_gv_T = create_emus_gv_T(params_gv_T, cfg, save_emus=True)

    params_lv = load_mesmer_output("params_lv", targ_names, esm, cfg, scen_type="tr")

    print(esm, "Create local variability emulations.")
    preds_list = [emus_gv_T]
    emus_lv = create_emus_lv(
        params_lv, preds_list, cfg, scenarios="emus", save_emus=True, submethod=""
    )
