# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/

import numpy as np
import xarray as xr

from .create_emus_gv import create_emus_gv
from .create_emus_lt import create_emus_lt
from .create_emus_lv import create_emus_lv
from .merge_emus import create_emus_l


def make_realisations(
    preds_lt,
    params_lt,
    params_lv,
    params_gv_T,
    time,
    n_realisations,
    seeds,
    land_fractions,
):
    """
    Make climate realisations based on pre-calculated MESMER parameters

    Parameters
    ----------
    preds_lt : dict
        nested dictionary of predictors for local trends with keys

        - [pred][scen] (1d/ 2d arrays (time)/(run, time) of predictor for specific scenario)
    params_lt : dict
        dictionary with the trained local trend parameters

        - ["targs"] (emulated variables, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["method_each_gp_sep"] (states if method is applied to each grid point
          separately, bool)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (emission scenarios used for training, list of strs)
        - [xx] (additional params depending on method employed)
        - ["full_model_contains_lv"] (whether the full model contains part of the local
          variability module, bool)
    params_lv : dict
        dictionary with the trained local variability parameters

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - [xx] (additional keys depend on employed method)
    params_gv : dict

        - ["targ"] (variable which is emulated, str)
        - ["esm"] (Earth System Model, str)
        - ["method"] (applied method, str)
        - ["preds"] (predictors, list of strs)
        - ["scenarios"] (scenarios which are used for training, list of strs)
        - [xx] (additional keys depend on employed method and are listed in
          train_gv_T_method() function)
    time : dict

        - ["scenario"] timepoints (1D np.ndarray) used for training of the scenario
          (note that hist and scenario e.g. ssp126 are kept separate)
    n_realisations : int
        Number of realisations to draw
    seeds : dict

        - ["esm"] (dict):
            ["scenario"] (dict):
                ["gv"] (seed for global variability)
                ["lv"] (seed for local variability)
    land_fractions : xr.DataArray
        Land fractions of each cell. Used to convert the MESMER outputs back onto grids.
    """

    class _Config:
        """Workaround to mock the ``cfg`` interface used elsewhere"""

        def __init__(self, n_realisations, seeds):
            self.nr_emus_v = n_realisations
            self.seed = seeds

    cfg = _Config(n_realisations, seeds)

    # TODO: add better checks for what happens if scenarios have different
    # time axis etc.
    a_scenario_key = [k for k in time.keys() if k != "hist"][0]
    time_all = np.concatenate([time["hist"], time[a_scenario_key]])
    esm_gv_T = params_gv_T["esm"]
    time_seeds = seeds[esm_gv_T].keys()
    preds_gv = {"time": {k: time_all for k in time_seeds}}

    emus_gv_T = create_emus_gv(params_gv_T, preds_gv, cfg, save_emus=False)
    preds_lv = {"gvtas": emus_gv_T}

    emus_lt = create_emus_lt(params_lt, preds_lt, cfg, concat_h_f=True, save_emus=False)
    emus_lv = create_emus_lv(params_lv, preds_lv, cfg, save_emus=False)
    emus_l = create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=False)

    # xarray DataArrays or DataSets with labelled dimensions
    out = _convert_raw_mesmer_to_xarray(emus_l, land_fractions, time=time)

    return out


def _convert_raw_mesmer_to_xarray(emulations, land_fractions, time):
    land_fractions_stacked = land_fractions.stack(z=("lat", "lon")).dropna("z")

    tmp = []
    for scenario, outputs in emulations.items():
        for variable, values in outputs.items():
            time = np.concatenate([time["hist"], time[scenario.replace("h-", "")]])
            variable_out = (
                land_fractions_stacked.expand_dims({"year": time})
                .expand_dims({"realisation": range(values.shape[0])})
                .copy()
            )
            variable_out.values = values

            variable_out = variable_out.reset_index("z").expand_dims(
                {"scenario": [scenario]}
            )
            variable_out.name = variable

            tmp.append(variable_out)

    out = xr.merge(tmp)

    return out
