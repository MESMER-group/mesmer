"""
Collection of functions to create emulations with MESMER.

"""
# flake8: noqa
import pandas as pd
import xarray as xr

from .create_emus_gt import *
from .create_emus_gv import *
from .create_emus_lt import *
from .create_emus_lv import *
from .merge_emus import *


def make_realisations(
    preds_lt,
    params_lt,
    preds_lv,
    params_lv,
    n_realisations,
    seeds,
    land_fractions,
):
    """
    Make climate realisations based on pre-calculated MESMER parameters

    Parameters
    ----------
    preds_lt : dict
        Dictionary where each key is a different variable. The values are
        themselves dictionaries. These sub-dictionaries have keys which are
        different scenarios ("hist" separate from scenarios), each of which is
        a :obj:`np.ndarray` of shape ``(n_timesteps)`` i.e. the length of the
        scenario.

    params_lt : dict
        Description to come

    preds_lv : dict
        Dictionary with a single key, ``gvtas``, whose value is itself a
        dictionary. The value should also have a single key, ``all``, the
        value for which should be an array of shape
        ``(n_realisations, n_timesteps)``.

    params_lv : dict
        Description to come

    n_realisations : int
        Number of realisations to draw

    seeds : dict
        Seeds to use for random number generators. Keys are different climate
        models, values are themselves dictionaries. Each value has keys which
        are different scenarios (or ``"all"``) and values which are themselves
        dictionaries. These final sub-values contain two keys, ``["gv",
        "lv"]``, whiche define the seeds for the global variability and local
        variability generators respectively.

    land_fractions : :obj:`xarray.DataArray`
        Land fractions of each cell. Used to convert the MESMER outputs back onto grids.
    """
    class _Config:
        """TODO: remove, just used now as a way to make things not explode"""
        def __init__(self, n_realisations, seeds):
            self.nr_emus_v = n_realisations
            self.seed = seeds

    cfg = _Config(n_realisations, seeds)

    emus_lt = create_emus_lt(
        params_lt, preds_lt, cfg, concat_h_f=True, save_emus=False
    )
    emus_lv = create_emus_lv(params_lv, preds_lv, cfg, save_emus=False)
    emus_l = create_emus_l(emus_lt, emus_lv, params_lt, params_lv, cfg, save_emus=False)

    # xarray DataArrays or DataSets with labelled dimensions
    out = _convert_raw_mesmer_to_xarray(emus_l, land_fractions)

    return out


def _convert_raw_mesmer_to_xarray(emulations, land_fractions):
    land_fractions_stacked = land_fractions.stack(z=("lat", "lon")).dropna("z")

    tmp = []
    for scenario, outputs in emulations.items():
        for variable, values in outputs.items():
            variable_out = (
                land_fractions_stacked
                # TODO: actually return date times, not just integers
                .expand_dims({"timestep": range(values.shape[1])})
                .expand_dims({"realisation": range(values.shape[0])})
                .copy()
            )
            variable_out.values = values

            variable_out = variable_out.reset_index("z").expand_dims({"scenario": [scenario]})
            variable_out.name = variable

            tmp.append(variable_out)

    out = xr.merge(tmp)

    return out
