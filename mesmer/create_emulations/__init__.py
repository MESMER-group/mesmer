"""
Collection of functions to create emulations with MESMER.

"""
# flake8: noqa

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

    params_lt : dict
        Description to come

    preds_lv : dict
        Dictionary with a single key, ``gvtas``, whose value is itself a
        dictionary. The value should also have a single key, ``all``, the
        value for which should be an array of shape (n_realisations,
        n_timesteps).

    a
    """
    import pdb
    pdb.set_trace()
