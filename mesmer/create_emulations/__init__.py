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

    land_fractions
    """
    import pdb
    pdb.set_trace()
