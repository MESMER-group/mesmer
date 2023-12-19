# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to process data.
"""


import numpy as np


def convert_dict_to_arr(var_dict):
    """Convert dictionary to array.

    Parameters
    ----------
    var_dict : dict
        nested variable (e.g., tas) dictionary with keys

        - [scen][run] (xd array (time, x) of variable)

    Returns
    -------
    var_arr : dict
        variable dictionary with keys

        - [scen] (xd array (run, time, x) of variable)

    """

    var_arr = {
        scen: np.stack(list(run_dict.values())) for scen, run_dict in var_dict.items()
    }

    return var_arr


def separate_hist_future(var_c, time_c, cfg):
    """Separate historical and future time periods into separate keys in dictionary.

    Parameters
    ----------
    var_c : dict
        variable dictionary with concatenated historical and future scenarios as keys

        - ["h-scen_f"] (xd array of variable (run, time, x), np.ndarray)

    time_c : dict
        time dictionary with concatenated historical and future scenarios as keys

        - ["h-scen_f"] (1d array of years, np.ndarray)

    cfg : module
        config file containing metadata

    Returns
    -------
    var_s : dict
        variable dictionary with separated historical and future scenarios as keys

        - [hist] / [scen_f] (xd array of variable (run, time, x), np.ndarray)
    time_s : dict
        time dictionary with separated historical and future scenarios as keys

        - ["hist] / [scen_f] (1d array of years, np.ndarray)

    Notes
    -----
    - Assumption
        - each scenario starts in the same year
        - at least 2-dim variable array is passed with 1st dim nr runs, 2nd dim nr ts

    """

    gen = cfg.gen
    scens_c = list(var_c.keys())  # concatenated scens
    scens_f = [scen.replace("h-", "") for scen in scens_c]  # future scens

    # get last year included in historical time period
    if gen == 5:
        end_year_hist = 2005
    if gen == 6:
        end_year_hist = 2014

    # assuming all time vectors are the same
    time = time_c[scens_c[0]]

    idx_start_fut = np.where(time == end_year_hist)[0][0] + 1

    var_s, time_s = {}, {}

    # gather hist
    hist = [np.atleast_2d(var_c[scen_c])[:, :idx_start_fut] for scen_c in scens_c]
    hist = np.vstack(hist)

    # exclude duplicate historical runs that are available in several scenarios
    __, idx = np.unique(hist, axis=0, return_index=True)
    # ensure hist is not re-ordered
    var_s["hist"] = np.take(hist, indices=np.sort(idx), axis=0)

    time_s["hist"] = time[:idx_start_fut]

    # gather proj
    for scen_f, scen_c in zip(scens_f, scens_c):
        var_s[scen_f] = np.atleast_2d(var_c[scen_c])[:, idx_start_fut:]
        time_s[scen_f] = time_c[scen_c][idx_start_fut:]

    return var_s, time_s
