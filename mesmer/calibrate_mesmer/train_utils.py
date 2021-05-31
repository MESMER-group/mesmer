# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to aid the training of MESMER.
"""


import numpy as np


def train_l_prepare_X_y_wgteq(preds, targs):
    """
    Create single array of predictors, single array of targets, and single array of
    weights.

    Parameters
    ----------
    preds : dict
        empty dictionary if none, else nested dictionary of predictors with keys

        - [pred][scen]  (1d/ 2d arrays (time)/(run, time) of predictor for specific scenario)
    targs : dict
        nested dictionary of targets with keys

        - [targ][scen] (3d array (run, time, gp) of target for specific scenario)

    Returns
    -------
    X : np.ndarray
        empty array if none, else 2d array (sample, pred) of predictors
    y : np.ndarray
        3d array (sample, gp, targ) of targets
    wgt_scen_eq : np.ndarray
        1d array (sample) of sample weights based on equal treatment of each scenario
        (if scen has more samples, each sample gets less weight)
    """

    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # because same approach for each targ
    pred_names = list(preds.keys())

    # identify characteristics of the predictors and the targets
    targ = targs[
        targ_name
    ]  # predictors are not influenced by whether there is a single or there are multiple targets
    scens = list(targ.keys())

    # assumption: nr_runs per scen and nr_ts for these runs can vary
    # derive weights such that each scenario receives same weight (divide by nr samples)
    nr_samples = 0
    wgt_scen_eq = []
    for scen in scens:
        nr_runs, nr_ts, nr_gps = targ[scen].shape
        nr_samples_scen = nr_runs * nr_ts
        wgt_scen_eq = np.append(wgt_scen_eq, np.repeat(1 / nr_runs, nr_samples_scen))
        nr_samples += nr_samples_scen

    nr_preds = len(pred_names)
    nr_targs = len(targ_names)

    # derive X (ie array of predictors)
    if nr_preds == 0:
        X = np.empty(0)
    else:
        X = np.zeros([nr_samples, nr_preds])
        for p in np.arange(nr_preds):  # index for predictors
            pred_name = pred_names[p]  # name of predictor p
            s = 0  # index for samples
            pred_raw = preds[pred_name]  # values of predictor p
            for scen in scens:
                if (
                    len(pred_raw[scen].shape) == 2
                ):  # if 1 time series per run for predictor (e.g., gv)
                    k = (
                        pred_raw[scen].shape[0] * pred_raw[scen].shape[1]
                    )  # nr_runs*nr_ts for this specific scenario
                    X[s : s + k, p] = pred_raw[scen].flatten()
                    s += k
                elif (
                    len(pred_raw[scen].shape) == 1
                ):  # if single time series as predictor (e.g. gt): repeat ts as many times as runs available
                    nr_runs, nr_ts, nr_gps = targ[scen].shape
                    nr_samples_scen = nr_runs * nr_ts
                    X[s : s + nr_samples_scen, p] = np.tile(pred_raw[scen], nr_runs)
                    s += nr_samples_scen
                else:
                    raise ValueError("Predictors of this shape cannot be processed.")

    # derive y (ie array of targets)
    y = np.zeros([nr_samples, nr_gps, nr_targs])
    for t, targ_name in enumerate(targ_names):
        targ = targs[targ_name]
        s = 0
        for scen in scens:
            k = (
                targ[scen].shape[0] * targ[scen].shape[1]
            )  # nr_runs*nr_ts for this scenario
            y[s : s + k, :, t] = targ[scen].reshape(k, -1)
            s += k

    return X, y, wgt_scen_eq
