# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to aid the training of MESMER.
"""

import numpy as np
import xarray as xr


def train_l_prepare_X_y_wgteq(preds, targs):
    """
    Create single array of predictors, single array of targets, and single array of
    weights.

    Parameters
    ----------
    preds : dict
        empty dictionary if none, else nested dictionary of predictors with keys

        - [pred][scen]  (1d/ 2d arrays (time)/(run, time) of predictor for specific
          scenario)
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
    # predictors are not influenced by whether there is a single or multiple targets
    targ = targs[targ_name]
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
        # index & name for predictors
        for p, pred_name in enumerate(pred_names):
            s = 0  # index for samples
            pred_raw = preds[pred_name]  # values of predictor p
            for scen in scens:
                # if 1 time series per run for predictor (e.g., gv)
                if pred_raw[scen].ndim == 2:
                    # nr_runs*nr_ts for this specific scenario
                    k = pred_raw[scen].shape[0] * pred_raw[scen].shape[1]
                    X[s : s + k, p] = pred_raw[scen].flatten()
                    s += k
                # if single time series as predictor (e.g. gt): repeat ts as many times
                # as runs available
                elif pred_raw[scen].ndim == 1:
                    nr_runs, nr_ts, nr_gps = targ[scen].shape
                    nr_samples_scen = nr_runs * nr_ts
                    X[s : s + nr_samples_scen, p] = np.tile(pred_raw[scen], nr_runs)
                    s += nr_samples_scen
                else:
                    raise ValueError("Predictors of this shape cannot be processed.")

    # derive y (i.e. array of targets)
    y = np.zeros([nr_samples, nr_gps, nr_targs])
    for t, targ_name in enumerate(targ_names):
        targ = targs[targ_name]
        s = 0
        for scen in scens:
            # nr_runs * nr_ts for this scenario
            k = targ[scen].shape[0] * targ[scen].shape[1]
            y[s : s + k, :, t] = targ[scen].reshape(k, -1)
            s += k

    return X, y, wgt_scen_eq


def _train_l_prepare_X_y_wgteq_xr(preds, targs):
    """As ``train_l_prepare_X_y_wgteq`` but returning xarray data objects

    TODO: remove and replace by functionality in high level data array

    Create single array of predictors, single array of targets, and single array of
    weights.
    """

    targ_names = list(targs.keys())
    targ_name = targ_names[0]  # same approach for each targ
    pred_names = list(preds.keys())

    if len(targ_names) != 1:
        raise ValueError("Can only handle one target")

    # identify characteristics of the predictors and the targets
    # predictors are not influenced by whether there is a single or multiple targets
    targ = targs[targ_name]
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

    wgt_scen_eq = xr.DataArray(wgt_scen_eq, dims="sample")

    nr_preds = len(pred_names)

    # derive X (ie array of predictors)
    X = dict()
    if nr_preds == 0:
        raise NotImplementedError("Cannot handle 0 predictors.")
    else:

        # index & name for predictors
        for p, pred_name in enumerate(pred_names):

            # values of predictor p
            pred_raw = preds[pred_name]

            out = list()
            for scen in scens:
                # if 1 time series per run for predictor (e.g., gv)
                if pred_raw[scen].ndim == 2:
                    out.append(pred_raw[scen].flatten())

                # if single time series as predictor (e.g. gt): repeat ts as many times
                # as runs available
                elif pred_raw[scen].ndim == 1:
                    nr_runs = targ[scen].shape[0]
                    out.append(np.tile(pred_raw[scen], nr_runs))
                else:
                    raise ValueError("Predictors of this shape cannot be processed.")

            out = np.concatenate(out)
            X[pred_name] = xr.DataArray(out, dims="sample")

    # derive y (i.e. array of targets)
    y = dict()
    for t, targ_name in enumerate(targ_names):
        targ = targs[targ_name]

        out = list()
        for scen in scens:
            # get rid of unused 'targ' dimension
            out.append(targ[scen].squeeze())

        out = np.concatenate(out)

        y[targ_name] = xr.DataArray(out, dims=("sample", "cell"))

    return X, y, wgt_scen_eq
