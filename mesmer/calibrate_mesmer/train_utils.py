# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Functions to aid the training of MESMER.
"""

import numpy as np


def get_scenario_weights(target):
    """
    derive scenario weights such that each has equal weight, i.e., 1 / number of samples
    (= nr_runs * nr_ts)

    Parameters
    ----------
    target : dict
        dictionary of targets with key

        - [scen] (3d array (run, time, gp) of target for specific scenario)

    Returns
    -------
    wgt_scen_eq : np.ndarray
        1d array (sample) of sample weights based on equal treatment of each scenario
        (if scen has more samples, each sample gets less weight)
    """

    weights = list()

    # loop through scenarios
    for array in target.values():

        nr_runs, nr_ts, __ = array.shape
        nr_samples_scen = nr_runs * nr_ts

        weights.append(np.full(nr_samples_scen, 1 / nr_runs))

    return np.concatenate(weights)


def _stack_target(target):
    """stack target for all scenarios"""

    out = list()

    # loop through scenarios
    for array in target.values():
        # flatten ensemble members
        array = array.reshape(1, -1, array.shape[-1])
        out.append(array.squeeze())

    return np.concatenate(out)


def _stack_predictor(predictor, target):
    """stack predictor for all scenarios"""

    out = list()
    for scen, values in predictor.items():

        # if 1 time series per run for predictor (e.g., gv)
        if values.ndim == 2:
            out.append(values.flatten())

        # if single time series as predictor (e.g., gt): repeat ts as many times
        # as runs available
        elif values.ndim == 1:
            nr_runs = target[scen].shape[0]
            out.append(np.tile(values, nr_runs))
        else:
            raise ValueError("Predictors of this shape cannot be processed.")

    return np.concatenate(out)


def stack_predictors_and_targets(preds, targs):
    """
    Create single array of predictors, and single array of targets

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
    X : dict of np.ndarray
        empty array if none, else 2d array (sample, pred) of predictors
    y : dict of np.ndarray
        3d array (sample, gp, targ) of targets
    """

    # can only be one target at the moment
    targ_name = list(targs.keys())[0]

    X = {key: _stack_predictor(pred, targs[targ_name]) for key, pred in preds.items()}

    y = {key: _stack_target(targ) for key, targ in targs.items()}

    return X, y
