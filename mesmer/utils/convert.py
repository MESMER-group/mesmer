"""
mesmer.utils.convert
===================
Functions to process data.


Functions:
    convert_dict_to_arr()

"""


import numpy as np


def convert_dict_to_arr(var_dict):
    """Convert dictionary to array.

    Args:
    - var_dict (dict): nested variable (e.g., tas) dictionary with keys
        [scen][run] (xd array (time,x) of variable)

    Returns:
    - var_arr (dict): variable dictionary with keys
        [scen] (xd array (run,time,x) of variable)

    """

    scenarios = list(var_dict.keys())

    var_arr = {}

    for scen in scenarios:
        runs = list(var_dict[scen])
        shape_run = list(var_dict[scen][runs[0]].shape)
        var_arr[scen] = np.zeros([len(runs)] + shape_run)

        for i in np.arange(len(runs)):
            var_arr[scen][i] = var_dict[scen][runs[i]]

    return var_arr
