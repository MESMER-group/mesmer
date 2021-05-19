# code vendored from xarray under the conditions of their license
# see licenses/XARRAY_LICENSE

import numpy as np

# TODO: potentially use xr.plot.utils._infer_interval_breaks. However, that's not
# public API so certainly need tests first.


def infer_interval_breaks(x, y, clip=False):
    """Find edges of gridcells, given their centers."""

    if len(x.shape) == 1:
        x = _infer_interval_breaks(x)
        y = _infer_interval_breaks(y)
    else:
        # we have to infer the intervals on both axes
        x = _infer_interval_breaks(x, axis=1)
        x = _infer_interval_breaks(x, axis=0)
        y = _infer_interval_breaks(y, axis=1)
        y = _infer_interval_breaks(y, axis=0)

    if clip:
        y = np.clip(y, -90, 90)

    return x, y


def _infer_interval_breaks(coord, axis=0):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> _infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    """

    if not _is_monotonic(coord, axis=axis):
        raise ValueError(
            "The input coordinate is not sorted in increasing "
            "order along axis %d. This can lead to unexpected "
            "results. Consider calling the `sortby` method on "
            "the input DataArray. To plot data with categorical "
            "axes, consider using the `heatmap` function from "
            "the `seaborn` statistical plotting library." % axis
        )

    coord = np.asarray(coord)
    deltas = 0.5 * np.diff(coord, axis=axis)
    if deltas.size == 0:
        deltas = np.array(0.0)
    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim_last = tuple(
        slice(None, -1) if n == axis else slice(None) for n in range(coord.ndim)
    )
    return np.concatenate([first, coord[trim_last] + deltas, last], axis=axis)


def _is_monotonic(coord, axis=0):
    """
    >>> _is_monotonic(np.array([0, 1, 2]))
    True
    >>> _is_monotonic(np.array([2, 1, 0]))
    True
    >>> _is_monotonic(np.array([0, 2, 1]))
    False
    """
    coord = np.asarray(coord)

    if coord.shape[axis] < 3:
        return True
    else:
        n = coord.shape[axis]
        delta_pos = coord.take(np.arange(1, n), axis=axis) >= coord.take(
            np.arange(0, n - 1), axis=axis
        )
        delta_neg = coord.take(np.arange(1, n), axis=axis) <= coord.take(
            np.arange(0, n - 1), axis=axis
        )

    return np.all(delta_pos) or np.all(delta_neg)
