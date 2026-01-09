from typing import Literal

import numpy as np
import xarray as xr


def assert_allclose_allowed_failures(
    actual,
    desired,
    *,
    rtol=1e-07,
    atol=0,
    allowed_failures=0,
    err_msg="",
):
    """check arrays are close but allow a number of failures

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    allowed_failures : int, default: 0
        Number of points that may violate the tolerance criteria

    Notes
    -----
    Only for numpy arrays at the moment.

    """

    __tracebackhide__ = True

    def comparison(actual, desired):

        __tracebackhide__ = True

        isclose = np.isclose(actual, desired, rtol=rtol, atol=atol)
        n_not_isclose = (~isclose).sum()
        if n_not_isclose > allowed_failures:
            return isclose
        return True

    np.testing.assert_array_compare(comparison, actual, desired, err_msg=err_msg)


def assert_dict_allclose(first, second, first_name="left", second_name="right"):
    """
    Recursively check that two dicts have the same keys and values are allclose.

    Parameters
    ----------
    first : dict
        The first dict to compare.
    second : dict
        The second dict to compare.
    first_name : str, default: "left"
        Name of the first dict for error messages.
    second_name : str, default: "right"
        Name of the second dict for error messages.

    Raises
    ------
    AssertionError
        If the dicts do not have the same keys or values are not allclose.


    """

    if not isinstance(first, dict) or not isinstance(second, dict):
        raise AssertionError(f"must be dicts, got {type(first)} and {type(second)}")

    extra_first = first.keys() - second.keys()
    if extra_first:
        raise AssertionError(f"'{first_name}' has extra keys: '{extra_first}'")

    extra_second = second.keys() - first.keys()
    if extra_second:
        raise AssertionError(f"'{second_name}' has extra keys: '{extra_second}'")

    for key, first_val in first.items():

        second_val = second[key]

        # allow mixing arrays and scalars
        if not (
            isinstance(first_val, np.ndarray) or isinstance(second_val, np.ndarray)
        ):
            type_first, type_second = type(first_val), type(second_val)
            assert type_first == type_second, f"{key}: {type_first} != {type_second}"

        if isinstance(first_val, dict):
            assert_dict_allclose(first_val, second_val, first_name, second_name)
        elif isinstance(first_val, np.ndarray):
            np.testing.assert_allclose(first_val, second_val, err_msg=key)
        elif isinstance(first_val, xr.DataArray | xr.Dataset):
            xr.testing.assert_allclose(first_val, second_val)
        elif np.issubdtype(np.array(first_val).dtype, np.number):
            np.testing.assert_allclose(first_val, second_val, err_msg=key)
        else:
            assert first_val == second_val, key


def trend_data_1D(*, n_timesteps=30, intercept=0.0, slope=1.0, scale=1.0, seed=0):
    """
    Generate timeseries data with linear trend and normally distributed noise.
    Parameters
    ----------
    n_timesteps : int, default: 30
        Number of time steps.
    intercept : float, default: 0.0
        Intercept of the trend.
    slope : float, default: 1.0
        Slope of the trend.
    scale : float, default: 1.0
        Standard deviation of the normally distributed noise.
    seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    xr.DataArray
        1D DataArray with dimensions ("time") with length `n_timesteps` and coordinates
        "time" with values from 0 to `n_timesteps`, containing the generated trend data.
    """

    time = np.arange(n_timesteps)

    rng = np.random.default_rng(seed)
    scatter = rng.normal(scale=scale, size=n_timesteps)

    data = intercept + slope * time + scatter

    return xr.DataArray(data, dims=("time"), coords={"time": time}, name="data")


def trend_data_2D(
    *, n_timesteps=30, n_lat=3, n_lon=2, intercept=0.0, slope=1.0, scale=1.0
) -> xr.DataArray:
    """
    Generate timeseries data with linear trend and normally distributed noise for cells.

    Parameters
    ----------
    n_timesteps : int, default: 30
        Number of time steps.
    n_lat : int, default: 3
        Number of latitude points.
    n_lon : int, default: 2
        Number of longitude points.
    intercept : float, default: 0.0
        Intercept of the trend.
    slope : float, default: 1.0
        Slope of the trend.
    scale : float, default: 1.0
        Standard deviation of the normally distributed noise.

    Returns
    -------
    xr.DataArray
        2D DataArray with dimensions ("cells", "time") where "cells" is a flattened
        combination of latitude and longitude points, and "time" has length `n_timesteps`.
        Coordinates include "time", "lat", and "lon". The DataArray contains the generated
        trend data.
    """

    n_cells = n_lat * n_lon
    time = np.arange(n_timesteps)

    rng = np.random.default_rng(0)
    scatter = rng.normal(scale=scale, size=(n_timesteps, n_cells)).T

    data = intercept + slope * time + scatter
    LON, LAT = np.meshgrid(np.arange(n_lon), np.arange(n_lat))

    coords = {
        "time": time,
        "lon": ("cells", LON.flatten()),
        "lat": ("cells", LAT.flatten()),
    }

    return xr.DataArray(data, dims=("cells", "time"), coords=coords, name="data")


def trend_data_3D(
    *, n_timesteps=30, n_lat=3, n_lon=2, intercept=0.0, slope=1.0, scale=1.0
):
    """
    Generate lat-lon field of timeseries data with a linear trend and normally distributed noise.

    Parameters
    ----------
    n_timesteps : int, default: 30
        Number of time steps.
    n_lat : int, default: 3
        Number of latitude points.
    n_lon : int, default: 2
        Number of longitude points.
    intercept : float, default: 0.0
        Intercept of the trend.
    slope : float, default: 1.0
        Slope of the trend.
    scale : float, default: 1.0
        Standard deviation of the normally distributed noise.

    Returns
    -------
    xr.DataArray
        3D DataArray with dimensions ("time", "lat", "lon") where "time" has length
        `n_timesteps`, and "lat" and "lon" have lengths `n_lat` and `n_lon`, respectively.
        Coordinates include "time", "lat", and "lon". The trend is along the "time" dimension.
    """

    data = trend_data_2D(
        n_timesteps=n_timesteps,
        n_lat=n_lat,
        n_lon=n_lon,
        intercept=intercept,
        slope=slope,
        scale=scale,
    )

    # reshape to 3D (time x lat x lon)
    return data.set_index(cells=("lat", "lon")).unstack("cells")


def _convert(da: xr.DataArray, datatype: Literal["DataArray", "Dataset", "DataTree"]):
    """Convert DataArray to specified `datatype`."""

    if datatype == "DataArray":
        return da

    if datatype == "Dataset":
        return da.to_dataset()

    if datatype == "DataTree":
        return xr.DataTree.from_dict({"node": da.to_dataset()})

    raise ValueError(f"Unknown datatype: {datatype}")
