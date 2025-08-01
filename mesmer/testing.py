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

    time = np.arange(n_timesteps)

    rng = np.random.default_rng(seed)
    scatter = rng.normal(scale=scale, size=n_timesteps)

    data = intercept + slope * time + scatter

    return xr.DataArray(data, dims=("time"), coords={"time": time}, name="data")


def trend_data_2D(
    *, n_timesteps=30, n_lat=3, n_lon=2, intercept=0.0, slope=1.0, scale=1.0
) -> xr.DataArray:

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


def _convert(da: xr.DataArray, datatype):

    if datatype == "DataArray":
        return da

    if datatype == "Dataset":
        return da.to_dataset()

    if datatype == "DataTree":
        return xr.DataTree.from_dict({"node": da.to_dataset()})

    raise ValueError(f"Unknown datatype: {datatype}")
