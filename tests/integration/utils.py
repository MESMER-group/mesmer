import numpy as np
import numpy.testing as npt
import xarray as xr
import xarray.testing as xrt


def _check_dict(first, second, first_name="left", second_name="right"):
    for k in first:
        first_val = first[k]
        try:
            second_val = second[k]
        except KeyError:
            raise AssertionError(
                "Key `{}` is in '{}' but is not in '{}'".format(
                    k, first_name, second_name
                )
            )

        assert type(first_val) == type(second_val)
        if isinstance(first_val, dict):
            _check_dict(first_val, second_val, first_name, second_name)
        elif isinstance(first_val, np.ndarray):
            npt.assert_allclose(first_val, second_val)
        elif isinstance(first_val, xr.DataArray):
            xrt.assert_allclose(first_val, second_val)
        elif np.issubdtype(np.array(first_val).dtype, np.number):
            npt.assert_allclose(first_val, second_val)
        else:
            assert first_val == second_val, k


def trend_data_1D(n_timesteps=30, intercept=0, slope=1, scale=1):

    time = np.arange(n_timesteps)

    rng = np.random.default_rng(0)
    scatter = rng.normal(scale=scale, size=n_timesteps)

    data = intercept + slope * time + scatter

    return xr.DataArray(data, dims=("time"), coords={"time": time})


def trend_data_2D(n_timesteps=30, n_lat=3, n_lon=2, intercept=0, slope=1, scale=1):

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

    return xr.DataArray(data, dims=("cells", "time"), coords=coords)
