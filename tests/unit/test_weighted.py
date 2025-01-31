import numpy as np
import pytest
import xarray as xr

import mesmer
from mesmer.core._datatreecompat import DataTree


def data_lon_lat(as_dataset, x_dim="lon", y_dim="lat"):

    lon = np.arange(0.5, 360, 2)
    lat = np.arange(90, -91, -2)
    time = np.arange(3)

    data = np.random.randn(*time.shape, *lat.shape, *lon.shape)

    da = xr.DataArray(
        data,
        dims=("time", y_dim, x_dim),
        coords={"time": time, x_dim: lon, y_dim: lat},
        attrs={"key": "da_attrs"},
    )

    ds = xr.Dataset(data_vars={"data": da, "scalar": 1}, attrs={"key": "ds_attrs"})

    if as_dataset:
        return ds
    return ds.data


def test_lat_weights_scalar():

    np.testing.assert_allclose(mesmer.weighted.lat_weights(90), 0.0, atol=1e-7)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(45), np.sqrt(2) / 2)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(0), 1.0, atol=1e-7)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(-45), np.sqrt(2) / 2)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(-90), 0.0, atol=1e-7)


def test_lat_weights():

    attrs = {"key": "value"}
    lat_coords = np.arange(90, -91, -1)
    lat = xr.DataArray(
        lat_coords, dims=("lat"), coords={"lat": lat_coords}, name="lat", attrs=attrs
    )

    expected = np.cos(np.deg2rad(lat_coords))
    expected = xr.DataArray(
        expected, dims=("lat"), coords={"lat": lat_coords}, name="lat", attrs=attrs
    )

    result = mesmer.weighted.lat_weights(lat)

    xr.testing.assert_identical(result, expected)


def test_lat_weights_2D_warn_2D():

    lat = np.arange(10).reshape(2, 5)

    with pytest.warns(UserWarning, match="non-regular grids"):
        mesmer.weighted.lat_weights(lat)


@pytest.mark.parametrize("lat", [-91, 90.1])
def test_lat_weights_2D_error_90(lat):

    with pytest.raises(ValueError, match="`lat_coords` must be between -90 and 90"):
        mesmer.weighted.lat_weights(lat)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_weighted_mean_errors_wrong_weights(as_dataset):

    data = data_lon_lat(as_dataset)
    weights = mesmer.weighted.lat_weights(data["lat"])
    weights = weights.isel(lat=slice(None, weights.size - 3))

    with pytest.raises(ValueError, match="`data` and `weights` don't exactly align."):
        mesmer.weighted.weighted_mean(data, weights=weights, dims=("lat", "lon"))

    with pytest.raises(ValueError, match="`data` and `weights` don't exactly align."):
        mesmer.weighted.weighted_mean(data, weights=weights, dims=("lat", "lon"))


def _test_weighted_mean(as_dataset, **kwargs):
    # not checking the actual values

    data = data_lon_lat(as_dataset, **kwargs)

    y_dim = kwargs.get("y_dim", "lat")
    weights = mesmer.weighted.lat_weights(data[y_dim])

    dims = list(kwargs.values()) if kwargs else ("lat", "lon")

    result = mesmer.weighted.weighted_mean(data, weights=weights, dims=dims)

    if as_dataset:
        # ensure scalar is not broadcast
        assert result.scalar.ndim == 0
        assert result.attrs == {"key": "ds_attrs"}

        result_da = result.data
    else:
        result_da = result

    assert result_da.ndim == 1
    assert result_da.notnull().all()

    assert result_da.attrs == {"key": "da_attrs"}


@pytest.mark.parametrize("as_dataset", [True, False])
def test_calc_weighted_mean_default(as_dataset):

    _test_weighted_mean(as_dataset)


@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("x_dim", ("x", "lon"))
@pytest.mark.parametrize("y_dim", ("y", "lat"))
def test_calc_weighted_mean(as_dataset, x_dim, y_dim):

    _test_weighted_mean(
        as_dataset,
        x_dim=x_dim,
        y_dim=y_dim,
    )


@pytest.mark.parametrize("as_dataset", (True, False))
def test_weighted_no_scalar_expand(as_dataset):

    data = data_lon_lat(as_dataset)
    weights = xr.ones_like(data.lat)

    result = mesmer.weighted.weighted_mean(data, weights=weights, dims="lon")

    expected = data.mean("lon")

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("x_dim", ("x", "lon"))
@pytest.mark.parametrize("y_dim", ("y", "lat"))
def test_global_mean_no_weights_passed(as_dataset, x_dim, y_dim):

    data = data_lon_lat(as_dataset, y_dim=y_dim, x_dim=x_dim)

    weights = mesmer.weighted.lat_weights(data[y_dim])

    result = mesmer.weighted.global_mean(data, x_dim=x_dim, y_dim=y_dim)

    dims = (x_dim, y_dim)
    expected = mesmer.weighted.weighted_mean(data, weights=weights, dims=dims)

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("as_dataset", (True, False))
def test_global_mean_weights_passed(as_dataset):

    data = data_lon_lat(as_dataset)

    weights = xr.ones_like(data["lat"])

    result = mesmer.weighted.global_mean(data, weights=weights)

    expected = data.mean(("lat", "lon"))

    xr.testing.assert_allclose(result, expected)


def test_equal_sceanrio_weights_from_datatree():
    dt = DataTree()

    n_members_ssp119 = 3
    n_members_ssp585 = 2
    n_gridcells = 3
    n_ts = 30

    ssp119 = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.ones((n_ts, n_members_ssp119)), dims=("time", "member")
            )
        }
    )
    ssp119 = ssp119.assign_coords(time=np.arange(n_ts))
    ssp585 = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.ones((n_ts, n_members_ssp585)), dims=("time", "member")
            )
        }
    )
    ssp585 = ssp585.assign_coords(member=np.arange(n_members_ssp585))
    dt = DataTree()
    dt["ssp119"] = DataTree(ssp119)
    dt["ssp585"] = DataTree(ssp585)

    result1 = mesmer.weighted.equal_scenario_weights_from_datatree(dt)
    expected = DataTree.from_dict(
        {
            "ssp119": DataTree(
                xr.full_like(ssp119, fill_value=1 / n_members_ssp119).rename(
                    {"tas": "weights"}
                )
            ),
            "ssp585": DataTree(
                xr.full_like(ssp585, fill_value=1 / n_members_ssp585).rename(
                    {"tas": "weights"}
                )
            ),
        }
    )

    # TODO: replace with datatree testing funcs when switching to xarray internal DataTree
    assert result1.equals(expected)

    dt["ssp119"] = DataTree(
        dt.ssp119.ds.expand_dims(gridcell=np.arange(n_gridcells), axis=1)
    )
    dt["ssp585"] = DataTree(
        dt.ssp585.ds.expand_dims(gridcell=np.arange(n_gridcells), axis=1)
    )

    result2 = mesmer.weighted.equal_scenario_weights_from_datatree(
        dt, ens_dim="member", time_dim="time"
    )
    # TODO: replace with datatree testing funcs when switching to xarray internal DataTree
    assert result2.equals(expected)


def test_create_equal_sceanrio_weights_from_datatree_checks():

    dt = DataTree()
    ssp119 = xr.Dataset(
        {"tas": xr.DataArray(np.ones((20, 2)), dims=("time", "member"))}
    )
    ssp585 = xr.Dataset(
        {"tas": xr.DataArray(np.ones((20, 3)), dims=("time", "member"))}
    )
    dt = DataTree()
    dt["ssp119"] = DataTree(ssp119)
    dt["ssp585"] = DataTree(ssp585)

    # too deep
    dt_too_deep = dt.copy()
    dt_too_deep["ssp585/1"] = DataTree(ssp585)
    with pytest.raises(ValueError, match="DataTree must have a depth of 1, not 2."):
        mesmer.weighted.equal_scenario_weights_from_datatree(dt_too_deep)

    # missing member dimension
    dt_no_member = dt.copy()
    dt_no_member["ssp119"] = DataTree(dt_no_member.ssp119.ds.sel(member=1))
    with pytest.raises(
        ValueError, match="Member dimension 'member' not found in dataset."
    ):
        mesmer.weighted.equal_scenario_weights_from_datatree(dt_no_member)

    # missing time dimension
    dt_no_time = dt.copy()
    dt_no_time["ssp119"] = DataTree(dt_no_time.ssp119.ds.sel(time=1))
    with pytest.raises(ValueError, match="Time dimension 'time' not found in dataset."):
        mesmer.weighted.equal_scenario_weights_from_datatree(dt_no_time)

    # multiple data variables
    dt_multiple_vars = dt.copy()
    dt_multiple_vars["ssp119"] = DataTree(
        xr.Dataset(
            {
                "tas": xr.DataArray(np.ones((20, 2)), dims=("time", "member")),
                "tas2": xr.DataArray(np.ones((20, 2)), dims=("time", "member")),
            }
        )
    )
    with pytest.raises(
        ValueError, match="Dataset must only contain one data variable."
    ):
        mesmer.weighted.equal_scenario_weights_from_datatree(dt_multiple_vars)
