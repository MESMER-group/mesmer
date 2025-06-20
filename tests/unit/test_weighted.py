from typing import Literal, overload

import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

import mesmer
from mesmer.core.datatree import map_over_datasets
from mesmer.core.utils import _check_dataarray_form, _check_dataset_form
from mesmer.testing import _convert


@overload
def data_lon_lat(
    datatype: Literal["DataArray"], x_dim="lon", y_dim="lat"
) -> xr.DataArray: ...


@overload
def data_lon_lat(
    datatype: Literal["Dataset"], x_dim="lon", y_dim="lat"
) -> xr.Dataset: ...
@overload
def data_lon_lat(
    datatype: Literal["DataTree"], x_dim="lon", y_dim="lat"
) -> xr.DataTree: ...


def data_lon_lat(
    datatype: Literal["DataArray", "Dataset", "DataTree"], x_dim="lon", y_dim="lat"
) -> xr.DataArray | xr.Dataset | xr.DataTree:

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

    if datatype == "Dataset":
        return ds
    elif datatype == "DataTree":
        return xr.DataTree.from_dict({"node": ds})
    return ds.data


def test_lat_weights_scalar():

    np.testing.assert_allclose(mesmer.weighted.lat_weights(90), 0.0, atol=1e-7)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(45), np.sqrt(2) / 2)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(0), 1.0, atol=1e-7)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(-45), np.sqrt(2) / 2)
    np.testing.assert_allclose(mesmer.weighted.lat_weights(-90), 0.0, atol=1e-7)


def test_lat_weights(datatype):

    attrs = {"key": "value"}
    lat_coords = xr.Variable("lat", np.arange(90, -91, -1), attrs=attrs)
    data = np.ones_like(lat_coords)
    lat = xr.DataArray(
        data, dims=("lat"), coords={"lat": lat_coords}, name="data", attrs=attrs
    )
    lat = _convert(lat, datatype)

    expected = np.cos(np.deg2rad(lat_coords))
    expected = xr.DataArray(
        expected, dims=("lat"), coords={"lat": lat_coords}, name="weights", attrs=attrs
    )
    expected = _convert(expected, datatype)

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


def test_weighted_mean_errors_wrong_weights(datatype):

    data = data_lon_lat(datatype)
    weights = mesmer.weighted.lat_weights(data, "lat")
    weights = weights.isel(lat=slice(None, -3))

    with pytest.raises(ValueError, match="`data` and `weights` don't exactly align."):
        mesmer.weighted.weighted_mean(data, weights, dims=("lat", "lon"))

    with pytest.raises(ValueError, match="`data` and `weights` don't exactly align."):
        mesmer.weighted.weighted_mean(data, weights, dims=("lat", "lon"))


def _test_weighted_mean(datatype, **kwargs):
    # not checking the actual values

    data = data_lon_lat(datatype, **kwargs)

    y_dim = kwargs.get("y_dim", "lat")

    weights = mesmer.weighted.lat_weights(data, y_dim)

    dims = list(kwargs.values()) if kwargs else ("lat", "lon")

    result = mesmer.weighted.weighted_mean(data, weights, dims=dims)

    if datatype == "DataTree":
        assert isinstance(result, xr.DataTree)
        result = result["node"].to_dataset()

    if datatype in ("DataTree", "Dataset"):
        # ensure scalar is not broadcast
        assert result.scalar.ndim == 0

        # NOTE: DataTree attrs fixed in https://github.com/pydata/xarray/pull/10219
        if datatype != "DataTree" or Version(xr.__version__) > Version("2025.3.1"):
            assert result.attrs == {"key": "ds_attrs"}

        result_da = result.data
    else:
        result_da = result

    assert result_da.ndim == 1
    assert result_da.notnull().all()

    assert result_da.attrs == {"key": "da_attrs"}


def test_calc_weighted_mean_default(datatype):

    _test_weighted_mean(datatype)


@pytest.mark.parametrize("x_dim", ("x", "lon"))
@pytest.mark.parametrize("y_dim", ("y", "lat"))
def test_calc_weighted_mean(datatype, x_dim, y_dim):

    _test_weighted_mean(
        datatype,
        x_dim=x_dim,
        y_dim=y_dim,
    )


def test_weighted_no_scalar_expand(datatype):

    data = data_lon_lat(datatype)

    lat = (data["node"].to_dataset() if datatype == "DataTree" else data).lat
    weights = xr.ones_like(lat)

    result = mesmer.weighted.weighted_mean(data, weights, dims="lon")

    expected = data.mean("lon")

    if datatype == "DataTree":
        map_over_datasets(xr.testing.assert_allclose, result, expected)
    else:
        xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("x_dim", ("x", "lon"))
@pytest.mark.parametrize("y_dim", ("y", "lat"))
def test_global_mean_no_weights_passed(datatype, x_dim, y_dim):

    data = data_lon_lat(datatype, y_dim=y_dim, x_dim=x_dim)

    result = mesmer.weighted.global_mean(data, x_dim=x_dim, y_dim=y_dim)

    dims = (x_dim, y_dim)
    weights = mesmer.weighted.lat_weights(data, y_dim)
    expected = mesmer.weighted.weighted_mean(data, weights, dims=dims)

    xr.testing.assert_equal(result, expected)


def test_global_mean_dataarray_weights_passed(datatype):

    data = data_lon_lat(datatype)

    lat = (data["node"].to_dataset() if datatype == "DataTree" else data)["lat"]
    weights = xr.ones_like(lat)

    result = mesmer.weighted.global_mean(data, weights=weights)

    expected = data.mean(("lat", "lon"))

    if datatype == "DataTree":
        map_over_datasets(xr.testing.assert_allclose, result, expected)
    else:
        xr.testing.assert_allclose(result, expected)


def test_global_mean_weights_passed(datatype):

    data = data_lon_lat(datatype)

    lat = (data["node"].to_dataset() if datatype == "DataTree" else data)["lat"]
    weights = xr.ones_like(lat)
    weights.name = "weights"
    weights = _convert(weights, datatype)

    result = mesmer.weighted.global_mean(data, weights=weights)

    expected = data.mean(("lat", "lon"))

    if datatype == "DataTree":
        map_over_datasets(xr.testing.assert_allclose, result, expected)
    else:
        xr.testing.assert_allclose(result, expected)


def test_equal_scenario_weights_from_datatree():
    dt = xr.DataTree()

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
    dt = xr.DataTree()
    dt["ssp119"] = xr.DataTree(ssp119)
    dt["ssp585"] = xr.DataTree(ssp585)

    result1 = mesmer.weighted.equal_scenario_weights_from_datatree(dt)
    expected = xr.DataTree.from_dict(
        {
            "ssp119": xr.DataTree(
                xr.full_like(ssp119, fill_value=1 / n_members_ssp119).rename(
                    {"tas": "weights"}
                )
            ),
            "ssp585": xr.DataTree(
                xr.full_like(ssp585, fill_value=1 / n_members_ssp585).rename(
                    {"tas": "weights"}
                )
            ),
        }
    )

    # TODO: replace with datatree testing funcs when switching to xarray internal DataTree
    assert result1.equals(expected)

    dt["ssp119"] = xr.DataTree(
        dt.ssp119.ds.expand_dims(gridcell=np.arange(n_gridcells), axis=1)
    )
    dt["ssp585"] = xr.DataTree(
        dt.ssp585.ds.expand_dims(gridcell=np.arange(n_gridcells), axis=1)
    )

    result2 = mesmer.weighted.equal_scenario_weights_from_datatree(
        dt, ens_dim="member", time_dim="time"
    )
    # TODO: replace with datatree testing funcs when switching to xarray internal DataTree
    assert result2.equals(expected)


def test_create_equal_scenario_weights_from_datatree_checks():

    dt = xr.DataTree()
    ssp119 = xr.Dataset(
        {"tas": xr.DataArray(np.ones((20, 2)), dims=("time", "member"))}
    )
    ssp585 = xr.Dataset(
        {"tas": xr.DataArray(np.ones((20, 3)), dims=("time", "member"))}
    )
    dt = xr.DataTree()
    dt["ssp119"] = xr.DataTree(ssp119)
    dt["ssp585"] = xr.DataTree(ssp585)

    # too deep
    dt_too_deep = dt.copy()
    dt_too_deep["ssp585/1"] = xr.DataTree(ssp585)
    with pytest.raises(ValueError, match="DataTree must have a depth of 1, not 2."):
        mesmer.weighted.equal_scenario_weights_from_datatree(dt_too_deep)

    # missing member dimension
    dt_no_member = dt.copy()
    dt_no_member["ssp119"] = xr.DataTree(dt_no_member.ssp119.ds.sel(member=1))
    with pytest.raises(
        ValueError, match="Member dimension 'member' not found in dataset."
    ):
        mesmer.weighted.equal_scenario_weights_from_datatree(dt_no_member)

    # missing time dimension
    dt_no_time = dt.copy()
    dt_no_time["ssp119"] = xr.DataTree(dt_no_time.ssp119.ds.sel(time=1))
    with pytest.raises(ValueError, match="Time dimension 'time' not found in dataset."):
        mesmer.weighted.equal_scenario_weights_from_datatree(dt_no_time)

    # multiple data variables
    dt_multiple_vars = dt.copy()
    dt_multiple_vars["ssp119"] = xr.DataTree(
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


def test_get_weights_density():
    n = 3

    weights = mesmer.weighted.get_weights_density(
        pred_data=np.arange(n),
    )

    np.testing.assert_equal(weights, weights / np.sum(weights))


def test_get_weights_density_ds():
    pred_data = xr.Dataset(
        {
            "predictor1": (("x", "y"), np.arange(9).reshape(3, 3)),
            "predictor2": (("x", "y"), np.arange(9).reshape(3, 3)),
        }
    )

    weights = mesmer.weighted.get_weights_density(
        pred_data=pred_data,
    )

    _check_dataset_form(weights, "weights", required_vars=["weights"])
    _check_dataarray_form(
        weights.weights, "weights", required_dims=("x", "y"), shape=(3, 3)
    )


def test_get_weights_density_dt():
    nts1, nts2 = 10, 12
    nmem1, nmem2 = 2, 3

    rng = np.random.default_rng(0)

    time_coord1 = np.arange(nts1)
    time_coord2 = np.arange(nts2)
    member_coord1 = np.arange(nmem1)
    member_coord2 = np.arange(nmem2)

    arr1 = xr.DataArray(
        rng.normal(loc=0, scale=0.1, size=(nts1, nmem1)),
        dims=("time", "member"),
        coords={"time": time_coord1, "member": member_coord1},
    )
    arr2 = xr.DataArray(
        rng.normal(loc=0, scale=0.1, size=(nts1, nmem1)),
        dims=("time", "member"),
        coords={"time": time_coord1, "member": member_coord1},
    )
    arr3 = xr.DataArray(
        rng.normal(loc=0, scale=0.1, size=(nts2, nmem2)),
        dims=("time", "member"),
        coords={"time": time_coord2, "member": member_coord2},
    )
    arr4 = xr.DataArray(
        rng.normal(loc=0, scale=0.1, size=(nts2, nmem2)),
        dims=("time", "member"),
        coords={"time": time_coord2, "member": member_coord2},
    )

    pred_data = xr.DataTree.from_dict(
        {
            "scenario1": xr.Dataset(
                {
                    "predictor1": arr1,
                    "predictor2": arr2,
                }
            ),
            "scenario2": xr.Dataset(
                {
                    "predictor1": arr3,
                    "predictor2": arr4,
                }
            ),
        }
    )

    weights = mesmer.weighted.get_weights_density(
        pred_data=pred_data,
    )

    scen1 = weights["scenario1"].to_dataset()
    scen2 = weights["scenario2"].to_dataset()

    _check_dataset_form(scen1, "weights", required_vars=["weights"])
    _check_dataarray_form(
        scen1.weights, "weights", required_dims=("time", "member"), shape=(nts1, nmem1)
    )

    _check_dataset_form(scen2, "weights", required_vars=["weights"])
    _check_dataarray_form(
        scen2.weights, "weights", required_dims=("time", "member"), shape=(nts2, nmem2)
    )


def test_weighted_median():
    data = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

    median = mesmer.weighted.weighted_median(data, weights)

    # The weighted median should be the value that splits the data into two halves
    expected_median = 3
    np.testing.assert_almost_equal(median, expected_median)
