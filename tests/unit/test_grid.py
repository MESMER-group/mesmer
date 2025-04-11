import numpy as np
import pytest
import xarray as xr

import mesmer
from mesmer.testing import _convert


def data_1D_coords(datatype, x_dim="lon", y_dim="lat", stack_dim="gridcell"):

    data = np.arange(2 * 3 * 4, dtype=float)
    time = [0, 1]
    lat, lon = [0, 1, 2], [0, 1, 2, 3]
    c_lat, c_lon = np.mgrid[:3, :4]

    name = "name"
    attrs = {"key": "value"}

    da_structured = xr.DataArray(
        data.reshape(2, 3, 4),
        dims=("time", y_dim, x_dim),
        coords={"time": time, y_dim: lat, x_dim: lon},
        name=name,
        attrs=attrs,
    )

    da_unstructured = xr.DataArray(
        data.reshape(2, 12),
        dims=("time", stack_dim),
        coords={
            "time": time,
            y_dim: (stack_dim, c_lat.flatten()),
            x_dim: (stack_dim, c_lon.flatten()),
        },
        name=name,
        attrs=attrs,
    )

    return _convert(da_structured, datatype), _convert(da_unstructured, datatype)


def data_2D_coords(datatype):

    data = np.arange(2 * 3 * 4, dtype=float)
    time = [0, 1]
    yc, xc = np.mgrid[90:87:-1, 0:4]
    y, x = np.mgrid[0:3, 0:4]

    name = "name"
    attrs = {"key": "value"}

    da_structured = xr.DataArray(
        data.reshape(2, 3, 4),
        dims=("time", "y", "x"),
        coords={"time": time, "yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
        name=name,
        attrs=attrs,
    )

    da_unstructured = xr.DataArray(
        data.reshape(2, 12),
        dims=("time", "gridcell"),
        coords={
            "time": time,
            "yc": ("gridcell", yc.flatten()),
            "xc": ("gridcell", xc.flatten()),
            "y": ("gridcell", y.flatten()),
            "x": ("gridcell", x.flatten()),
        },
        name=name,
        attrs=attrs,
    )

    return _convert(da_structured, datatype), _convert(da_unstructured, datatype)


def test_to_unstructured_defaults(datatype):
    da, expected = data_1D_coords(datatype)

    result = mesmer.grid.stack_lat_lon(da)

    xr.testing.assert_identical(result, expected)


def test_to_unstructured_multiindex(datatype):
    da, expected = data_1D_coords(datatype)

    result = mesmer.grid.stack_lat_lon(da, multiindex=True)

    # TODO: simplify once DataTree has set_index
    if datatype == "DataTree":

        expected = xr.DataTree.from_dict(
            {"node": expected["node"].dataset.set_index({"gridcell": ("lat", "lon")})}
        )

    else:
        expected = expected.set_index({"gridcell": ("lat", "lon")})

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("x_dim", ["lon", "x"])
@pytest.mark.parametrize("y_dim", ["lat", "y"])
@pytest.mark.parametrize("cell_dim", ["cell", "gridpoint"])
def test_to_unstructured(x_dim, y_dim, cell_dim, datatype):
    da, expected = data_1D_coords(
        datatype, x_dim=x_dim, y_dim=y_dim, stack_dim=cell_dim
    )

    result = mesmer.grid.stack_lat_lon(da, x_dim=x_dim, y_dim=y_dim, stack_dim=cell_dim)

    xr.testing.assert_identical(result, expected)


def test_to_unstructured_2D_coords(datatype):
    da, expected = data_2D_coords(datatype)

    result = mesmer.grid.stack_lat_lon(da, x_dim="x", y_dim="y")

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("coords", ["1D", "2D"])
@pytest.mark.parametrize("time_pos", [0, None])
def test_to_unstructured_dropna(dropna, coords, time_pos):

    if coords == "1D":
        da, expected = data_1D_coords("DataArray")
        kwargs = {}
    else:
        da, expected = data_2D_coords("DataArray")
        kwargs = {"x_dim": "x", "y_dim": "y"}

    da[slice(time_pos), 0, 0] = np.nan

    # the gridpoint is dropped if ANY time step is NaN
    expected[:, 0] = np.nan

    if dropna:
        expected = expected.dropna("gridcell")

    result = mesmer.grid.stack_lat_lon(da, dropna=dropna, **kwargs)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("coords", ["1D", "2D"])
def test_unstructured_roundtrip_dropna_row(coords):

    if coords == "1D":
        da_structured, __ = data_1D_coords("DataArray")
        kwargs = {"x_dim": "lon", "y_dim": "lat"}
    else:
        da_structured, __ = data_2D_coords("DataArray")
        kwargs = {"x_dim": "x", "y_dim": "y"}

    coords_orig = da_structured.coords.to_dataset()[list(kwargs.values())]

    da_structured[:, :, 0] = np.nan
    expected = da_structured

    da_unstructured = mesmer.grid.stack_lat_lon(da_structured, **kwargs)
    result = mesmer.grid.unstack_lat_lon_and_align(
        da_unstructured, coords_orig, **kwargs
    )

    # roundtripping adds x & y coords - not sure if there is something to be done about
    if coords == "2D":
        result = result.drop_vars(["x", "y"])

    xr.testing.assert_identical(result, expected)


def _get_coords(data, datatype, x_dim, y_dim):

    if datatype == "DataTree":
        data = data["node"]

    return data.coords.to_dataset()[[x_dim, y_dim]]


def test_from_unstructured_defaults(datatype):
    expected, da = data_1D_coords(datatype)

    coords_orig = _get_coords(expected, datatype, "lon", "lat")

    result = mesmer.grid.unstack_lat_lon_and_align(da, coords_orig)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("x_dim", ["lon", "x"])
@pytest.mark.parametrize("y_dim", ["lat", "y"])
@pytest.mark.parametrize("stack_dim", ["cell", "gridpoint"])
def test_from_unstructured(x_dim, y_dim, stack_dim, datatype):
    expected, da = data_1D_coords(
        datatype, x_dim=x_dim, y_dim=y_dim, stack_dim=stack_dim
    )

    coords_orig = _get_coords(expected, datatype, x_dim, y_dim)
    result = mesmer.grid.unstack_lat_lon_and_align(
        da, coords_orig, x_dim=x_dim, y_dim=y_dim, stack_dim=stack_dim
    )

    xr.testing.assert_identical(result, expected)


def test_unstructured_roundtrip_1D_coords(datatype):

    da_structured, da_unstructured = data_1D_coords(datatype)

    coords_orig = _get_coords(da_structured, datatype, "lon", "lat")

    result = mesmer.grid.unstack_lat_lon_and_align(
        mesmer.grid.stack_lat_lon(da_structured), coords_orig
    )
    xr.testing.assert_identical(result, da_structured)

    result = mesmer.grid.stack_lat_lon(
        mesmer.grid.unstack_lat_lon_and_align(da_unstructured, coords_orig)
    )
    xr.testing.assert_identical(result, da_unstructured)


def test_unstructured_roundtrip_2D_coords(datatype):

    da_structured, da_unstructured = data_2D_coords(datatype)

    dims = {"x_dim": "x", "y_dim": "y"}

    coords_orig = _get_coords(da_structured, datatype, "x", "y")

    result = mesmer.grid.stack_lat_lon(
        mesmer.grid.unstack_lat_lon_and_align(da_unstructured, coords_orig, **dims),
        **dims,
    )
    xr.testing.assert_identical(result, da_unstructured)
