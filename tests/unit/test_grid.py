import numpy as np
import pytest
import xarray as xr

import mesmer


def data_1D_coords(as_dataset, x_dim="lon", y_dim="lat", stack_dim="gridcell"):

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

    if as_dataset:
        return da_structured.to_dataset(), da_unstructured.to_dataset()

    return da_structured, da_unstructured


def data_2D_coords(as_dataset):

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

    if as_dataset:
        return da_structured.to_dataset(), da_unstructured.to_dataset()

    return da_structured, da_unstructured


@pytest.mark.parametrize("as_dataset", [True, False])
def test_to_unstructured_defaults(as_dataset):
    da, expected = data_1D_coords(as_dataset)

    result = mesmer.grid.stack_lat_lon(da)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_to_unstructured_multiindex(as_dataset):
    da, expected = data_1D_coords(as_dataset)

    result = mesmer.grid.stack_lat_lon(da, multiindex=True)

    expected = expected.set_index({"gridcell": ("lat", "lon")})

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("x_dim", ["lon", "x"])
@pytest.mark.parametrize("y_dim", ["lat", "y"])
@pytest.mark.parametrize("cell_dim", ["cell", "gridpoint"])
@pytest.mark.parametrize("as_dataset", [True, False])
def test_to_unstructured(x_dim, y_dim, cell_dim, as_dataset):
    da, expected = data_1D_coords(
        as_dataset, x_dim=x_dim, y_dim=y_dim, stack_dim=cell_dim
    )

    result = mesmer.grid.stack_lat_lon(da, x_dim=x_dim, y_dim=y_dim, stack_dim=cell_dim)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_to_unstructured_2D_coords(as_dataset):
    da, expected = data_2D_coords(as_dataset)

    result = mesmer.grid.stack_lat_lon(da, x_dim="x", y_dim="y")

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("coords", ["1D", "2D"])
@pytest.mark.parametrize("time_pos", [0, None])
def test_to_unstructured_dropna(dropna, coords, time_pos):

    if coords == "1D":
        da, expected = data_1D_coords(as_dataset=False)
        kwargs = {}
    else:
        da, expected = data_2D_coords(as_dataset=False)
        kwargs = {"x_dim": "x", "y_dim": "y"}

    da[slice(time_pos), 0, 0] = np.NaN

    # the gridpoint is dropped if ANY time step is NaN
    expected[:, 0] = np.NaN

    if dropna:
        expected = expected.dropna("gridcell")

    result = mesmer.grid.stack_lat_lon(da, dropna=dropna, **kwargs)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("coords", ["1D", "2D"])
def test_unstructured_roundtrip_dropna_row(coords):

    if coords == "1D":
        da_structured, __ = data_1D_coords(as_dataset=False)
        kwargs = {"x_dim": "lon", "y_dim": "lat"}
    else:
        da_structured, __ = data_2D_coords(as_dataset=False)
        kwargs = {"x_dim": "x", "y_dim": "y"}

    coords_orig = da_structured.coords.to_dataset()[list(kwargs.values())]

    da_structured[:, :, 0] = np.NaN
    expected = da_structured

    da_unstructured = mesmer.grid.stack_lat_lon(da_structured, **kwargs)
    result = mesmer.grid.unstack_lat_lon_and_align(
        da_unstructured, coords_orig, **kwargs
    )

    # roundtripping adds x & y coords - not sure if there is something to be done about
    if coords == "2D":
        result = result.drop_vars(["x", "y"])

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_from_unstructured_defaults(as_dataset):
    expected, da = data_1D_coords(as_dataset)

    coords_orig = expected.coords.to_dataset()[["lon", "lat"]]

    result = mesmer.grid.unstack_lat_lon_and_align(da, coords_orig)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("x_dim", ["lon", "x"])
@pytest.mark.parametrize("y_dim", ["lat", "y"])
@pytest.mark.parametrize("stack_dim", ["cell", "gridpoint"])
@pytest.mark.parametrize("as_dataset", [True, False])
def test_from_unstructured(x_dim, y_dim, stack_dim, as_dataset):
    expected, da = data_1D_coords(
        as_dataset, x_dim=x_dim, y_dim=y_dim, stack_dim=stack_dim
    )

    coords_orig = expected.coords.to_dataset()[[x_dim, y_dim]]
    result = mesmer.grid.unstack_lat_lon_and_align(
        da, coords_orig, x_dim=x_dim, y_dim=y_dim, stack_dim=stack_dim
    )

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_unstructured_roundtrip_1D_coords(as_dataset):

    da_structured, da_unstructured = data_1D_coords(as_dataset)

    coords_orig = da_structured.coords.to_dataset()[["lon", "lat"]]

    result = mesmer.grid.unstack_lat_lon_and_align(
        mesmer.grid.stack_lat_lon(da_structured), coords_orig
    )
    xr.testing.assert_identical(result, da_structured)

    result = mesmer.grid.stack_lat_lon(
        mesmer.grid.unstack_lat_lon_and_align(da_unstructured, coords_orig)
    )
    xr.testing.assert_identical(result, da_unstructured)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_unstructured_roundtrip_2D_coords(as_dataset):

    da_structured, da_unstructured = data_2D_coords(as_dataset)

    dims = {"x_dim": "x", "y_dim": "y"}

    coords_orig = da_structured.coords.to_dataset()[["x", "y"]]

    result = mesmer.grid.stack_lat_lon(
        mesmer.grid.unstack_lat_lon_and_align(da_unstructured, coords_orig, **dims),
        **dims,
    )
    xr.testing.assert_identical(result, da_unstructured)
