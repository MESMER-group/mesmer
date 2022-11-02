import numpy as np
import pytest
import xarray as xr

import mesmer.xarray_utils as mxu


def data_1D_dims(as_dataset, x_dim="lon", y_dim="lat", cell_dim="cell"):

    data = np.arange(2 * 3 * 4, dtype=float)
    time = [0, 1]
    lat = [0, 1, 2]
    lon = [0, 1, 2, 3]
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
        dims=("time", cell_dim),
        coords={
            "time": time,
            y_dim: (cell_dim, sorted(4 * lat)),
            x_dim: (cell_dim, 3 * lon),
        },
        name=name,
        attrs=attrs,
    )

    if as_dataset:
        return da_structured.to_dataset(), da_unstructured.to_dataset()

    return da_structured, da_unstructured


def data_2D_dims(as_dataset):

    data = np.arange(2 * 3 * 4, dtype=float)
    time = [0, 1]
    yc = np.array([[90, 90, 90, 90], [89, 89, 89, 89], [88, 88, 88, 88]])
    xc = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    yc, xc = np.mgrid[90:87:-1, 0:4]

    name = "name"
    attrs = {"key": "value"}
    da_structured = xr.DataArray(
        data.reshape(2, 3, 4),
        dims=("time", "y", "x"),
        coords={"time": time, "yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
        name=name,
        attrs=attrs,
    )

    y, x = np.mgrid[0:3, 0:4]

    da_unstructured = xr.DataArray(
        data.reshape(2, 12),
        dims=("time", "cell"),
        coords={
            "time": time,
            "yc": ("cell", yc.flatten()),
            "xc": ("cell", xc.flatten()),
            "y": ("cell", y.flatten()),
            "x": ("cell", x.flatten()),
        },
        name=name,
        attrs=attrs,
    )

    if as_dataset:
        return da_structured.to_dataset(), da_unstructured.to_dataset()

    return da_structured, da_unstructured


@pytest.mark.parametrize("as_dataset", [True, False])
def test_to_unstructured_defaults(as_dataset):
    da, expected = data_1D_dims(as_dataset)

    result = mxu.to_unstructured(da)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("x_dim", ["lon", "x"])
@pytest.mark.parametrize("y_dim", ["lat", "y"])
@pytest.mark.parametrize("cell_dim", ["cell", "gridpoint"])
@pytest.mark.parametrize("as_dataset", [True, False])
def test_to_unstructured(x_dim, y_dim, cell_dim, as_dataset):
    da, expected = data_1D_dims(as_dataset, x_dim=x_dim, y_dim=y_dim, cell_dim=cell_dim)

    result = mxu.to_unstructured(da, x_dim=x_dim, y_dim=y_dim, cell_dim=cell_dim)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_to_unstructured_2D_dims(as_dataset):
    da, expected = data_2D_dims(as_dataset)

    result = mxu.to_unstructured(da, x_dim="x", y_dim="y")
    # to_unstructured adds coordinates for "Dimensions without coordinates"
    # result = result.drop_vars(("x", "y"))

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("time_pos", [0, None])
def test_to_unstructured_dropna(dropna, time_pos):

    da, expected = data_1D_dims(as_dataset=False)

    da[slice(time_pos), 0, 0] = np.NaN

    # the gridpoint is droped if ANY time step is NaN
    expected[:, 0] = np.NaN

    if dropna:
        expected = expected.dropna("cell")

    result = mxu.to_unstructured(da, dropna=dropna)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_from_unstructured_defaults(as_dataset):
    expected, da = data_1D_dims(as_dataset)

    # todo: get only lon & lat without time for DataArray and Dataset
    coords_orig = expected.coords.to_dataset()[["lon", "lat"]]

    result = mxu.from_unstructured(da, coords_orig)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("x_dim", ["lon", "x"])
@pytest.mark.parametrize("y_dim", ["lat", "y"])
@pytest.mark.parametrize("cell_dim", ["cell", "gridpoint"])
@pytest.mark.parametrize("as_dataset", [True, False])
def test_from_unstructured(x_dim, y_dim, cell_dim, as_dataset):
    expected, da = data_1D_dims(as_dataset, x_dim=x_dim, y_dim=y_dim, cell_dim=cell_dim)
    # todo: get only lon & lat without time for DataArray and Dataset
    coords_orig = expected.coords.to_dataset()[[x_dim, y_dim]]
    result = mxu.from_unstructured(
        da, coords_orig, x_dim=x_dim, y_dim=y_dim, cell_dim=cell_dim
    )

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_unstructured_roundtrip_1D_dim(as_dataset):

    da_structured, da_unstructured = data_1D_dims(as_dataset)

    coords_orig = da_structured.coords.to_dataset()[["lon", "lat"]]

    result = mxu.from_unstructured(mxu.to_unstructured(da_structured), coords_orig)
    xr.testing.assert_identical(result, da_structured)

    result = mxu.to_unstructured(mxu.from_unstructured(da_unstructured, coords_orig))
    xr.testing.assert_identical(result, da_unstructured)


@pytest.mark.parametrize("as_dataset", [True, False])
def test_unstructured_roundtrip_2D_dim(as_dataset):

    da_structured, da_unstructured = data_2D_dims(as_dataset)

    dims = {"x_dim": "x", "y_dim": "y"}

    coords_orig = da_structured.coords.to_dataset()[["x", "y"]]
    print(coords_orig)

    result = mxu.to_unstructured(
        mxu.from_unstructured(da_unstructured, coords_orig, **dims), **dims
    )
    xr.testing.assert_identical(result, da_unstructured)
