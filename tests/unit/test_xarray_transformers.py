# test_sklearn_xarray_transformer.py
import numpy as np
import pytest
import xarray as xr
from sklearn.preprocessing import StandardScaler

from mesmer.stats._xarray_transformers import SklearnXarrayTransformer


# you used data from mesmer.testing for testing the power transformer,
# but none of the test data there is suitable for these tests.
# So I create some test data here first.
@pytest.fixture
def xr_data():
    """
    Deterministic test DataArray with:
    - sample_dim: year
    - feature_dim: gridcell
    - group dims: month, extra
    """
    rng = np.random.default_rng(0)
    data = rng.normal(size=(10, 4, 3, 2))

    return xr.DataArray(
        data,
        dims=("year", "gridcell", "month", "extra"),
        coords={
            "year": np.arange(10),
            "gridcell": np.arange(4),
            "month": np.arange(3),
            "extra": np.arange(2),
        },
        name="tas",
    )


# group_dims can be none, single, or multiple dimensions, the code should
# handle either case correctly.
@pytest.mark.parametrize(
    "group_dims",
    [
        None,
        ["month"],
        ["month", "extra"],
    ],
)
def test_roundtrip_standard_scaler(xr_data, group_dims):
    """
    ensuring that transform --> inverse_transform recovers the
    input up to floating point error. should work for no, single,
    and multiple group dimensions.
    """

    if group_dims is None:
        xr_data_test = xr_data.isel(month=0, extra=0)
    elif group_dims == ["month"]:
        xr_data_test = xr_data.isel(extra=0)
    else:
        xr_data_test = xr_data

    tr = SklearnXarrayTransformer(
        StandardScaler(),
        sample_dim="year",
        feature_dim="gridcell",
        group_dims=group_dims,
    )

    transformed = tr.fit_transform(xr_data_test)
    inverted = tr.inverse_transform(transformed)

    # note, this fails, if one dimension is a
    # multiindex, though applying np.allclose to the
    # actual values still succeeds.
    xr.testing.assert_allclose(inverted, xr_data_test, atol=1e-12)


@pytest.mark.parametrize(
    "group_dims",
    [
        None,
        ["month"],
        ["month", "extra"],
    ],
)
def test_dims_and_coords_preserved(xr_data, group_dims):
    """
    ensure dimensions and coordinates are preserved
    during transformation
    """
    tr = SklearnXarrayTransformer(
        StandardScaler(),
        sample_dim="year",
        feature_dim="gridcell",
        group_dims=group_dims,
    )

    if group_dims is None:
        xr_data_test = xr_data.isel(month=0, extra=0)
    elif group_dims == ["month"]:
        xr_data_test = xr_data.isel(extra=0)
    else:
        xr_data_test = xr_data

    out = tr.fit_transform(xr_data_test)

    assert out.dims == xr_data_test.dims
    for dim in xr_data_test.dims:
        xr.testing.assert_equal(out[dim], xr_data_test[dim])


def test_matches_sklearn_standard_scaler():
    """
    When no group_dims are specified, the transformer should produce
    results identical to applying sklearn's transformer to a 2D numpy array
    """

    # create test data
    rng = np.random.default_rng(1)
    da = xr.DataArray(
        rng.normal(size=(20, 5)),
        dims=("year", "gridcell"),
        coords={
            "year": np.arange(20),
            "gridcell": np.arange(5),
        },
        name="tas",
    )

    # expected sklearn baseline
    X = da.transpose("year", "gridcell").values
    sk = StandardScaler().fit(X)
    expected = sk.transform(X)

    # xarray transformer class
    tr = SklearnXarrayTransformer(
        StandardScaler(),
        sample_dim="year",
        feature_dim="gridcell",
    )

    out = tr.fit_transform(da)

    np.testing.assert_allclose(
        out.transpose("year", "gridcell").values,
        expected,
        atol=1e-12,
    )


def test_feature_coord_mismatch_raises(xr_data):
    """
    changing feature coordinates between fit and transform
    should raise an error
    """
    tr = SklearnXarrayTransformer(
        StandardScaler(),
        sample_dim="year",
        feature_dim="gridcell",
        group_dims=["extra", "month"],
    )

    tr.fit(xr_data)

    bad = xr_data.assign_coords(
        gridcell=np.arange(100, 100 + xr_data.sizes["gridcell"])
    )

    with pytest.raises(ValueError, match="Feature coordinates differ"):
        tr.transform(bad)


def test_transform_before_fit_raises(xr_data):
    """
    Calling transform before fit should raise an error.
    """
    tr = SklearnXarrayTransformer(
        StandardScaler(),
        sample_dim="year",
        feature_dim="gridcell",
    )

    with pytest.raises(ValueError, match="not fitted"):
        tr.transform(xr_data)


def test_get_params_as_xarray_mean_scale(xr_data):
    """
    get_params_as_xarray must return correctly-shaped DataArrays.
    should I also test this for multiple groupd_dims?
    """

    tr = SklearnXarrayTransformer(
        StandardScaler(),
        sample_dim="year",
        feature_dim="gridcell",
        group_dims=["month", "extra"],
    )

    tr.fit(xr_data)

    mean = tr.get_params_as_xarray("mean_")
    scale = tr.get_params_as_xarray("scale_")

    assert mean.dims == ("extra", "month", "gridcell")
    assert scale.dims == ("extra", "month", "gridcell")

    xr.testing.assert_equal(mean["gridcell"], xr_data["gridcell"])
    xr.testing.assert_equal(scale["gridcell"], xr_data["gridcell"])
