import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

import mesmer
from mesmer._core.utils import _check_dataarray_form, _check_dataset_form
from mesmer.datatree import _datatree_wrapper, map_over_datasets
from mesmer.testing import trend_data_1D, trend_data_2D


def test_collapse_datatree_into_dataset():
    n_ts = 30
    ds1 = xr.Dataset({"tas": trend_data_1D(n_timesteps=n_ts)})
    ds2 = ds1 * 2
    ds3 = ds1 * 3

    dim = xr.Variable("member", np.arange(3))
    leaf1 = xr.concat([ds1, ds2, ds3], dim=dim)
    dim = xr.Variable("member", np.arange(2))
    leaf2 = xr.concat([ds1, ds2], dim=dim)

    dt = xr.DataTree.from_dict({"scen1": leaf1, "scen2": leaf2})

    collapse_dim = "scenario"
    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)

    assert isinstance(res, xr.Dataset)
    assert collapse_dim in res.dims
    assert (res[collapse_dim] == ["scen1", "scen2"]).all()
    assert len(res.dims) == 3
    assert np.isnan(res.sel(scenario="scen2", member=2)).all()

    # error if data set has no coords along dim (bc then it is not concatenable if lengths differ)
    leaf_missing_coords = leaf1.drop_vars("member")
    dt = xr.DataTree.from_dict({"scen1": leaf_missing_coords, "scen2": leaf2})
    with pytest.raises(
        ValueError, match="cannot reindex or align along dimension 'member'"
    ):
        res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)

    # Dimension along which to concatenate already exists
    leaf1_scen = leaf1.assign_coords({"scenario": "scen1"}).expand_dims(collapse_dim)
    leaf2_scen = leaf2.assign_coords({"scenario": "scen2"}).expand_dims(collapse_dim)
    dt = xr.DataTree.from_dict({"scen1": leaf1_scen, "scen2": leaf2_scen})

    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)
    assert isinstance(res, xr.Dataset)

    scen1 = res.sel(scenario="scen1")
    xr.testing.assert_equal(scen1.drop_vars("scenario"), leaf1)

    # only one leaf works
    dt = xr.DataTree.from_dict({"scen1": leaf1})
    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)

    assert isinstance(res, xr.Dataset)
    assert collapse_dim in res.dims
    assert (res[collapse_dim] == ["scen1"]).all()
    assert len(res.dims) == 3

    xr.testing.assert_equal(scen1.drop_vars(collapse_dim), leaf1)

    # test data in root works
    dt = xr.DataTree(leaf1, name="scen1")
    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)

    assert isinstance(res, xr.Dataset)
    assert collapse_dim in res.dims
    assert (res[collapse_dim] == ["scen1"]).all()
    assert len(res.dims) == 3

    xr.testing.assert_equal(scen1.drop_vars(collapse_dim), leaf1)

    # nested DataTree works
    dt = xr.DataTree()
    dt["scen1/sub_scen1"] = xr.DataTree(leaf1)
    dt["scen1/sub_scen2"] = xr.DataTree(leaf2)
    dt["scen2"] = xr.DataTree(leaf2)

    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)
    assert isinstance(res, xr.Dataset)
    assert collapse_dim in res.dims
    assert len(res.dims) == 3
    assert set(res[collapse_dim].values) == {"sub_scen1", "sub_scen2", "scen2"}

    # more than one datavariable - works and fills with nans if necessary
    ds = ds3.rename({"tas": "tas2"})

    leaf3 = xr.merge(
        [ds1.assign_coords({"member": 1}), ds.assign_coords({"member": 1})],
        compat="override",
    ).expand_dims("member")
    dt = xr.DataTree.from_dict({"scen1": leaf1, "scen2": leaf2, "scen3": leaf3})

    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)
    assert isinstance(res, xr.Dataset)
    assert collapse_dim in res.dims
    assert len(res.dims) == 3
    assert (res[collapse_dim] == ["scen1", "scen2", "scen3"]).all()
    assert len(res.data_vars) == 2
    assert np.isnan(res.sel(scenario="scen1").tas2).all()

    # two time dimensions that have different length fills missing values with nans
    ds_with_different_time = ds1.shift(time=1)

    badleaf = ds_with_different_time.assign_coords({"member": 0}).expand_dims("member")
    dt = xr.DataTree.from_dict({"scen1": leaf1, "scen2": badleaf})

    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)

    assert np.isnan(res.sel(scenario="scen2", time=leaf1.time)).all()

    # make sure it also works with stacked dimension
    # NOTE: only works if the stacked dimension has the same size on all datasets
    n_lat, n_lon = 2, 3
    da1 = mesmer.testing.trend_data_2D(n_timesteps=n_ts, n_lat=n_lat, n_lon=n_lon)
    ds1 = xr.Dataset({"tas": da1})
    da2 = mesmer.testing.trend_data_2D(n_timesteps=n_ts, n_lat=n_lat, n_lon=n_lon)
    ds2 = xr.Dataset({"tas": da2})

    dt = xr.DataTree.from_dict({"mem1": ds1, "mem2": ds2})
    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim="members")

    # empty nodes are removed before concatenating
    # NOTE: implicitly this is already there in the other tests, since the root node is always empty
    # but it is nice to have it explicitly too
    dt = xr.DataTree.from_dict({"scen1": leaf1, "scen2": xr.DataTree()})
    res = mesmer.datatree.collapse_datatree_into_dataset(dt, dim=collapse_dim)
    expected = leaf1.expand_dims(collapse_dim).assign_coords(
        {collapse_dim: np.array(["scen1"])}
    )
    xr.testing.assert_equal(res, expected)


def test_extract_single_dataarray_from_dt():
    da = trend_data_1D(n_timesteps=30).rename("tas")
    dt = xr.DataTree.from_dict({"/": xr.Dataset({"tas": da})})

    res = mesmer.datatree._extract_single_dataarray_from_dt(dt)
    xr.testing.assert_equal(res, da)

    dt = xr.DataTree(xr.Dataset({"tas": da, "tas2": da}))
    with pytest.raises(
        ValueError,
        match="Node must only contain one data variable, node has tas2 and tas.",
    ):
        mesmer.datatree._extract_single_dataarray_from_dt(dt)

    dt = xr.DataTree.from_dict(
        {"scen1": xr.Dataset({"tas": da, "tas2": da}), "scen2": xr.Dataset({"tas": da})}
    )

    # passing empty root
    with pytest.raises(ValueError, match="node has no data."):
        mesmer.datatree._extract_single_dataarray_from_dt(dt)

    # check name
    with pytest.raises(
        ValueError,
        match="Node must only contain one data variable, scen1 has tas2 and tas.",
    ):
        mesmer.datatree._extract_single_dataarray_from_dt(dt["scen1"], name="scen1")

    res = mesmer.datatree._extract_single_dataarray_from_dt(dt["scen2"])
    xr.testing.assert_equal(res, da)

    # passing empty Dataree
    with pytest.raises(ValueError, match="node has no data."):
        mesmer.datatree._extract_single_dataarray_from_dt(xr.DataTree())


def test_broadcast_and_pool_scen_ens():
    n_ts, n_lat, n_lon = 30, 2, 3
    member_dim = "member"
    time_dim = "time"
    scenario_dim = "scenario"
    stacked_dim = "sample"

    d2D_1 = xr.Dataset(
        {"tas": trend_data_2D(n_timesteps=n_ts, n_lat=n_lat, n_lon=n_lon)}
    )
    d2D_2 = d2D_1 * 2
    d2D_3 = d2D_1 * 3
    d2D_4 = d2D_1 * 4
    d2D_5 = d2D_1 * 5

    leaf1 = xr.concat([d2D_1, d2D_2, d2D_3], dim=member_dim).assign_coords(
        {member_dim: np.arange(3)}
    )
    leaf2 = xr.concat([d2D_4, d2D_5], dim=member_dim).assign_coords(
        {member_dim: np.arange(2)}
    )

    target = xr.DataTree.from_dict({"scen1": leaf1, "scen2": leaf2})

    d1D_1 = xr.Dataset(
        {
            "tas": trend_data_1D(n_timesteps=n_ts),
            "tas2": trend_data_1D(n_timesteps=n_ts) ** 2,
            "hfds": trend_data_1D(n_timesteps=n_ts) * 0.5,
        }
    )
    d1D_2 = d1D_1 * 2

    predictors = xr.DataTree.from_dict({"scen1": d1D_1, "scen2": d1D_2})

    weights = map_over_datasets(xr.ones_like, target.sel(cells=0))
    weights = map_over_datasets(
        lambda ds: ds.rename({var: "weights" for var in ds.data_vars}), weights
    )

    predictors_stacked, target_stacked, weights_stacked = (
        mesmer.datatree.broadcast_and_pool_scen_ens(
            predictors,
            target,
            weights,
            member_dim=member_dim,
            time_dim=time_dim,
            scenario_dim=scenario_dim,
            sample_dim=stacked_dim,
        )
    )

    n_samples = n_ts * (2 + 3)  # 2 members for scen1, 3 members for scen2

    _check_dataset_form(
        predictors_stacked,
        name="predictors",
        required_vars={"tas", "tas2", "hfds"},
    )
    for var in predictors_stacked.data_vars:
        da = predictors_stacked[var]
        _check_dataarray_form(
            da,
            name=da.name,
            ndim=1,
            required_dims={"sample"},
            shape=(n_samples,),
        )

    _check_dataarray_form(
        target_stacked.tas,
        ndim=2,
        required_dims={"cells", "sample"},
        shape=(n_samples, n_lat * n_lon),
    )
    _check_dataarray_form(
        weights_stacked.weights, ndim=1, required_dims={"sample"}, shape=(n_samples,)
    )

    # check if datasets align
    target_aligned, predictors_aligned = xr.align(
        target_stacked, predictors_stacked, join="exact"
    )
    xr.testing.assert_equal(target_stacked, target_aligned)
    xr.testing.assert_equal(predictors_stacked, predictors_aligned)

    target_aligned, weights_aligned = xr.align(
        target_stacked, weights_stacked, join="exact"
    )
    xr.testing.assert_equal(target_stacked, target_aligned)
    xr.testing.assert_equal(weights_stacked, weights_aligned)

    predictors_stacked, target_stacked, weights_stacked = (
        mesmer.datatree.broadcast_and_pool_scen_ens(predictors, target, None)
    )
    assert weights_stacked is None, "Weights should be None if not provided"

    # check if exclude_dim can be empty
    predictors_stacked, target_stacked, weights_stacked = (
        mesmer.datatree.broadcast_and_pool_scen_ens(
            predictors,
            target.sel(cells=0),
            weights,
        )
    )

    target_aligned, predictors_aligned = xr.align(
        target_stacked, predictors_stacked, join="exact"
    )
    xr.testing.assert_equal(target_stacked, target_aligned)
    xr.testing.assert_equal(predictors_stacked, predictors_aligned)


def test_datatree_wrapper_dt_kwarg_errors():

    @_datatree_wrapper
    def func(arg):
        return arg

    dt = xr.DataTree()

    with pytest.raises(TypeError, match="Passed a `DataTree` as keyword argument"):
        func(arg=dt)


def test_datatree_wrapper():

    @_datatree_wrapper
    def func(arg):
        assert isinstance(arg, xr.Dataset)
        return arg

    da = xr.DataArray([1, 2, 3], dims="x")
    ds = xr.Dataset(data_vars={"da": da})

    dt = xr.DataTree.from_dict({"node": ds})

    result_ds = func(ds)
    assert isinstance(result_ds, xr.Dataset)

    result_dt = func(dt)
    assert isinstance(result_dt, xr.DataTree)


@pytest.mark.parametrize("scenario_dim", ("scenario", "scen"))
@pytest.mark.parametrize("time_dim", ("time", "t"))
@pytest.mark.parametrize("member_dim", ("member", "m"))
@pytest.mark.parametrize("sample_dim", ("sample", "s"))
def test_pool_scen_ens(scenario_dim, time_dim, member_dim, sample_dim):

    time = np.arange(3)
    data = np.arange(6).reshape(2, 3).T
    da1 = xr.DataArray(
        data,
        dims=(time_dim, member_dim),
        coords={time_dim: time, member_dim: ["a1", "a2"]},
    )
    ds1 = xr.Dataset(data_vars={"var": da1})

    time = np.arange(2)
    data = np.arange(2).reshape(2, 1) * 10
    da2 = xr.DataArray(
        data, dims=(time_dim, member_dim), coords={time_dim: time, member_dim: ["a3"]}
    )
    ds2 = xr.Dataset(data_vars={"var": da2})

    dt = xr.DataTree.from_dict({"scen1": ds1, "scen2": ds2})

    result = mesmer.datatree.pool_scen_ens(
        dt,
        member_dim=member_dim,
        time_dim=time_dim,
        scenario_dim=scenario_dim,
        sample_dim=sample_dim,
    )

    # =========
    data = np.concatenate([np.arange(6), np.arange(2) * 10])
    time = [0, 1, 2, 0, 1, 2, 0, 1]
    member = ["a1"] * 3 + ["a2"] * 3 + ["a3"] * 2
    scen = ["scen1"] * 6 + ["scen2"] * 2

    da = xr.DataArray(
        data,
        dims=sample_dim,
        coords={
            time_dim: (sample_dim, time),
            member_dim: (sample_dim, member),
            scenario_dim: (sample_dim, scen),
        },
    )
    expected = xr.Dataset(data_vars={"var": da})

    xr.testing.assert_equal(result, expected)


def test_pool_scen_ens_missing_member_dim():

    time = np.arange(2)
    data = np.arange(2)
    da = xr.DataArray(data, coords={"time": time})
    ds = xr.Dataset(data_vars={"var": da})
    dt = xr.DataTree.from_dict({"scen": ds})

    with pytest.raises(
        ValueError, match=r"`member_dim` \('member'\) not available in node 'scen'"
    ):
        mesmer.datatree.pool_scen_ens(dt)


def test_pool_scen_ens_no_member_dim():

    time = np.arange(2)
    data = np.arange(2)
    da = xr.DataArray(data, coords={"time": time})
    ds = xr.Dataset(data_vars={"var": da})

    dt = xr.DataTree.from_dict({"scen": ds})

    result = mesmer.datatree.pool_scen_ens(dt, member_dim=None)

    # =========
    scen = ["scen"] * 2

    da = xr.DataArray(
        data,
        dims="sample",
        coords={
            "time": ("sample", time),
            "scenario": ("sample", scen),
        },
    )
    expected = xr.Dataset(data_vars={"var": da})

    xr.testing.assert_equal(result, expected)


def test_pool_scen_ens_keep_other_dims():

    time = np.arange(2)
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    da = xr.DataArray(
        data,
        dims=("time", "member", "gridpoint"),
        coords={"time": time, "member": ["a1", "a2", "a3"]},
    )
    ds1 = xr.Dataset(data_vars={"var": da})

    time = np.arange(3)
    data = np.arange(3 * 4).reshape(3, 1, 4)
    da2 = xr.DataArray(
        data,
        dims=("time", "member", "gridpoint"),
        coords={"time": time, "member": ["a3"]},
    )
    ds2 = xr.Dataset(data_vars={"var": da2})

    dt = xr.DataTree.from_dict({"scen1": ds1, "scen2": ds2})

    result = mesmer.datatree.pool_scen_ens(dt)

    mesmer._core.utils._check_dataset_form(result, "result", required_vars="var")
    mesmer._core.utils._check_dataarray_form(
        result["var"],
        "result.var",
        ndim=2,
        required_dims=("sample", "gridpoint"),
        shape=(9, 4),
    )


@pytest.mark.parametrize("scenario_dim", ("scenario", "scen"))
@pytest.mark.parametrize("time_dim", ("time", "t"))
@pytest.mark.parametrize("member_dim", ("member", "m"))
@pytest.mark.parametrize("sample_dim", ("sample", "s"))
def test_unpool_scen_ens(scenario_dim, time_dim, member_dim, sample_dim):

    time = np.arange(3)
    data = np.arange(6).reshape(2, 3).T
    da1 = xr.DataArray(
        data,
        dims=(time_dim, member_dim),
        coords={time_dim: time, member_dim: ["a1", "a2"]},
    )
    ds1 = xr.Dataset(data_vars={"var": da1})

    time = np.arange(2)
    data = np.arange(2).reshape(2, 1) * 10
    da2 = xr.DataArray(
        data, dims=(time_dim, member_dim), coords={time_dim: time, member_dim: ["a3"]}
    )
    ds2 = xr.Dataset(data_vars={"var": da2})

    dt = xr.DataTree.from_dict({"scen1": ds1, "scen2": ds2})

    pooled = mesmer.datatree.pool_scen_ens(
        dt,
        member_dim=member_dim,
        time_dim=time_dim,
        scenario_dim=scenario_dim,
        sample_dim=sample_dim,
    )

    result = mesmer.datatree._unpool_scen_ens(
        pooled, scenario_dim=scenario_dim, sample_dim=sample_dim
    )
    expected = dt

    xr.testing.assert_equal(result, expected)


def test_unpool_scen_ens_error():

    obj = xr.DataArray([1, 2, 3])
    with pytest.raises(
        ValueError, match="Passed DataArray must have a '.name' attribute"
    ):
        mesmer.datatree._unpool_scen_ens(obj)


@pytest.mark.skipif(
    Version(xr.__version__) < Version("2025.07"),
    reason="pydata/xarray#10440 and pydata/xarray#10442 ",
)
@pytest.mark.parametrize("ascending_scen_order", [True, False])
def test_unpool_scen_ens_other_dims(ascending_scen_order):

    time = np.arange(2)
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    gridpoint = np.arange(10, 41, 10)

    da = xr.DataArray(
        data,
        dims=("time", "member", "gridpoint"),
        coords={"time": time, "member": ["a1", "a2", "a3"], "gridpoint": gridpoint},
    )
    ds1 = xr.Dataset(data_vars={"var": da})

    time = np.arange(3)
    data = np.arange(3 * 4).reshape(3, 1, 4)
    da2 = xr.DataArray(
        data,
        dims=("time", "member", "gridpoint"),
        coords={"time": time, "member": ["a3"], "gridpoint": gridpoint},
    )
    ds2 = xr.Dataset(data_vars={"var": da2})

    if ascending_scen_order:
        dt = xr.DataTree.from_dict({"scen1": ds1, "scen2": ds2})
    else:
        dt = xr.DataTree.from_dict({"scen2": ds1, "scen1": ds2})

    pooled = mesmer.datatree.pool_scen_ens(dt)

    result = mesmer.datatree._unpool_scen_ens(pooled)
    expected = dt

    xr.testing.assert_equal(result, expected, check_dim_order=False)

    if ascending_scen_order:
        assert dt.groups == ("/", "/scen1", "/scen2")
    else:
        assert dt.groups == ("/", "/scen2", "/scen1")


def test_merge():
    ts = 20
    time = np.arange(ts)
    data = np.arange(ts)
    da = xr.DataArray(data, coords={"time": time})
    ds1 = xr.Dataset(data_vars={"var1": da})
    ds2 = xr.Dataset(data_vars={"var2": da * 2})

    dt1 = xr.DataTree(ds1)
    dt2 = xr.DataTree(ds2)

    result = xr.merge([dt1, dt2])

    assert isinstance(result, xr.DataTree)

    expected_ds = xr.merge([ds1, ds2])
    xr.testing.assert_equal(result.to_dataset(), expected_ds)

    # test with multiple nodes
    dt1["scen2"] = xr.Dataset(data_vars={"var1": da * 3})
    dt2["scen2"] = xr.Dataset(data_vars={"var2": da * 4})

    result = xr.merge([dt1, dt2])

    xr.testing.assert_equal(
        result.to_dataset(),
        xr.merge([dt1.to_dataset(), dt2.to_dataset()]),
    )

    xr.testing.assert_equal(
        result["scen2"].to_dataset(),
        xr.merge([dt1["scen2"].to_dataset(), dt2["scen2"].to_dataset()]),
    )


def test_merge_compat():
    # copied and adjusted to DataTree from https://github.com/pydata/xarray/blob/main/xarray/tests/test_merge.py
    # used as general test that keyword arguments are passed correctly, could be extended, but not putting
    # more effort in hopes of a future xarray version that has the same interface for DataTree

    dt1 = xr.DataTree(xr.Dataset({"x": 0}))
    dt2 = xr.DataTree(xr.Dataset({"x": 1}))
    for compat in ["broadcast_equals", "equals", "identical", "no_conflicts"]:
        with pytest.raises(xr.MergeError):
            xr.merge([dt1, dt2], compat=compat)

    dt2 = xr.DataTree(xr.Dataset({"x": [0, 0]}))
    for compat in ["equals", "identical"]:
        with pytest.raises(ValueError, match=r"should be coordinates or not"):
            xr.merge([dt1, dt2], compat=compat)

    dt2 = xr.DataTree(xr.Dataset({"x": ((), 0, {"foo": "bar"})}))
    with pytest.raises(xr.MergeError):
        xr.merge([dt1, dt2], compat="identical")

    with pytest.raises(ValueError, match=r"compat=.* invalid"):
        xr.merge([dt1, dt2], compat="foobar")

    assert dt1.identical(xr.merge([dt1, dt2], compat="override"))


def test_map_over_dataset():
    # test empty nodes are skipped

    ds = xr.Dataset(data_vars={"data": ("x", [1, 2])})
    dt = xr.DataTree.from_dict({"node": ds})

    def rename(ds):
        return ds.rename(data="variable")

    result = map_over_datasets(rename, dt)
    expected = xr.DataTree.from_dict({"node": ds.rename(data="variable")})
    xr.testing.assert_equal(result, expected)

    # test not-first arg is a DataTree

    def rename_second(new_name, ds):
        return ds.rename(data=new_name)

    result = map_over_datasets(rename_second, "variable", dt)
    expected = xr.DataTree.from_dict({"node": ds.rename(data="variable")})
    xr.testing.assert_equal(result, expected)

    # test if there are only coords
    ds_coords = xr.Dataset(coords={"x": [1, 2]})
    dt = xr.DataTree.from_dict({"node": ds_coords})

    def rename_coords(ds):
        return ds.rename(x="y")

    result = map_over_datasets(rename_coords, dt)
    expected = xr.DataTree.from_dict({"node": ds_coords.rename(x="y")})
    xr.testing.assert_equal(result, expected)
