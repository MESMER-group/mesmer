import numpy as np
import pandas as pd
import pytest
import xarray as xr
from packaging.version import Version

import mesmer.core.utils


@pytest.mark.parametrize(
    "values, expected",
    [((5, 4, 3, 2, 3, 0), 3), ((0, 1, 2), 0)],
)
def test_minimize_local_discrete(values, expected):

    data_dict = {key: value for key, value in enumerate(values)}

    def func(i):
        return data_dict[i]

    result = mesmer.core.utils._minimize_local_discrete(func, data_dict.keys())

    assert result == expected


def test_create_equal_dim_names():

    with pytest.raises(ValueError, match="must provide exactly two suffixes"):
        mesmer.core.utils.create_equal_dim_names("dim", "a")

    result = mesmer.core.utils.create_equal_dim_names("dim", (".1", ".2"))
    assert result == ("dim.1", "dim.2")


def test_minimize_local_discrete_warning():
    def func(i, data_dict):
        return data_dict[i]

    data_dict = {key: value for key, value in enumerate((3, 2, 1))}

    with pytest.warns(mesmer.core.utils.OptimizeWarning, match="No local minimum"):
        result = mesmer.core.utils._minimize_local_discrete(
            func, data_dict.keys(), data_dict=data_dict
        )

    assert result == 2

    data_dict = {key: value for key, value in enumerate((5, np.inf, 3))}

    with pytest.warns(mesmer.core.utils.OptimizeWarning, match="`fun` returned `inf`"):
        result = mesmer.core.utils._minimize_local_discrete(
            func, data_dict.keys(), data_dict=data_dict
        )

    assert result == 0


def test_minimize_local_discrete_error():
    def func(i):
        return float("-inf")

    with pytest.raises(ValueError, match=r"`fun` returned `\-inf`"):
        mesmer.core.utils._minimize_local_discrete(func, [0])


@pytest.mark.parametrize("obj", (None, xr.DataArray()))
def test_check_dataset_form_wrong_type(obj):

    with pytest.raises(TypeError, match="Expected obj to be an xr.Dataset"):
        mesmer.core.utils._check_dataset_form(obj)

    with pytest.raises(TypeError, match="Expected test to be an xr.Dataset"):
        mesmer.core.utils._check_dataset_form(obj, name="test")


def test_check_dataset_form_required_vars():

    ds = xr.Dataset()

    with pytest.raises(ValueError, match="obj is missing the required data_vars"):
        mesmer.core.utils._check_dataset_form(ds, required_vars="missing")

    with pytest.raises(ValueError, match="test is missing the required data_vars"):
        mesmer.core.utils._check_dataset_form(ds, "test", required_vars="missing")

    # no error
    mesmer.core.utils._check_dataset_form(ds)
    mesmer.core.utils._check_dataset_form(ds, required_vars=set())
    mesmer.core.utils._check_dataset_form(ds, required_vars=None)

    ds = xr.Dataset(data_vars={"var": ("x", [0])})

    # no error
    mesmer.core.utils._check_dataset_form(ds)
    mesmer.core.utils._check_dataset_form(ds, required_vars="var")
    mesmer.core.utils._check_dataset_form(ds, required_vars={"var"})


def test_check_dataset_form_requires_other_vars():

    ds = xr.Dataset()

    with pytest.raises(ValueError, match="Expected additional variables on obj"):
        mesmer.core.utils._check_dataset_form(ds, requires_other_vars=True)

    with pytest.raises(ValueError, match="Expected additional variables on test"):
        mesmer.core.utils._check_dataset_form(ds, "test", requires_other_vars=True)

    with pytest.raises(ValueError, match="Expected additional variables on obj"):
        mesmer.core.utils._check_dataset_form(
            ds, optional_vars="var", requires_other_vars=True
        )

    ds = xr.Dataset(data_vars={"var": ("x", [0])})

    with pytest.raises(ValueError, match="Expected additional variables on obj"):
        mesmer.core.utils._check_dataset_form(
            ds, required_vars="var", requires_other_vars=True
        )

    with pytest.raises(ValueError, match="Expected additional variables on obj"):
        mesmer.core.utils._check_dataset_form(
            ds, optional_vars="var", requires_other_vars=True
        )


@pytest.mark.parametrize("obj", (None, xr.Dataset()))
def test_check_dataarray_form_wrong_type(obj):

    with pytest.raises(TypeError, match="Expected obj to be an xr.DataArray"):
        mesmer.core.utils._check_dataarray_form(obj)

    with pytest.raises(TypeError, match="Expected test to be an xr.DataArray"):
        mesmer.core.utils._check_dataarray_form(obj, name="test")


@pytest.mark.parametrize("ndim", (0, 1, 3))
def test_check_dataarray_form_ndim(ndim):

    da = xr.DataArray(np.ones((2, 2)))

    with pytest.raises(ValueError, match=f"obj should be {ndim}D"):
        mesmer.core.utils._check_dataarray_form(da, ndim=ndim)

    with pytest.raises(ValueError, match=f"test should be {ndim}D"):
        mesmer.core.utils._check_dataarray_form(da, ndim=ndim, name="test")

    # no error
    mesmer.core.utils._check_dataarray_form(da, ndim=2)


def test_check_dataarray_form_ndim_several():

    da = xr.DataArray(np.ones((2, 2)))

    with pytest.raises(ValueError, match="obj should be 1D or 3D"):
        mesmer.core.utils._check_dataarray_form(da, ndim=(1, 3))

    with pytest.raises(ValueError, match="test should be 0D, 1D or 3D"):
        mesmer.core.utils._check_dataarray_form(da, ndim=(0, 1, 3), name="test")


@pytest.mark.parametrize("required_dims", ("foo", ["foo"], ["foo", "bar"]))
def test_check_dataarray_form_required_dims(required_dims):

    da = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))

    with pytest.raises(ValueError, match="obj is missing the required dims"):
        mesmer.core.utils._check_dataarray_form(da, required_dims=required_dims)

    with pytest.raises(ValueError, match="test is missing the required dims"):
        mesmer.core.utils._check_dataarray_form(
            da, required_dims=required_dims, name="test"
        )

    # no error
    mesmer.core.utils._check_dataarray_form(da, required_dims="x")
    mesmer.core.utils._check_dataarray_form(da, required_dims="y")
    mesmer.core.utils._check_dataarray_form(da, required_dims=["x", "y"])
    mesmer.core.utils._check_dataarray_form(da, required_dims={"x", "y"})


def test_check_dataarray_form_shape():

    da = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))

    for shape in ((), (1,), (1, 2), (2, 1), (1, 2, 3)):
        with pytest.raises(ValueError, match="obj has wrong shape"):
            mesmer.core.utils._check_dataarray_form(da, shape=shape)

    with pytest.raises(ValueError, match="test has wrong shape"):
        mesmer.core.utils._check_dataarray_form(da, name="test", shape=())

    # no error
    mesmer.core.utils._check_dataarray_form(da, shape=(2, 2))


def _get_time(*args, **kwargs):
    # TODO: use xr.date_range once requiring xarray >= v0.21

    calendar = kwargs.pop("calendar", "standard")
    freq = kwargs.pop("freq", None)

    # translate freq strings
    pandas_calendars = ["standard", "gregorian"]
    if freq:
        if calendar not in pandas_calendars and Version(xr.__version__) < Version(
            "2023.11"
        ):
            freq = freq.replace("Y", "A").replace("ME", "M")
        if calendar in pandas_calendars and Version(pd.__version__) < Version("2.2"):
            freq = freq.replace("Y", "A").replace("ME", "M")

    if Version(xr.__version__) >= Version("0.21"):
        time = xr.date_range(*args, calendar=calendar, freq=freq, **kwargs)
    else:

        if calendar == "standard":
            time = pd.date_range(*args, freq=freq, **kwargs)
        else:
            time = xr.cftime_range(*args, calendar=calendar, freq=freq, **kwargs)

    return xr.DataArray(time, dims="time")


@pytest.mark.parametrize(
    "calendar", ["standard", "gregorian", "proleptic_gregorian", "365_day", "julian"]
)
def test_assert_annual_data(calendar):

    time = _get_time("2000", "2005", freq="Y", calendar=calendar)

    # no error
    mesmer.core.utils._assert_annual_data(time)


@pytest.mark.parametrize("calendar", ["standard", "gregorian", "365_day"])
@pytest.mark.parametrize("freq", ["2Y", "ME"])
def test_assert_annual_data_wrong_freq(calendar, freq):

    time = _get_time("2000", periods=5, freq=freq, calendar=calendar)

    with pytest.raises(
        ValueError, match="Annual data is required but data with frequency"
    ):
        mesmer.core.utils._assert_annual_data(time)


def test_assert_annual_data_unkown_freq():

    time1 = _get_time("2000", periods=2, freq="Y")
    time2 = _get_time("2002", periods=3, freq="ME")
    time = xr.concat([time1, time2], dim="time")

    with pytest.raises(ValueError, match="Annual data is required but data of unknown"):
        mesmer.core.utils._assert_annual_data(time)
