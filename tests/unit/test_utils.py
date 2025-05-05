import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mesmer.core.utils


def make_dummy_yearly_data(freq, calendar="standard"):

    # NOTE: "YM" is a made-up "Year-Middle" freq string
    if freq == "YM":
        time = xr.date_range(start="2000", periods=5, freq="YS-JUL", calendar=calendar)
        time = time + pd.Timedelta("14d")
    else:
        time = xr.date_range(start="2000", periods=5, freq=freq, calendar=calendar)

    data = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=("time"), coords={"time": time})
    return data


def make_dummy_monthly_data(freq, calendar="standard"):

    start = "2000-01"
    periods = 5 * 12

    # NOTE: "MM" is a made-up "Month-Middle" freq string
    if freq == "MM":
        time = xr.date_range(start=start, periods=periods, freq="MS", calendar=calendar)
        time = time + pd.Timedelta("14d")
    else:
        time = xr.date_range(start=start, periods=periods, freq=freq, calendar=calendar)

    data = xr.DataArray(np.arange(periods), dims=("time"), coords={"time": time})
    return data


@pytest.mark.parametrize("freq_y", ["YM", "YS", "YE", "YS-JUL", "YS-NOV"])
@pytest.mark.parametrize("freq_m", ["MM", "MS", "ME"])
@pytest.mark.parametrize("calendar", ["standard", "gregorian", "365_day"])
def test_upsample_yearly_data(freq_y, freq_m, calendar):

    yearly_data = make_dummy_yearly_data(freq_y, calendar=calendar)
    monthly_data = make_dummy_monthly_data(freq_m, calendar=calendar)

    upsampled_years = mesmer.core.utils.upsample_yearly_data(
        yearly_data, monthly_data.time
    )

    xr.testing.assert_equal(upsampled_years.time, monthly_data.time)

    # check if the values for each month are the same as the yearly values
    yearly_data = yearly_data.groupby("time.year").mean()
    assert (upsampled_years.groupby("time.year") == yearly_data).all()


def test_upsample_yearly_data_wrong_dims():
    yearly_data = make_dummy_yearly_data("YS")
    yearly_data = yearly_data.rename({"time": "year"})
    monthly_data = make_dummy_monthly_data("MM")

    with pytest.raises(ValueError, match="yearly_data is missing the required dims"):
        mesmer.core.utils.upsample_yearly_data(yearly_data, monthly_data.time)

    yearly_data = make_dummy_yearly_data("YS")
    monthly_data = monthly_data.rename({"time": "months"})
    with pytest.raises(ValueError, match="monthly_time is missing the required dims"):
        mesmer.core.utils.upsample_yearly_data(yearly_data, monthly_data.months)

    monthly_data = make_dummy_monthly_data("MM")
    monthly_data = monthly_data.expand_dims({"extra": 1})
    with pytest.raises(ValueError, match="monthly_time should be 1D, but is 2D"):
        mesmer.core.utils.upsample_yearly_data(yearly_data, monthly_data)


def test_upsample_yearly_data_wrong_length():
    yearly_data = make_dummy_yearly_data("YS")
    monthly_data = make_dummy_monthly_data("MM").isel(time=slice(0, 12 * 3))

    with pytest.raises(
        ValueError,
        match="Length of monthly time not equal to 12 times the length of yearly data.",
    ):
        mesmer.core.utils.upsample_yearly_data(yearly_data, monthly_data.time)


@pytest.mark.parametrize(
    "values, expected",
    [((5, 4, 3, 2, 3, 0), 3), ((1, 0, 1, 2), 1)],
)
def test_minimize_local_discrete(values, expected):

    data_dict = {key: value for key, value in enumerate(values)}

    def func(i):
        return data_dict[i]

    result = mesmer.core.utils._minimize_local_discrete(func, data_dict.keys())

    assert result == expected


def test_create_equal_dim_names():

    with pytest.raises(ValueError, match="must provide exactly two suffixes"):
        mesmer.core.utils._create_equal_dim_names("dim", "a")

    result = mesmer.core.utils._create_equal_dim_names("dim", (".1", ".2"))
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

    data_dict = {key: value for key, value in enumerate((1, 2, 3))}

    with pytest.warns(
        mesmer.core.utils.OptimizeWarning, match="First element is local minimum."
    ):
        result = mesmer.core.utils._minimize_local_discrete(
            func, data_dict.keys(), data_dict=data_dict
        )

    assert result == 0

    data_dict = {key: value for key, value in enumerate((5, 2, np.inf, 3))}

    with pytest.warns(mesmer.core.utils.OptimizeWarning, match="`fun` returned `inf`"):
        result = mesmer.core.utils._minimize_local_discrete(
            func, data_dict.keys(), data_dict=data_dict
        )

    assert result == 1


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

    calendar = kwargs.pop("calendar", "standard")
    freq = kwargs.pop("freq", None)

    time = xr.date_range(*args, calendar=calendar, freq=freq, **kwargs)

    return xr.DataArray(time, dims="time")


@pytest.mark.parametrize(
    "calendar", ["standard", "gregorian", "proleptic_gregorian", "365_day", "julian"]
)
def test_assert_annual_data(calendar):

    time = _get_time("2000", "2005", freq="YE", calendar=calendar)

    # no error
    mesmer.core.utils._assert_annual_data(time)


@pytest.mark.parametrize("calendar", ["standard", "gregorian", "365_day"])
@pytest.mark.parametrize("freq", ["2YE", "ME"])
def test_assert_annual_data_wrong_freq(calendar, freq):

    time = _get_time("2000", periods=5, freq=freq, calendar=calendar)

    with pytest.raises(
        ValueError, match="Annual data is required but data with frequency"
    ):
        mesmer.core.utils._assert_annual_data(time)


def test_assert_annual_data_unknown_freq():

    time1 = _get_time("2000", periods=2, freq="YE")
    time2 = _get_time("2002", periods=3, freq="ME")
    time = xr.concat([time1, time2], dim="time")

    with pytest.raises(ValueError, match="Annual data is required but data of unknown"):
        mesmer.core.utils._assert_annual_data(time)
