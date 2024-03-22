import datetime
import numpy as np
import pytest
import xarray as xr

from mesmer.mesmer_m import upsample_yearly_data

def make_dummy_yearly_data(freq):
    if freq == "YM":
        time = [datetime.datetime(i, 7, 15) for i in range(2000, 2005)]
    else:
        time = xr.cftime_range(start="2000", periods=5, freq=freq)
        
    data = xr.DataArray([1, 2, 3, 4, 5], dims=("time"), coords={"time": time})
    return data

def make_dummy_monthly_data(freq):
    if freq == "MM":
        time = [datetime.datetime(i, j, 15) for i in range(2000, 2005) for j in range(1, 13)]
    else:
        time = xr.cftime_range(start="2000-01", periods=5*12, freq=freq)
        
    data = xr.DataArray(np.ones(5*12), dims=("time"), coords={"time": time})
    return data

@pytest.mark.parametrize("freq_y", ["YM", "YS", "YE", "YS-JUL", "YS-NOV"])
@pytest.mark.parametrize("freq_m", ["MM", "MS", "ME"])
def test_upsample_yearly_data(freq_y, freq_m):
    yearly_data = make_dummy_yearly_data(freq_y)
    monthly_data = make_dummy_monthly_data(freq_m)
    
    upsampled_years = upsample_yearly_data(yearly_data, monthly_data)

    assert (upsampled_years.time == monthly_data.time).all()
    
    for i in range(len(yearly_data)):
        assert (upsampled_years.values[i*12:(i+1)*12] == yearly_data.values[i]).all()
