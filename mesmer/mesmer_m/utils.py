"""
MESMER-M utility functions
"""

import xarray as xr

def upsample_yearly_data(yearly_data, monthly_data):
    """Upsample yearly data to monthly data by repeating yearly mean for each month.

    Parameters
    ----------
    yearly_data : xarray.DataArray
        Yearly values to upsample.

    monthly_data: xarray.DataArray
        Monthly values used to define the time dimension of the upsampled data.

    Returns
    -------
    monthly_data: xarray.Dataset, xarray.DataArray
        Upsampled monthly temperature values.
    """

    ye = yearly_data.resample(time="YE").ffill()
    me = ye.resample(time="ME").bfill()

    # resample misses first year
    firstyear = monthly_data.isel(time=slice(0, 11)).resample(time="ME").ffill()
    firstyear.values[:,] = yearly_data.isel(time=0)

    year_to_mon_data = xr.concat([firstyear, me], dim="time")
    year_to_mon_data["time"] = monthly_data["time"]

    return year_to_mon_data