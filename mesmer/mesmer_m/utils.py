"""
MESMER-M utility functions
"""

import xarray as xr

def upsample_yearly_data(yearly_data, monthly_time):
    """Upsample yearly data to monthly data by repeating yearly value for each month.

    Parameters
    ----------
    yearly_data : xarray.DataArray
        Yearly values to upsample.

    monthly_time: xarray.DataArray
        Monthly time used to define the time dimension of the upsampled data.

    Returns
    -------
    upsampled_yearly_data: xarray.DataArray
        Upsampled monthly temperature values containing the yearly values for every month of the corresponding year.
    """
    # make sure monthly and yearly data both start at the beginning of the period
    year = yearly_data.resample(time = "YS").bfill()
    month = monthly_time.resample(time = "MS").bfill()

    # forward fill yearly values to monthly resolution
    upsampled_yearly_data = year.reindex_like(month, method="ffill")

    # make sure the time dimension of the upsampled data is the same as the original monthly time
    upsampled_yearly_data = year.reindex_like(monthly_time, method="ffill")

    return upsampled_yearly_data