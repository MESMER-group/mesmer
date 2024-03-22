"""
MESMER-M utility functions
"""

import xarray as xr

def upsample_yearly_data(yearly_data, monthly_data):
    """Upsample yearly data to monthly data by repeating yearly value for each month.

    Parameters
    ----------
    yearly_data : xarray.DataArray
        Yearly values to upsample.

    monthly_data: xarray.DataArray
        Monthly values used to define the time dimension of the upsampled data.

    Returns
    -------
    upsampled_yearly_data: xarray.Dataset, xarray.DataArray
        Upsampled monthly temperature values containing the yearly values for every month of the corresponding year.
    """

    year = yearly_data.resample(time = "YS").bfill()
    month = monthly_data.resample(time = "MS").bfill()

    upsampled_yearly_data = year.reindex_like(month, method="ffill")
    upsampled_yearly_data = year.reindex_like(monthly_data, method="ffill")

    return upsampled_yearly_data