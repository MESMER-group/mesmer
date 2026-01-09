import pandas as pd
import xarray as xr

from mesmer._core.utils import _assert_required_coords, _check_dataarray_form
from mesmer.datatree import _datatree_wrapper


@_datatree_wrapper
def upsample_yearly_data(
    yearly_data: xr.DataArray | xr.Dataset | xr.DataTree,
    monthly_time: xr.DataArray | xr.Dataset | xr.DataTree,
    time_dim: str = "time",
):
    """Upsample yearly data to monthly resolution by repeating yearly values.

    Parameters
    ----------
    yearly_data : xarray.DataArray | xr.Dataset | xr.DataTree
        Yearly values to upsample.

    monthly_time : xarray.DataArray | xr.Dataset | xr.DataTree
        Monthly time used to define the time coordinates of the upsampled data.

    time_dim : str, default: 'time'
        Name of the time dimension.

    Returns
    -------
    upsampled_yearly_data: xarray.DataArray
        Upsampled yearly temperature values containing the yearly values for every month
        of the corresponding year.
    """

    _assert_required_coords(yearly_data, "yearly_data", required_coords=time_dim)
    _assert_required_coords(monthly_time, "monthly_time", required_coords=time_dim)

    # read out time coords - this also works if it's already time coords
    monthly_time = monthly_time[time_dim]
    _check_dataarray_form(monthly_time, "monthly_time", ndim=1)

    if yearly_data[time_dim].size * 12 != monthly_time.size:
        raise ValueError(
            "Length of monthly time not equal to 12 times the length of yearly data."
        )

    # we need to pass the dim (`time_dim` may be a no-dim-coordinate)
    # i.e., time_dim and sample_dim may or may not be the same
    (sample_dim,) = monthly_time.dims

    if isinstance(yearly_data.indexes.get(sample_dim), pd.MultiIndex):
        raise ValueError(
            f"The dimension of the time coords ({sample_dim}) is a pandas.MultiIndex,"
            " which is currently not supported. Potentially call"
            f" `yearly_data.reset_index('{sample_dim}')` first."
        )

    upsampled_yearly_data = (
        # repeats the data along new dimension
        yearly_data.expand_dims({"__new__": 12})
        # stack to remove new dim; target dim must have new name
        .stack(__sample__=(sample_dim, "__new__"), create_index=False)
        # so we need to rename it back
        .swap_dims(__sample__=sample_dim)
        # and ensure the time coords the ones from the monthly data
        .assign_coords({time_dim: (sample_dim, monthly_time.values)})
    )

    return upsampled_yearly_data
