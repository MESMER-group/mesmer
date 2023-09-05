import numpy as np
import xarray as xr

from mesmer.core.utils import _check_dataarray_form


def lowess(data, dim, *, frac, use_coords_as_x=False, it=0):
    """LOWESS (Locally Weighted Scatterplot Smoothing) for xarray objects

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Data to smooth (y-values).
    dim : str
        Dimension along which to smooth (x-dimension)
    frac : float
        Between 0 and 1. The fraction of the data used when estimating each y-value.
    use_coords_as_x : boolean, default: False
        If True uses ``data[dim]`` as x-values else uses ``np.arange(data[dim].size)``
        (useful if ``dim`` are time coordinates).
    it : int, default: 0
        The number of residual-based re-weightings to perform.
    """

    from statsmodels.nonparametric.smoothers_lowess import lowess

    if not isinstance(dim, str):
        raise ValueError("Can only pass a single dimension.")

    coords = data[dim]
    _check_dataarray_form(coords, name=dim, ndim=1)

    if use_coords_as_x:
        x = coords
    else:
        x = xr.ones_like(coords)
        x.data = np.arange(coords.size)

    def _lowess(da: xr.DataArray) -> xr.DataArray:

        # skip data var if input_core_dim is missing
        if dim not in da.dims:
            return da

        out = xr.apply_ufunc(
            lowess,
            da,
            x,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[dim]],
            vectorize=True,
            kwargs={"frac": frac, "it": it, "return_sorted": False},
        )

        return out

    if isinstance(data, xr.Dataset):
        return data.map(_lowess)

    return _lowess(data)
