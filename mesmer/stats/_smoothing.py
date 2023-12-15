import numpy as np
import xarray as xr

from mesmer.core.utils import _check_dataarray_form


def lowess(
    data, dim, *, combine_dim=None, n_steps=None, frac=None, use_coords=True, it=0
):
    """LOWESS (Locally Weighted Scatterplot Smoothing) for xarray objects

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Data to smooth (y-values).
    dim : str
        Dimension along which to smooth (x-dimension)
    combine_dim : str, default: None
        Dimension along which to pool the data. This will stack the data and estimate
        the smoothing on the stacked data.
    n_steps : int
        The number of data points used to estimate each y-value, must be between 0 and
        the length of dim. If given used to calculate ``frac``. Exactly one of
        ``n_steps`` and ``frac`` must be given.
    frac : float
        The fraction of the data used when estimating each y-value. Between 0 and 1.
        Exactly one of ``n_steps`` and ``frac`` must be given.
    use_coords : boolean, default: True
        If True uses ``data[dim]`` as x-values else uses ``np.arange(data[dim].size)``
        (useful if ``dim`` are time coordinates).
    it : int, default: 0
        The number of residual-based re-weightings to perform.

    Returns
    -------
    out : xr.DataArray | xr.Dataset
        LOWESS smoothed array

    See Also
    --------
    statsmodels.nonparametric.smoothers_lowess.lowess

    Notes
    -----
    For ``it=0``, the following three options are equivalent::

        mesmer.stats.lowess(data.mean("cells"), "time", frac=0.3)
        mesmer.stats.lowess(data, "time", combine_dim="cells", frac=0.3)
        mesmer.stats.lowess(data, "time", frac=0.3).mean("cells")
    """

    from statsmodels.nonparametric.smoothers_lowess import lowess

    if not isinstance(dim, str):
        raise ValueError("Can only pass a single dimension.")

    if (n_steps is None and frac is None) or (n_steps is not None and frac is not None):
        raise ValueError("Exactly one of ``n_steps`` and ``frac`` must be given.")

    coords = data[dim]
    _check_dataarray_form(coords, name=dim, ndim=1)

    n_coords = coords.size
    if n_steps is not None:

        if n_steps > n_coords:
            raise ValueError(
                f"``n_steps`` ({n_steps}) cannot be be larger than the length of '{dim}' ({n_coords})"
            )

        frac = n_steps / n_coords

    # TODO: could instead convert datetime coords to numeric (see e.g. in flox)
    if use_coords:
        try:
            # test if coords can be cast to float (required by statsmodels..lowess)
            # use safe casting so we don't convert np datetime to float
            # while this technically works, the x values are then too large such that
            # a missing year is no longer detected...
            x = coords.astype(float, casting="safe")
        except TypeError as e:
            raise TypeError(
                f"Cannot convert coords ({dim}) of type `{coords.dtype}` to float. "
                "Set ``use_coords=False`` to use enumerated coords "
                "(``np.arange(data[dim].size)``) instead."
            ) from e

    else:
        x = xr.ones_like(coords)
        x.data = np.arange(coords.size)

    if combine_dim is not None:
        # remove non-dimension coords along combine_dims
        data = data.drop_vars(data[combine_dim].coords.keys())

        # need to broadcast and stack due to the datetime shenanigans above
        __, x = xr.broadcast(data, x)
        data = data.stack(__sample__=(dim, combine_dim))
        x = x.stack(__sample__=(dim, combine_dim))
        dim = "__sample__"

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

    result = data.map(_lowess) if isinstance(data, xr.Dataset) else _lowess(data)

    if combine_dim is not None:
        result = result.unstack()
        # all the estimates are the same, so we can use the first
        result = result.isel({combine_dim: 0}, drop=True)
        return result

    return result
