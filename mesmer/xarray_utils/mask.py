import numpy as np
import regionmask
import xarray as xr

import mesmer.utils


def _where_if_dim(obj, cond, dims):

    # xarray applies where to all data_vars - even if they do not have the corresponding
    # dimensions - we don't want that https://github.com/pydata/xarray/issues/7027

    def _where(da):
        if all(dim in da.dims for dim in dims):
            return da.where(cond)
        return da

    if isinstance(obj, xr.Dataset):
        return obj.map(_where, keep_attrs=True)

    return obj.where(cond)


def mask_ocean_fraction(obj, threshold, *, x_coords="lon", y_coords="lat"):
    """mask out ocean using fractional overlap

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Array to mask.
    threshold : float
        Threshold above which land fraction to consider a grid point as a land grid
        point. Must be must be between 0 and 1 inclusive.
    x_coords : str, default: "lon"
        Name of the x-coordinates.
    y_coords : str, default: "lat"
        Name of the y-coordinates.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array with ocean grid points masked out.

    Notes
    -----
    - Uses the 1:110m land mask from Natural Earth (http://www.naturalearthdata.com).
    - The fractional overlap of individual grid points and the land mask can only be
      computed for regularly-spaced 1D x- and y-coordinates. For irregularly spaced
      coordinates use :py:func:`mesmer.xarray_utils.mask_land`.
    """

    if np.ndim(threshold) != 0 or (threshold < 0) or (threshold > 1):
        raise ValueError("`threshold` must be a scalar between 0 and 1 (inclusive).")

    # TODO: allow other masks?
    land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110

    try:
        mask_fraction = mesmer.utils.regionmaskcompat.mask_3D_frac_approx(
            land_110, obj[x_coords], obj[y_coords]
        )
    except mesmer.utils.regionmaskcompat.InvalidCoordsError as e:
        raise ValueError(
            "Cannot calculate fractional mask for irregularly-spaced coords - please "
            "``mask_land`` instead."
        ) from e

    # drop region-specific coords
    mask_fraction = mask_fraction.squeeze(drop=True)

    mask_bool = mask_fraction > threshold

    # only mask data_vars that have the coords
    return _where_if_dim(obj, mask_bool, [y_coords, x_coords])


def mask_ocean(obj, *, x_coords="lon", y_coords="lat"):
    """mask out ocean

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Array to mask.
    x_coords : str, default: "lon"
        Name of the x-coordinates.
    y_coords : str, default: "lat"
        Name of the y-coordinates.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array with ocean grid points masked out.

    Notes
    -----
    - Uses the 1:110m land mask from Natural Earth (http://www.naturalearthdata.com).
    """

    # TODO: allow other masks?
    land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110

    mask_bool = land_110.mask_3D(obj[x_coords], obj[y_coords])

    mask_bool = mask_bool.squeeze(drop=True)

    # only mask data_vars that have the coords
    return _where_if_dim(obj, mask_bool, [y_coords, x_coords])


def mask_antarctica(obj, *, y_coords="lat"):
    """mask out ocean

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Array to mask.
    y_coords : str, default: "lat"
        Name of the y-coordinates.

    Returns
    -------
    obj : xr.Dataset | xr.DataArray
        Array with Antarctic grid points masked out.

    Notes
    -----
    - Masks grid points below 60°S.
    """

    mask_bool = obj[y_coords] >= -60

    # only mask if obj has y_coords
    return _where_if_dim(obj, mask_bool, [y_coords])
