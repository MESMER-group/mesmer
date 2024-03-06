import numpy as np
import regionmask
import xarray as xr

import mesmer


def _where_if_coords(obj, cond, coords):

    # xarray applies where to all data_vars - even if they do not have the corresponding
    # dimensions - we don't want that https://github.com/pydata/xarray/issues/7027

    def _where(da):
        if all(coord in da.coords for coord in coords):
            return da.where(cond)
        return da

    if isinstance(obj, xr.Dataset):
        return obj.map(_where, keep_attrs=True)

    return obj.where(cond)


def mask_ocean_fraction(data, threshold, *, x_coords="lon", y_coords="lat"):
    """mask out ocean using fractional overlap

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
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
    data : xr.Dataset | xr.DataArray
        Array with ocean grid points masked out.

    Notes
    -----
    - Uses the 1:110m land mask from Natural Earth (http://www.naturalearthdata.com).
    - The fractional overlap of individual grid points and the land mask can only be
      computed for regularly-spaced 1D x- and y-coordinates. For irregularly spaced
      coordinates use :py:func:`mesmer.mask.mask_land`.
    """

    if np.ndim(threshold) != 0 or (threshold < 0) or (threshold > 1):
        raise ValueError("`threshold` must be a scalar between 0 and 1 (inclusive).")

    # TODO: allow other masks?
    land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110

    try:
        mask_fraction = mesmer.core.regionmaskcompat.mask_3D_frac_approx(
            land_110, data[x_coords], data[y_coords]
        )
    except mesmer.core.regionmaskcompat.InvalidCoordsError as e:
        raise ValueError(
            "Cannot calculate fractional mask for irregularly-spaced coords - use "
            "``mask_land`` instead."
        ) from e

    # drop region-specific coords
    mask_fraction = mask_fraction.squeeze(drop=True)

    mask_bool = mask_fraction > threshold

    # only mask data_vars that have the coords
    return _where_if_coords(data, mask_bool, [y_coords, x_coords])


def mask_ocean(data, *, x_coords="lon", y_coords="lat"):
    """mask out ocean

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Array to mask.
    x_coords : str, default: "lon"
        Name of the x-coordinates.
    y_coords : str, default: "lat"
        Name of the y-coordinates.

    Returns
    -------
    data : xr.Dataset | xr.DataArray
        Array with ocean grid points masked out.

    Notes
    -----
    - Uses the 1:110m land mask from Natural Earth (http://www.naturalearthdata.com).
    - Whether a grid cell is in the ocean or on land is based on its center. For
      regularly spaced coordinates use :py:func:`mesmer.mask.mask_land_fraction`.
    """

    # TODO: allow other masks?
    land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110

    mask_bool = land_110.mask_3D(data[x_coords], data[y_coords])

    mask_bool = mask_bool.squeeze(drop=True)

    # only mask data_vars that have the coords
    return _where_if_coords(data, mask_bool, [y_coords, x_coords])


def mask_antarctica(data, *, y_coords="lat"):
    """mask out ocean

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Array to mask.
    y_coords : str, default: "lat"
        Name of the y-coordinates.

    Returns
    -------
    data : xr.Dataset | xr.DataArray
        Array with Antarctic grid points masked out.

    Notes
    -----
    - Masks grid points below 60°S.
    """

    mask_bool = data[y_coords] >= -60

    # only mask if data has y_coords
    return _where_if_coords(data, mask_bool, [y_coords])
