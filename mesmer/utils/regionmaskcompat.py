# code vendored from regionmask under the conditions of their license
# see licenses/REGIONMASK_LICENSE

import numpy as np
import xarray as xr


def mask_percentage(regions, lon, lat):
    """Sample with 10 times higher resolution.

    Notes
    -----
    - assumes equally-spaced lat & lon!
    - copied from Mathias Hauser: https://github.com/mathause/regionmask/issues/38 in
      August 2020
    - prototype of what will eventually be integrated in his regionmask package

    """

    lon_sampled = sample_coord(lon)
    lat_sampled = sample_coord(lat)

    mask = regions.mask(lon_sampled, lat_sampled)

    isnan = np.isnan(mask.values)

    numbers = np.unique(mask.values[~isnan])
    numbers = numbers.astype(np.int)

    mask_sampled = list()
    for num in numbers:
        # coarsen the mask again
        mask_coarse = (mask == num).coarsen(lat=10, lon=10).mean()
        mask_sampled.append(mask_coarse)

    mask_sampled = xr.concat(
        mask_sampled, dim="region", compat="override", coords="minimal"
    )

    mask_sampled = mask_sampled.assign_coords(region=("region", numbers))

    return mask_sampled


def sample_coord(coord):
    """Sample coords for the percentage overlap.

    Notes
    -----
    - copied from Mathias Hauser: https://github.com/mathause/regionmask/issues/38
      in August 2020
    -> prototype of what will eventually be integrated in his regionmask package

    """
    d_coord = coord[1] - coord[0]

    n_cells = len(coord)

    left = coord[0] - d_coord / 2 + d_coord / 20
    right = coord[-1] + d_coord / 2 - d_coord / 20

    return np.linspace(left, right, n_cells * 10)
