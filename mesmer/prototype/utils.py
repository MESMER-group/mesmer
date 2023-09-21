import numpy as np
import pyproj
import xarray as xr


def calculate_gaspari_cohn_correlation_matrices(
    latitudes,
    longitudes,
    localisation_radii,
):
    """
    Calculate Gaspari-Cohn correlation matrices for a range of localisation radiis

    Parameters
    ----------
    latitudes : :obj:`xr.DataArray`
        Latitudes (one-dimensional)

    longitudes : :obj:`xr.DataArray`
        Longitudes (one-dimensional)

    localisation_radii : list-like
        Localisation radii to test (in metres)

    Returns
    -------
    dict[float: :obj:`xr.DataArray`]
        Gaspari-Cohn correlation matrix (values) for each localisation radius (keys)

    Notes
    -----
    Values in ``localisation_radii`` should not exceed 10'000 by much because
    it can lead to ``ValueError: the input matrix must be positive semidefinite``
    """
    # I wonder if xarray can apply a function to all pairs of points in arrays
    # or something
    geodistance = calculate_geodistance_exact(latitudes, longitudes)

    gaspari_cohn_correlation_matrices = {
        lr: calculate_gaspari_cohn_values(geodistance / lr) for lr in localisation_radii
    }

    return gaspari_cohn_correlation_matrices


def calculate_geodistance_exact(latitudes, longitudes):
    """
    Calculate exact great circle distance based on WSG 84

    Parameters
    ----------
    latitudes : :obj:`xr.DataArray`
        Latitudes (one-dimensional)

    longitudes : :obj:`xr.DataArray`
        Longitudes (one-dimensional)

    Returns
    -------
    :obj:`xr.DataArray`
        2D array of great circle distances between points represented by ``latitudes``
        and ``longitudes``
    """
    if longitudes.shape != latitudes.shape or longitudes.ndim != 1:
        raise ValueError("lon and lat need to be 1D arrays of the same shape")

    geod = pyproj.Geod(ellps="WGS84")

    n_points = longitudes.shape[0]

    geodistance = np.zeros([n_points, n_points])

    # calculate only the upper right half of the triangle first
    for i in range(n_points):

        # need to duplicate gridpoint (required by geod.inv)
        lat = np.tile(latitudes[i], n_points - (i + 1))
        lon = np.tile(longitudes[i], n_points - (i + 1))

        geodistance[i, i + 1 :] = geod.inv(
            lon, lat, longitudes.values[i + 1 :], latitudes.values[i + 1 :]
        )[2]

    # convert m to km
    geodistance /= 1000

    # fill the lower left half of the triangle (in-place)
    geodistance += np.transpose(geodistance)

    if latitudes.dims != longitudes.dims:
        raise AssertionError(
            f"latitudes and longitudes have different dims: {latitudes.dims} vs. {longitudes.dims}"
        )

    geodistance = xr.DataArray(
        geodistance, dims=list(latitudes.dims) * 2, coords=latitudes.coords
    )

    return geodistance


def calculate_gaspari_cohn_values(inputs):
    """
    Calculate smooth, exponentially decaying Gaspari-Cohn values

    Parameters
    ----------
    inputs : :obj:`xr.DataArray`
        Inputs at which to calculate the value of the smooth, exponentially decaying Gaspari-Cohn
        correlation function (these could be e.g. normalised geographical distances)

    Returns
    -------
    :obj:`xr.DataArray`
        Gaspari-Cohn correlation function applied to each point in ``inputs``
    """
    inputs_abs = abs(inputs)
    out = np.zeros_like(inputs)

    sel_zero_to_one = (inputs_abs.values >= 0) & (inputs_abs.values < 1)
    r_s = inputs_abs.values[sel_zero_to_one]
    out[sel_zero_to_one] = (
        1 - 5 / 3 * r_s**2 + 5 / 8 * r_s**3 + 1 / 2 * r_s**4 - 1 / 4 * r_s**5
    )

    sel_one_to_two = (inputs_abs.values >= 1) & (inputs_abs.values < 2)
    r_s = inputs_abs.values[sel_one_to_two]

    out[sel_one_to_two] = (
        4
        - 5 * r_s
        + 5 / 3 * r_s**2
        + 5 / 8 * r_s**3
        - 1 / 2 * r_s**4
        + 1 / 12 * r_s**5
        - 2 / (3 * r_s)
    )

    out = xr.DataArray(out, dims=inputs.dims, coords=inputs.coords)

    return out
