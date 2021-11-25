import numpy as np
import pyproj


def calculate_gaspari_cohn_correlation_matrix(
    latitudes,
    longitudes,
):
    """
    Calculate Gaspari-Cohn correlation matrix

    Parameters
    ----------
    latitudes : :obj:`xr.DataArray`
        Latitudes (one-dimensional)

    longitudes : :obj:`xr.DataArray`
        Longitudes (one-dimensional)
    """
    # I wonder if xarray can apply a function to all pairs of points in arrays
    # or something
    geodist = calculate_geodistance_exact(longitudes, latitudes)


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
    geodist : np.array
        2D array of great circle distances.
    """
    if longitudes.shape != latitudes.shape or longitudes.ndim != 1:
        raise ValueError("lon and lat need to be 1D arrays of the same shape")

    geod = pyproj.Geod(ellps="WGS84")

    n_points = longitudes.shape[0]

    geodist = np.zeros([n_points, n_points])

    # calculate only the upper right half of the triangle first
    for i in range(n_points):

        # need to duplicate gridpoint (required by geod.inv)
        lat = np.tile(latitudes[i], n_points - (i + 1))
        lon = np.tile(longitudes[i], n_points - (i + 1))

        geodist[i, i + 1 :] = geod.inv(lon, lat, longitudes[i + 1 :], latitudes[i + 1 :])[2]

    # convert m to km
    geodist /= 1000

    # fill the lower left half of the triangle (in-place)
    geodist += np.transpose(geodist)

    # should be able to keep co-ordinate information here too
    import pdb
    pdb.set_trace()

    return geodist
