import operator

import xarray as xr
from mesmer.core._datatreecompat import DataTree, map_over_datasets


def calc_anomaly(
    dt: DataTree, reference_period: slice, *, time_dim="time", ref_scenario="historical"
) -> DataTree:
    """subtract mean over the reference period

    Parameters
    ----------
    dt : DataTree
        Data to to calculate anomalies from. Must be a DataTree object which contains
        the historical scenario as node and may contain several projections. Individual
        ensmble members must be on the scenario nodes and
    reference_period : slice(str, str)
        Reference period, e.g. ``slice("1850", "1900")``.
    time_dim : str, default: "time"
        Name of the time dimension.
    ref_scenario : str, default: "historical"
        Name of the node containing the reference scenario.


    Returns
    -------
    anomalies: DataTree
        ``dt`` with the reference period subtracted.

    Notes
    -----
    - subtracts the reference of each individual ensmble member""

    """

    # NOTE: this corresponds to `ref["type"] == "individ"`

    if ref_scenario not in dt.children:
        raise ValueError(f"The ref_scenario ({ref_scenario}) is missing from `dt`")

    # calculate anomalies w.r.t. the reference period
    ref = dt[ref_scenario].sel({time_dim: reference_period})

    if ref[time_dim].size == 0:
        raise ValueError("No data selected for reference period")

    ref = ref.mean(time_dim)

    # https://github.com/pydata/xarray/issues/10013
    # anomalies = dt - ref.ds
    anomalies = map_over_datasets(operator.sub, dt, ref.ds)

    _assert_same_coords(dt, anomalies, ref_scenario)

    return anomalies


def _assert_same_coords(ref, anom, ref_scenario):


    for path, (ref_scen, anom_scen) in xr.group_subtrees(ref, anom):
        if not ref_scen.coords.equals(anom_scen.coords):

            msg = (
                f"Subtracting the reference changed the coordinates for '{path}'. "
                f"Most likely because the ref_scenario ({ref_scenario}) is missing "
                "some ensemble members.\n"
            )

            raise ValueError(msg)

    return ref
