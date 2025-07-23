import warnings

import xarray as xr

from mesmer.core._data import load_stratospheric_aerosol_optical_depth_obs
from mesmer.core.datatree import _datatree_wrapper
from mesmer.core.utils import _assert_annual_data, _check_dataarray_form
from mesmer.stats import LinearRegression


def _load_and_align_strat_aod_obs(
    time: xr.DataArray, hist_period: slice | None, version="2022"
):
    """
    load stratospheric aerosol optical depth observations and align them to the to
    calendar and time of `time`.

    Parameters
    ----------
    time: xr.DataArray
        DataArray containing the time coords to align the aerosol optical depth
        observations to.
    hist_period : slice | None
        Slice object indicating the years of the historical period. E.g.
        ``slice("1850", "2014")``. If None uses the entire period in ``time``.
    version : str, default: "2022"
        Which version of the dataset to load. Currently only "2022" is available

    Returns
    -------
    aod : xr.DataArray
        stratospheric aerosol optical depth

    """

    aod = load_stratospheric_aerosol_optical_depth_obs(version=version, resample=True)

    if hist_period is None:
        beg, end = time[0].dt.year.item(), time[-1].dt.year.item()
        aod_beg, aod_end = aod.time[0].dt.year.item(), aod.time[-1].dt.year.item()
        if beg < aod_beg or end > aod_end:
            msg = (
                f"Time period of passed array ({beg}-{end}) exceeds time of stratospheric"
                f" aerosol optical depth observations ({aod_beg}-{aod_end}). Do you need"
                " to pass ``hist_period``?"
            )
            raise ValueError(msg)

        hist_period = slice(str(beg), str(end))

    aod = aod.sel(time=hist_period)

    # replace time axis of aod -> so they have the same calendar
    dim = time.name
    time_hist = time.sel({dim: hist_period})
    aod = aod.assign_coords({dim: time_hist})

    # expand aod to the full time period
    __, aod = xr.align(time, aod, fill_value=0.0, join="outer")

    return aod


def fit_volcanic_influence(
    tas_residuals, hist_period=None, *, dim="time", version="2022"
):
    """
    estimate volcanic influence on temperature residuals using aerosol optical depth
    observations as proxy

    Parameters
    ----------
    tas_residuals : xr.DataArray
        DataArray containing global mean temperature residual to estimate the volcanic
        influence from.
    hist_period : slice | None
        Slice object indicating the years of the historical period. E.g.
        ``slice("1850", "2014")``.  If None uses the entire time period of ``tas_residuals``.
    dim : str, default: "time"
        Dimension along which to estimate the volcanic influence.
    version : str, default: "2022"
        Which version of the aerosol optical depth observations to use. Currently only
        "2022" is valid.

    Returns
    -------
    params : xr.Dataset
        Parameters of the linear regression fit to the residuals.
    """

    _check_dataarray_form(
        tas_residuals, ndim=(1, 2), required_dims=dim, name="tas_residuals"
    )

    time = tas_residuals[dim]
    aod = _load_and_align_strat_aod_obs(time, hist_period, version=version)

    # TODO: extract this out of the function?
    if tas_residuals.ndim == 2:

        aod, __ = xr.broadcast(aod, tas_residuals)

        dims = tas_residuals.dims

        tas_residuals = tas_residuals.stack(__sample__=dims)
        aod = aod.stack(__sample__=dims)
        dim = "__sample__"

    lr = LinearRegression()

    # TODO: name of 'aod'
    lr.fit(
        predictors={"aod": aod},
        target=tas_residuals,
        dim=dim,
        fit_intercept=False,
    )

    params = lr.params

    params.attrs["version"] = version

    if params["aod"] >= 0:
        warnings.warn(
            f"The slope of 'aod' is positive ({params['aod'].values:0.2f}) but is "
            "expected to be negative - did you pass the residuals?"
        )

    return params


def _predict_volcanic_contribution(time, hist_period, params, version="2022"):
    """
    predict volcanic contribution to temperature anomalies using aerosol optical depth
    observations as proxy

    Parameters
    ----------
    time : xr.DataArray
        DataArray containing the time coords to predict the volcanic contribution for.
    hist_period : slice
        Slice object indicating the years of the historical period. E.g.
        ``slice("1850", "2014")``.
    params : xr.Dataset
        Parameters of the linear regression fit, obtained from
        ``fit_volcanic_influence``.
    version : str, default: "2022"
        Which version of the aerosol optical depth observations to use. Currently
        only "2022" is valid.

    Returns
    -------
    volcanic_contribution : xr.DataArray
        DataArray containing the volcanic contribution to temperature anomalies.
    """

    # TODO: check version from params

    # ensure the time axis of aod and the model data aligns
    _assert_annual_data(time)
    aod = _load_and_align_strat_aod_obs(time, hist_period, version=version)

    # set up linear regression model
    lr = LinearRegression.from_params(params)

    # estimate volcanic contribution
    volcanic_contribution = lr.predict({"aod": aod})

    return volcanic_contribution


@_datatree_wrapper
def superimpose_volcanic_influence(
    tas_globmean_lowess, params, hist_period=None, *, dim="time", version="2022"
):
    """
    superimpose volcanic influence on smooth temperature anomalies using aerosol optical
    depth observations as proxy

    Parameters
    ----------
    tas_globmean_lowess : xr.DataArray | xr.Dataset | xr.DataTree
        DataArray containing smooth global mean temperature anomalies to superimpose
        the volcanic influence.
    params : xr.Dataset
        Parameters of the linear regression fit, obtained from
        ``fit_volcanic_influence``.
    hist_period : slice | None
        Slice object indicating the years of the historical period. E.g.
        ``slice("1850", "2014")``. If None uses the entire time period in
        ``tas_globmean_lowess``.
    dim : str, default: "time"
        Dimension along which to estimate the volcanic influence.
    version : str, default: "2022"
        Which version of the aerosol optical depth observations to use. Currently only
        "2022" is valid.

    Returns
    -------
    tas : xr.DataArray | xr.Dataset | xr.DataTree
        tas_globmean_lowess with the estimated volcanic influene superimposed
    """

    time = tas_globmean_lowess[dim]
    volcanic_contribution = _predict_volcanic_contribution(
        time, hist_period, params, version=version
    ).prediction

    tas_globmean_lowess_volc = tas_globmean_lowess + volcanic_contribution

    return tas_globmean_lowess_volc
