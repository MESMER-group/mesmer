import re

import numpy as np
import pytest
import xarray as xr

import mesmer


@pytest.fixture
def example_tas():

    data_hist = np.arange(12).reshape(3, 4)

    time_hist = xr.date_range("1850-01-01", "2000-01-01", freq="50YS")

    ens_hist = ["0", "1", "2"]

    dta = xr.DataArray(
        data_hist,
        dims=("ens", "time"),
        coords={"ens": ens_hist, "time": time_hist},
        name="tas",
        attrs={"name": "tas"},
    )
    hist = xr.Dataset(data_vars={"tas": dta}, attrs={"model": "foo"})

    data_proj = np.arange(4).reshape(2, 2) * 10
    time_proj = xr.date_range("2050-01-01", "2100-01-01", freq="50YS")

    ens_proj = ["0", "2"]
    dta = xr.DataArray(
        data_proj,
        dims=("ens", "time"),
        coords={"ens": ens_proj, "time": time_proj},
        name="tas",
    )
    proj = xr.Dataset(data_vars={"tas": dta})

    tas = xr.DataTree.from_dict({"historical": hist, "proj": proj})

    return tas


def test_calc_anomaly_errors(example_tas):
    # TODO: uncomment tests once requiring new xarray version

    with pytest.raises(
        ValueError, match=re.escape("The ref_scenario (wrong) is missing from `dt`")
    ):
        mesmer.anomaly.calc_anomaly(
            example_tas, slice("1850", "1900"), ref_scenario="wrong"
        )

    # different error is raised in datatree than xarray
    # with pytest.raises(ValueError, match="Dimensions {'wrong'} do not exist"):
    #     mesmer.anomaly.calc_anomaly(
    #         example_tas, slice("1850", "1900"), time_dim="wrong"
    #     )

    with pytest.raises(ValueError, match="No data selected for reference period"):
        mesmer.anomaly.calc_anomaly(example_tas, slice("1750", "1800"))

    # add scenario containing a ensemble that is missing from historical
    # example_tas["proj1"] = example_tas["proj"].dataset.assign_coords(ens=["0", "3"])

    ds = example_tas["proj"].to_dataset().assign_coords(ens=["0", "3"])
    example_tas["proj1"] = xr.DataTree(ds)

    with pytest.raises(
        ValueError, match="Subtracting the reference changed the coordinates."
    ):
        mesmer.anomaly.calc_anomaly(example_tas, slice("1850", "1900"))


def test_calc_anomaly(example_tas):

    result = mesmer.anomaly.calc_anomaly(example_tas, slice("1850", "1900"))

    ref = example_tas["historical"].sel(time=slice("1850", "1900")).mean("time")

    expected = example_tas.copy(deep=True)

    expected["historical"] = expected["historical"] - ref
    expected["proj"] = expected["proj"] - ref

    xr.testing.assert_equal(result, expected)

    # ensure attrs are conserved
    assert result["historical"].dataset.attrs == {"model": "foo"}
    assert result["historical"].dataset.tas.attrs == {"name": "tas"}
