import numpy as np
import pytest
import xarray as xr

import mesmer.core.utils


@pytest.mark.parametrize("da", (None, xr.Dataset()))
def test_check_dataarray_form_wrong_type(da):

    with pytest.raises(TypeError, match="Expected da to be an xr.DataArray"):
        mesmer.core.utils._check_dataarray_form(da)

    with pytest.raises(TypeError, match="Expected test to be an xr.DataArray"):
        mesmer.core.utils._check_dataarray_form(da, name="test")


@pytest.mark.parametrize("ndim", (0, 1, 3))
def test_check_dataarray_form_ndim(ndim):

    da = xr.DataArray(np.ones((2, 2)))

    with pytest.raises(ValueError, match=f"da should be {ndim}-dimensional"):
        mesmer.core.utils._check_dataarray_form(da, ndim=ndim)

    with pytest.raises(ValueError, match=f"test should be {ndim}-dimensional"):
        mesmer.core.utils._check_dataarray_form(da, ndim=ndim, name="test")

    # no error
    mesmer.core.utils._check_dataarray_form(da, ndim=2)


@pytest.mark.parametrize("required_dims", ("foo", ["foo"], ["foo", "bar"]))
def test_check_dataarray_form_required_dims(required_dims):

    da = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))

    with pytest.raises(ValueError, match="da is missing the required dims"):
        mesmer.core.utils._check_dataarray_form(da, required_dims=required_dims)

    with pytest.raises(ValueError, match="name is missing the required dims"):
        mesmer.core.utils._check_dataarray_form(
            da, required_dims=required_dims, name="test"
        )

    # no error
    mesmer.core.utils._check_dataarray_form(da, required_dims="x")
    mesmer.core.utils._check_dataarray_form(da, required_dims="y")
    mesmer.core.utils._check_dataarray_form(da, required_dims=["x", "y"])
    mesmer.core.utils._check_dataarray_form(da, required_dims={"x", "y"})
