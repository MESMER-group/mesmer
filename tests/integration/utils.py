import numpy as np
import numpy.testing as npt
import xarray as xr
import xarray.testing as xrt


def _check_dict(first, second, first_name="left", second_name="right"):
    for k in first:
        first_val = first[k]
        try:
            second_val = second[k]
        except KeyError:
            raise AssertionError(
                "Key `{}` is in '{}' but is not in '{}'".format(
                    k, first_name, second_name
                )
            )

        assert type(first_val) == type(second_val)
        if isinstance(first_val, dict):
            _check_dict(first_val, second_val, first_name, second_name)
        elif isinstance(first_val, np.ndarray):
            npt.assert_allclose(first_val, second_val)
        elif isinstance(first_val, xr.DataArray):
            xrt.assert_allclose(first_val, second_val)
        elif np.issubdtype(np.array(first_val).dtype, np.number):
            npt.assert_allclose(first_val, second_val)
        else:
            assert first_val == second_val, k
