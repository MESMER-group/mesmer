import numpy as np
import pytest
import xarray as xr

from mesmer.testing import assert_dict_allclose


def test_assert_dict_allclose_type_key_errors():

    a = {"a": 1}
    b = {"a": 1, "b": 2}
    c = {"a": "a"}

    # not passing a dict
    with pytest.raises(AssertionError):
        assert_dict_allclose(None, a)

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, None)

    # differing keys
    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    # different type of value
    with pytest.raises(AssertionError):
        assert_dict_allclose(a, c)

    with pytest.raises(AssertionError):
        assert_dict_allclose(c, a)


def test_assert_dict_allclose_nonnumeric():

    a = {"a": "a"}
    b = {"a": "b"}
    c = {"a": "a"}

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)


@pytest.mark.parametrize(
    "value1, value2",
    (
        (1.0, 2.0),
        (np.array([1.0, 2.0]), np.array([2.0, 1.0])),
        (xr.DataArray([1.0]), xr.DataArray([2.0])),
        (xr.Dataset(data_vars={"a": [1.0]}), xr.Dataset(data_vars={"a": [2.0]})),
    ),
)
def test_assert_dict_allclose_unequal_values(value1, value2):

    a = {"a": value1}
    b = {"a": value2}
    c = {"a": value1}
    d = {"a": value1 + 0.99e-7}

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)

    assert_dict_allclose(a, d)
    assert_dict_allclose(d, a)


def test_assert_dict_allclose_mixed_numpy():

    a = {"a": 1}
    b = {"a": np.array(1)}
    c = {"a": np.array([1])}

    assert_dict_allclose(a, b)
    assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)


def test_assert_dict_allclose_nested():

    a = {"a": {"a": 1}}
    b = {"a": {"a": 2}}
    c = {"a": {"a": 1}}

    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)

    with pytest.raises(AssertionError):
        assert_dict_allclose(b, a)

    assert_dict_allclose(a, c)
    assert_dict_allclose(c, a)
