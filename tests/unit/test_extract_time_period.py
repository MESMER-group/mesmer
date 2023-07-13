import numpy as np
import pytest

from mesmer.utils.select import extract_time_period


def test_extract_time_period_deprecation():

    time = np.arange(1950, 2050)

    data = np.linspace(0, 1, time.size)

    with pytest.warns(FutureWarning, match="`extract_time_period` is deprecated"):
        extract_time_period(data, time, 1955, 2005)


@pytest.mark.filterwarnings("ignore:`extract_time_period` is deprecated")
def test_extract_time_period_1D():

    time = np.arange(1950, 2050)

    data = np.linspace(0, 1, time.size)

    result_data, result_time = extract_time_period(data, time, 1955, 2005)

    expected_data = data[5 : 5 + 50 + 1]
    expected_time = np.arange(1955, 2005 + 1)

    np.testing.assert_equal(result_data, expected_data)
    np.testing.assert_equal(result_time, expected_time)


@pytest.mark.filterwarnings("ignore:`extract_time_period` is deprecated")
def test_extract_time_period_2D():

    time = np.arange(1900, 2000)

    data = np.arange(3 * time.size).reshape(3, -1)

    result_data, result_time = extract_time_period(data, time, 1911, 1995)

    expected_data = data[:, 11 : 95 + 1]
    expected_time = np.arange(1911, 1995 + 1)

    np.testing.assert_equal(result_data, expected_data)
    np.testing.assert_equal(result_time, expected_time)


@pytest.mark.filterwarnings("ignore:`extract_time_period` is deprecated")
def test_extract_time_period_3D():

    time = np.arange(1900, 2000)

    # (run, time, gridpoint)
    data = np.arange(3 * 5 * time.size).reshape(3, -1, 5)

    result_data, result_time = extract_time_period(data, time, 1911, 1995)

    expected_data = data[:, 11 : 95 + 1, :]
    expected_time = np.arange(1911, 1995 + 1)

    np.testing.assert_equal(result_data, expected_data)
    np.testing.assert_equal(result_time, expected_time)
