import os

import pytest
import threadpoolctl

import mesmer
from mesmer.core.utils import _set_threads_from_options


@pytest.mark.parametrize("invalid_option", [None, "None", "__foo__"])
def test_option_invalid_error(invalid_option) -> None:

    with pytest.raises(ValueError, match="not in the set of valid options"):

        mesmer.set_options(invalid_option=invalid_option)  # type:ignore[call-arg]


def test_options_threads_errors() -> None:

    default = mesmer.core.options.OPTIONS["threads"]
    assert default == "default"

    msg = "'threads' must be 'default', a positive integer or None"

    with pytest.raises(ValueError, match=msg):
        mesmer.set_options(threads=0)

    with pytest.raises(ValueError, match=msg):
        mesmer.set_options(threads=-3)

    with pytest.raises(ValueError, match=msg):
        mesmer.set_options(threads=3.5)  # type:ignore[arg-type]


def test_options_threads() -> None:

    @_set_threads_from_options()
    def func():
        # return number of selected threads - not sure how brittle this is
        return threadpoolctl.threadpool_info()[0]["num_threads"]

    expected_default = min(16, os.cpu_count() // 2)

    assert func() == expected_default

    with mesmer.set_options(threads=1):
        assert func() == 1

    with mesmer.set_options(threads=None):
        assert func() == os.cpu_count()
