# adapted from xarray under the terms of its license - see licences/XARRAY_LICENSE
from __future__ import annotations

from typing import Literal, TypedDict, Unpack


class _OPTIONS(TypedDict, total=False):
    threads: Literal["default"] | int | None


OPTIONS: _OPTIONS = {
    "threads": "default",
}


def _assert_valid_threads_option(name, threads):

    if not (
        threads == "default"
        or threads is None
        or (isinstance(threads, int) and threads > 0)
    ):
        msg = f"'{name}' must be 'default', a positive integer or None, got '{threads}'"
        raise ValueError(msg)


_VALIDATORS = {
    "threads": _assert_valid_threads_option,
}


class set_options:
    """
    Set options for mesmer in a controlled context.

    Parameters
    ----------
    threads : "default" |  int | None, default: "default"
        Number of threads to use. Restricting the number of threads was found to speed
        up matrix decomposition on linux systems. Only applied within mesmer.

        * "default": uses ``min(os.cpu_count() // 2, 16)``

        * None: uses the currently selected number of threads

        * int: sets the maximum number of threads to `threads`

    Examples
    --------
    >>> import mesmer
    >>> mesmer.set_options(threads=None)

    """

    def __init__(self, **kwargs: Unpack[_OPTIONS]):

        self.old = {}

        for key, value in kwargs.items():
            if key not in OPTIONS:
                raise ValueError(
                    f"{key!r} is not in the set of valid options {set(OPTIONS)!r}"
                )

            _VALIDATORS[key](key, value)

            # mypy does not know that key must be a literal from _OPTIONS TypedDict
            self.old[key] = OPTIONS[key]  # type:ignore[literal-required]

        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)


def get_options():
    """
    Get options for mesmer.

    See Also
    --------
    set_options
    """
    return OPTIONS
