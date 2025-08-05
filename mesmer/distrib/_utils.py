import functools
import warnings


def _ignore_warnings(func):

    # adapted from https://stackoverflow.com/a/70292317
    # TODO: don't suppress all warnings
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return func(*args, **kwargs)

    return _wrapper
