import warnings
from contextlib import contextmanager


def format_record(record) -> str:
    """Format warning record like `FutureWarning('Function will be deprecated...')`"""
    return f"{str(record.category)[8:-2]}('{record.message}'))"


@contextmanager
def assert_no_warnings():
    with warnings.catch_warnings(record=True) as record:
        yield record
        assert (
            len(record) == 0
        ), f"Got {len(record)} unexpected warning(s): {[format_record(r) for r in record]}"
