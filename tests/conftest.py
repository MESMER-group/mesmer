import os.path

import pytest

TEST_DATA_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test-data"
)


@pytest.fixture(scope="session")
def test_data_root_dir():
    if not os.path.isdir(TEST_DATA_ROOT_DIR):
        pytest.skip("test data required")
    return TEST_DATA_ROOT_DIR


@pytest.fixture
def update_expected_files(request):
    return request.config.getoption("--update-expected-files")


# add markers and options


def pytest_addoption(parser):
    parser.addoption(
        "--update-expected-files",
        action="store_true",
        default=False,
        help="Overwrite expected files",
    )

    parser.addoption("--all", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --all option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
