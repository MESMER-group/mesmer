import os.path

import joblib
import pytest

TEST_DATA_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test-data"
)


@pytest.fixture(scope="session")
def test_data_root_dir():
    if not os.path.isdir(TEST_DATA_ROOT_DIR):
        pytest.skip("test data required")
    return TEST_DATA_ROOT_DIR


@pytest.fixture()
def test_mesmer_bundle(test_data_root_dir):
    return joblib.load(os.path.join(test_data_root_dir, "test-mesmer-bundle.pkl"))


def pytest_addoption(parser):
    parser.addoption(
        "--update-expected-files",
        action="store_true",
        default=False,
        help="Overwrite expected files",
    )


@pytest.fixture
def update_expected_files(request):
    return request.config.getoption("--update-expected-files")
