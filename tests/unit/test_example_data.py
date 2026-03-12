import importlib.resources

import pytest

import mesmer

# import pathlib

# TODO: enable test and add network flag to pytest
# def test_download_cmip6_ng_data_one(tmp_path):
#     remote_resource = mesmer.example_data._get_remote_resource(cache_dir=tmp_path)

#     name = list(mesmer.example_data._REGISTRY.keys())[0]

#     path = remote_resource.fetch(name)
#     path = pathlib.Path(path)

#     assert path.exists()


def test_download_cmip6_ng_data_without_download(tmp_path):

    # path of cmip6-ng example data - if working in a repo and editable install
    path = importlib.resources.files("mesmer").parent / "data" / "cmip6-ng"

    if not path.exists():
        pytest.skip("not a git repo - data not available")

    # create a symbolic link to the data dir - this is necessary because the data
    # is downloaded to a "main" or vX.Y.Z folder

    remote_resource = mesmer.example_data._get_remote_resource(cache_dir=tmp_path)
    remote_resource.abspath.mkdir()
    (remote_resource.abspath / "cmip6-ng").symlink_to(path)

    mesmer.example_data._download_cmip6_ng_data(cache_dir=tmp_path)
