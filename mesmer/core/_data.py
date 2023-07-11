import pooch

import mesmer


def fetch_remote_data(name):
    """
    uses pooch to cache files
    """

    cache_dir = pooch.os_cache("mesmer")

    REMOTE_RESSOURCE = pooch.create(
        path=cache_dir,
        # The remote data is on Github
        base_url="https://github.com/MESMER-group/mesmer/raw/{version}/data/",
        registry={
            "isaod_gl_2022.dat": "c83881a18e74912385ad578282de721cc8e866b62cbbc75446e52e7041c81cff",
        },
        version=f"v{mesmer.__version__}",
        version_dev="main",
    )

    # the file will be downloaded automatically the first time this is run.
    return REMOTE_RESSOURCE.fetch(name)
