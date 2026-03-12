import importlib
import pathlib

import pooch

import mesmer


def _get_remote_resource(cache_dir=None):

    if cache_dir is None:
        cache_dir = pooch.os_cache("mesmer")

    remote_resource = pooch.create(
        path=cache_dir,
        # The remote data is on Github
        base_url="https://github.com/MESMER-group/mesmer/raw/{version}/data/",
        registry=_REGISTRY,
        version=f"v{mesmer.__version__}",
        version_dev="main",
    )

    return remote_resource


def cmip6_ng_path(*, relative=False):
    """path of the cmip6_ng example data"""

    try:
        path = importlib.resources.files("mesmer").parent / "data" / "cmip6-ng"
    except ImportError:
        raise ModuleNotFoundError("mesmer must be installed")

    if not path.exists():
        _download_cmip6_ng_data()

        remote_resource = _get_remote_resource()

        path = remote_resource.abspath / "cmip6-ng"

    if relative:
        path = path.relative_to(pathlib.Path.cwd(), walk_up=True)

    return path


_REGISTRY = {
    "cmip6-ng/hfds/ann/g025/hfds_ann_IPSL-CM6A-LR_ssp585_r2i1p1f1_g025.nc": "9bd161ae44ff66a207f6240e20daf82b5e981ac71415452841c6874ff2060f2d",
    "cmip6-ng/hfds/ann/g025/hfds_ann_IPSL-CM6A-LR_ssp585_r1i1p1f1_g025.nc": "2412aa8cffc5f8f42006fd7e2e2c21d9a16e1f67a167f220d1223a6fe11cea79",
    "cmip6-ng/hfds/ann/g025/hfds_ann_IPSL-CM6A-LR_historical_r1i1p1f1_g025.nc": "98083946467208d3e3d02bafab583d51a306151684a213058eda5497871d0150",
    "cmip6-ng/hfds/ann/g025/hfds_ann_IPSL-CM6A-LR_ssp126_r1i1p1f1_g025.nc": "d67134a55de44ddfbc49fb3c7110f4cf7977cda5e2160aaf098e8f13624c2ef8",
    "cmip6-ng/hfds/ann/g025/hfds_ann_IPSL-CM6A-LR_historical_r2i1p1f1_g025.nc": "cce064dfbcd550d2eb65a63b5942a05edd66b4805f7b8535c8479588366b2a52",
    "cmip6-ng/tasmax/ann/g025/tasmax_ann_IPSL-CM6A-LR_ssp585_r2i1p1f1_g025.nc": "4ee50c310ba455a6b72fa5fd1faf89ec889225f2f01c9acdb76007f302bb17b0",
    "cmip6-ng/tasmax/ann/g025/tasmax_ann_IPSL-CM6A-LR_historical_r2i1p1f1_g025.nc": "959060559d8de9b4823871ca98f54b794a1e1990456900623de9f57167210370",
    "cmip6-ng/tasmax/ann/g025/tasmax_ann_IPSL-CM6A-LR_historical_r1i1p1f1_g025.nc": "67242041122090ec3e1792bc393d3588a4f3b2f588649669a8f05f4d78cdc27e",
    "cmip6-ng/tasmax/ann/g025/tasmax_ann_IPSL-CM6A-LR_ssp126_r1i1p1f1_g025.nc": "b6d7ab03cc6119e11d4d8991ed751dcfb753b5f9c55b73476012741f60612edf",
    "cmip6-ng/tasmax/ann/g025/tasmax_ann_IPSL-CM6A-LR_ssp585_r1i1p1f1_g025.nc": "5ba8b235cfb1a93740b458e20a3ab6630c1013b1ed9ff141418da41ee4949c0f",
    "cmip6-ng/tas/mon/g025/tas_mon_IPSL-CM6A-LR_ssp585_r1i1p1f1_g025.nc": "489ce62bc83af7153b8620ac7ef2b9e67dc3d9f0f77a2d2eb08257077bddc1a6",
    "cmip6-ng/tas/mon/g025/tas_mon_IPSL-CM6A-LR_ssp585_r2i1p1f1_g025.nc": "9be02fbb1f50f0280eb8b1d8612cf0a4de12af0571624e882bb2e8ca538de26d",
    "cmip6-ng/tas/mon/g025/tas_mon_IPSL-CM6A-LR_ssp126_r1i1p1f1_g025.nc": "22650a7dc02698c7d7003fbd1068da2c3f1bc92aaafbd927d0146a767e53289d",
    "cmip6-ng/tas/mon/g025/tas_mon_IPSL-CM6A-LR_historical_r1i1p1f1_g025.nc": "6cb08e527b7ef6038fdf0899c62a8603353038908ee163f98fa81cac79c88d24",
    "cmip6-ng/tas/mon/g025/tas_mon_IPSL-CM6A-LR_historical_r2i1p1f1_g025.nc": "eef721eac7d2030b0f92da3ae8a5ce133d65e37b770a94b4b7d40a8002310e3b",
    "cmip6-ng/tas/ann/g025/tas_ann_IPSL-CM6A-LR_ssp585_r2i1p1f1_g025.nc": "05eaa02c3782101ec96213276074ed1cf00ac6f16fa9cdfdf00266ffa04b29b8",
    "cmip6-ng/tas/ann/g025/tas_ann_IPSL-CM6A-LR_historical_r2i1p1f1_g025.nc": "5e575d597c81aa57580d3dad34620f66752fa5e186eb91a784a8d03cfeb78df9",
    "cmip6-ng/tas/ann/g025/tas_ann_IPSL-CM6A-LR_ssp126_r1i1p1f1_g025.nc": "b36d2846d9db643ea0c674c565b2180e60380485007e6759b1a58502ada5917b",
    "cmip6-ng/tas/ann/g025/tas_ann_IPSL-CM6A-LR_ssp585_r1i1p1f1_g025.nc": "813de4f222f48f1dc817339f935e453b0fb8418159383383ea1d8938dc2a963b",
    "cmip6-ng/tas/ann/g025/tas_ann_IPSL-CM6A-LR_historical_r1i1p1f1_g025.nc": "6bceba2da6b44f871b8a0e62af6511a628b2722a4fbbc2e1b599dd9ac31d1fa1",
}


def _download_cmip6_ng_data(cache_dir=None):
    """
    uses pooch to cache files
    """

    remote_resource = _get_remote_resource(cache_dir=cache_dir)

    for name in _REGISTRY:
        # the file will be downloaded automatically the first time this is run.
        remote_resource.fetch(name)
