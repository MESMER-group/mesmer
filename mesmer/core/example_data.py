import importlib
import pathlib


def cmip6_ng_path(*, relative=False):
    """path of the cmip6_ng example data"""

    try:
        path = importlib.resources.files("mesmer").parent / "data" / "cmip6-ng"
    except ImportError:
        raise ModuleNotFoundError("mesmer must be installed")

    if not path.exists():
        raise FileNotFoundError(
            "The example data is only available from the mesmer repository "
            "(i.e. when mesmer is cloned and not when installed via pip/ conda)."
        )

    if relative:
        path = path.relative_to(pathlib.Path.cwd(), walk_up=True)

    return path
