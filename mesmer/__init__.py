# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
The mesmer package provides tools to train the MESMER emulator, create emulations, and
analyze the results.
"""

from importlib.metadata import version as _get_version

from . import calibrate_mesmer, core, create_emulations, io, stats, testing, utils
from .core import _data as data
from .core import geospatial, grid, mask, volc, weighted

# "legacy" modules
__all__ = [
    "calibrate_mesmer",
    "create_emulations",
    "io",
    "utils",
]

# "new" "modules"
__all__ += [
    "core",
    "data",
    "geospatial",
    "grid",
    "mask",
    "stats",
    "testing",
    "volc",
    "weighted",
]


try:
    __version__ = _get_version("mesmer-emulator")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
