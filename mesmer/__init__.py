# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
The mesmer package provides tools to train the MESMER emulator, create emulations, and
analyze the results.
"""

from importlib.metadata import version as _get_version

from mesmer import (
    anomaly,
    calibrate_mesmer,
    core,
    create_emulations,
    datatree,
    example_data,
    geospatial,
    grid,
    io,
    mask,
    stats,
    testing,
    utils,
    volc,
    weighted,
)
from mesmer.core import _data as data
from mesmer.core.options import get_options, set_options

# "legacy" modules
__all__ = [
    "calibrate_mesmer",
    "create_emulations",
    "io",
    "utils",
]

# "new" "modules"
__all__ += [
    "anomaly",
    "core",
    "data",
    "datatree",
    "example_data",
    "geospatial",
    "get_options",
    "grid",
    "mask",
    "set_options",
    "stats",
    "testing",
    "volc",
    "weighted",
]


try:
    __version__ = _get_version("mesmer-emulator")
except Exception:  # pragma: no cover
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
