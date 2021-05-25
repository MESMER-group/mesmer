# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0; see LICENSE or https://www.gnu.org/licenses/
"""
The mesmer package provides tools to train the MESMER emulator, create emulations, and
analyze the results.
"""
# flake8: noqa

# import subpackages (so that if I import mesmer, I can directly access e.g., io content
# by calling e.g., mesmer.io.load_const())
from . import calibrate_Create_emulations, io, utils

# like this could directly access all functions (i.e., if I import mesmer, I can
# directly access fcts, e.g., mesmer.load_const() decided against it for now to keep the
# package structure more visible) from .io import *
