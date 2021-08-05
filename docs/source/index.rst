.. mesmer documentation master file, created by
   sphinx-quickstart on Mon May 10 23:27:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MESMER's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For users

   installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 2
   :caption: Versions

   changelog

MESMER: spatially resolved Earth System Model emulations
--------------------------------------------------------
**MESMER** is a **M**\ odular **E**\ arth **S**\ ystem **M**\ odel **E**\ mulator with
spatially **R**\ esolved output, which stochastically creates Earth System
Model-specific spatio-temporally correlated climate variable field realizations at a
negligible computational cost.

In combination with a global mean temperature emulator, MESMER can account for all three
major sources of climate change projection uncertainty at the local scale: (i) internal
variability uncertainty, i.e., unforced natural climate variability; (ii) forced climate
response uncertainty, i.e., the Earth’s system response to forced natural changes (solar
and volcanic) and human influences (greenhouse gas and aerosol emissions, land use
changes etc.); and (iii) emission uncertainty, i.e., uncertainty in the emission pathway
humans decide to follow. An interface between MESMER and global mean temperature
emulators can be found at https://github.com/MESMER-group/mesmer-openscmrunner.

MESMER is under active development both scientifically and technically. Future work will
increase its user friendliness and extend its emulation capabilities to include
additional emulation methods and target climate variables.

Citing MESMER
-------------

Scientific publications using MESMER should cite the following publication:

Beusch, Lea, Lukas Gudmundsson, and Sonia I. Seneviratne, 2020: Emulating Earth system
model temperatures with MESMER: from global mean temperature trajectories to
grid-point-level realizations on land. Earth Syst. Dynam., 11, 139–159, 2020,
https://doi.org/10.5194/esd-11-139-2020

License
-------

Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.

MESMER is free software; you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, version 3  or
(at your option) any later version.

MESMER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with MESMER. If
not, see https://www.gnu.org/licenses/.

The full list of code contributors can be found in AUTHORS or on
`github.com/contributors <https://github.com/MESMER-group/mesmer/graphs/contributors>`_
