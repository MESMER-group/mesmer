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

At the moment, we provide infrastrucutre to emulate annual and monthly near-surface air
temperature (MESMER and MESMER-M) as well as various climate extreme indicators (MESMER-X).

MESMER is under active development both scientifically and technically. Future work will
increase its user friendliness and extend its emulation capabilities to include
additional emulation methods and target climate variables.

Citing MESMER
-------------

Scientific publications using MESMER Software should cite the following publication:

  placeholder for MESMER publication, to be added once published.

Depending on which application of MESMER you use (MESMER, MESMER-M, MESMER-X, etc.) please
also cite the relevant publications found in our publication list on
`readthedocs <https://mesmer-emulator.readthedocs.io/en/latest/publications.html>`_.

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

Mesmer bundles data for analysis, tutorials, and testing:

- CMIP6 data from IPSL-CM6A-LR model, available under a `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`__ license.
  Used in tutorial notebooks and tests.

- Stratospheric aerosol optical depth data sourced licensed as `Creative Commons Zero <https://creativecommons.org/public-domain/cc0/>`__.
  Used to estimate the volcanic influence.

For details see `data README <data/README.md>`_.
