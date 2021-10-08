Changelog
=========

v0.9.0 - unreleased
-------------------

New Features
^^^^^^^^^^^^


Breaking changes
^^^^^^^^^^^^^^^^


Deprecations
^^^^^^^^^^^^


Bug fixes
^^^^^^^^^


Documentation
^^^^^^^^^^^^^

v0.8.2 - 2021-10-07
-------------------

Bug fixes
^^^^^^^^^

- Reintroduce ability to read in cmip5 data from the cmip5-ng archive at ETH
  (`#90 <https://github.com/MESMER-group/mesmer/pull/90>`_).
  By `Lea Beusch <https://github.com/leabeusch>`_.

Internal Changes
^^^^^^^^^^^^^^^^
- Reproduce the test files because of a change in regionmask which affected the mesmer
  tests (`#95 <https://github.com/MESMER-group/mesmer/issues/95>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Refactor and speed up of the Gaspari-Cohn function and the calculation of the great
  circle distance (`#85 <https://github.com/MESMER-group/mesmer/pull/85>`_,
  `#88 <https://github.com/MESMER-group/mesmer/pull/88>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- The geopy package is no longer a dependency of mesmer
  (`#88 <https://github.com/MESMER-group/mesmer/pull/88>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Convert README from Markdown to reStructuredText to fix package build errors. Also allows
  to include the README in the docs to avoid duplication
  (`#102 <https://github.com/MESMER-group/mesmer/issues/102>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

v0.8.1 - 2021-07-15
-------------------

- Update example script (`#80 <https://github.com/MESMER-group/mesmer/pull/80>`_).

v0.8.0 - 2021-07-13
-------------------

- First release on PyPI and conda (`#79 <https://github.com/MESMER-group/mesmer/pull/79>`_).
