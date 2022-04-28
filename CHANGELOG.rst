Changelog
=========

v0.9.0 - unreleased
-------------------

New Features
^^^^^^^^^^^^
- Create :py:class:`mesmer.core.linear_regression.LinearRegression` which encapsulates
  ``fit``, ``predict``, etc. methods around linear regression
  (`#134 <https://github.com/MESMER-group/mesmer/pull/134>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Add ``mesmer.core._fit_linear_regression_xr``: xarray wrapper for ``mesmer.core._fit_linear_regression_np``.
  (`#123 <https://github.com/MESMER-group/mesmer/pull/123>`_ and `#142 <https://github.com/MESMER-group/mesmer/pull/142>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Add ``mesmer.core.auto_regression._fit_auto_regression_xr``: xarray wrapper to fit an
  auto regression model (`#139 <https://github.com/MESMER-group/mesmer/pull/139>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

Breaking changes
^^^^^^^^^^^^^^^^

- Refactor and split :py:func:`train_l_prepare_X_y_wgteq` into two functions:
  :py:func:`get_scenario_weights` and :py:func:`stack_predictors_and_targets`
  (`#143 <https://github.com/MESMER-group/mesmer/pull/143>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

Deprecations
^^^^^^^^^^^^


Bug fixes
^^^^^^^^^


Documentation
^^^^^^^^^^^^^

- Add development/contributing docs (`#121 <https://github.com/MESMER-group/mesmer/pull/121>`_).
  By `Zeb Nicholls <https://github.com/znicholls>`_.

Internal Changes
^^^^^^^^^^^^^^^^

- Restore compatibility with regionmask v0.9.0 (`#136 <https://github.com/MESMER-group/mesmer/pull/136>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Renamed the ``interpolation`` keyword of ``np.quantile`` to ``method`` changed in
  numpy v1.22.0 (`#137 <https://github.com/MESMER-group/mesmer/pull/137>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

v0.8.3 - 2021-12-23
-------------------

New Features
^^^^^^^^^^^^

- Add ``mesmer.core._linear_regression`` (renamed to ``mesmer.core._fit_linear_regression_np``
  in `#142 <https://github.com/MESMER-group/mesmer/pull/142>`_). Starts the process of
  refactoring the codebase (`#116 <https://github.com/MESMER-group/mesmer/pull/116>`_).
  By `Zeb Nicholls <https://github.com/znicholls>`_.

Bug fixes
^^^^^^^^^

- Initialize ``llh_max`` to ``-inf`` to ensure the cross validation loop is entered
  (`#110 <https://github.com/MESMER-group/mesmer/pull/110>`_).
  By `Jonas Schwaab <https://github.com/woodhome23>`_.

Documentation
^^^^^^^^^^^^^

- Fix copyright notice and release version in documentation
  (`#127 <https://github.com/MESMER-group/mesmer/pull/127>`_).
  By `Zeb Nicholls <https://github.com/znicholls>`_.

Internal Changes
^^^^^^^^^^^^^^^^

- Automatically upload the code coverage to codecov.io after the test suite has run
  (`#99 <https://github.com/MESMER-group/mesmer/pull/99>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Internal refactor: moved a number of inline comments to their own line (especially if
  this allows to have the code on one line instead of several) and other minor cleanups
  (`#98 <https://github.com/MESMER-group/mesmer/pull/98>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Refactor ``io.load_cmipng_tas`` and ``io.load_cmipng_hfds`` to
  de-duplicate their code and add tests for them
  (`#55 <https://github.com/MESMER-group/mesmer/pull/55>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.


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
- Convert README from Markdown to reStructuredText to fix package build errors. Also
  allows to include the README in the docs to avoid duplication
  (`#102 <https://github.com/MESMER-group/mesmer/issues/102>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

v0.8.1 - 2021-07-15
-------------------

- Update example script (`#80 <https://github.com/MESMER-group/mesmer/pull/80>`_).

v0.8.0 - 2021-07-13
-------------------

- First release on PyPI and conda
  (`#79 <https://github.com/MESMER-group/mesmer/pull/79>`_).
