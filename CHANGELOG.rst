Changelog
=========

v0.9.0 - unreleased
-------------------

New Features
^^^^^^^^^^^^

- Refactored statistical functionality for linear regression:
   - Create :py:class:`mesmer.stats.linear_regression.LinearRegression` which encapsulates
     ``fit``, ``predict``, etc. methods around linear regression
     (`#134 <https://github.com/MESMER-group/mesmer/pull/134>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - Add ``mesmer.stats._fit_linear_regression_xr``: xarray wrapper for ``mesmer.stats._fit_linear_regression_np``.
     (`#123 <https://github.com/MESMER-group/mesmer/pull/123>`_ and `#142 <https://github.com/MESMER-group/mesmer/pull/142>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - Add add ``fit_intercept`` argument to the ``linear_regression`` fitting methods and
     functions (`#144 <https://github.com/MESMER-group/mesmer/pull/144>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - Allow to pass 1-dimensional targets to :py:meth:`mesmer.stats.linear_regression.LinearRegression.fit`
     (`#221 <https://github.com/MESMER-group/mesmer/pull/221>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.

- Refactored statistical functionality for auto regression:
   - Add ``mesmer.stats.auto_regression._fit_auto_regression_xr``: xarray wrapper to fit an
     auto regression model (`#139 <https://github.com/MESMER-group/mesmer/pull/139>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - Add ``mesmer.stats.auto_regression._draw_auto_regression_correlated_np``: to draw samples of an
     auto regression model (`#161 <https://github.com/MESMER-group/mesmer/pull/161>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - Extract function to select the order of the auto regressive model: ``mesmer.stats.auto_regression._select_ar_order_xr``
     (`#176 <https://github.com/MESMER-group/mesmer/pull/176>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.

- Refactored functions dealing with the spatial covariance and its localization:
   - Add xarray wrappers :py:func:`mesmer.stats.localized_covariance.adjust_covariance_ar1`
     and :py:func:`mesmer.stats.localized_covariance.find_localized_empirical_covariance`
     (`#191 <https://github.com/MESMER-group/mesmer/pull/191>`__).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - Refactor and extract numpy-based functions dealing with the spatial covariance and its localization
     (`#167 <https://github.com/MESMER-group/mesmer/pull/167>`__ and `#184
     <https://github.com/MESMER-group/mesmer/pull/184>`__).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - Allow to pass `1 x n` arrays to :py:func:`mesmer.stats.localized_covariance.adjust_covariance_ar1`
     (`#224 <https://github.com/MESMER-group/mesmer/pull/224>`__).
     By `Mathias Hauser <https://github.com/mathause>`_.

- Other refactorings:
   - Extract the LOWESS smoothing for xarray objects: :py:func:`mesmer.stats.smoothing.lowess`.
     (`#193 <https://github.com/MESMER-group/mesmer/pull/193>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.

- Added helper functions to process xarray-based model data:
   - Added functions to stack regular lat-lon grids to 1D grids and unstack them again (`#217
     <https://github.com/MESMER-group/mesmer/pull/217>`_). By `Mathias Hauser
     <https://github.com/mathause>`_.
   - Added functions to mask the ocean and Antarctica (`#219
     <https://github.com/MESMER-group/mesmer/pull/219>`_). By `Mathias Hauser
     <https://github.com/mathause>`_.
   - Added functions to calculate the weighted global mean (`#220
     <https://github.com/MESMER-group/mesmer/pull/220>`_). By `Mathias Hauser
     <https://github.com/mathause>`_.

Breaking changes
^^^^^^^^^^^^^^^^

- Localization radii that lead to singular matrices are now skipped (`#187 <https://github.com/MESMER-group/mesmer/issues/187>`__).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Refactor and split :py:func:`train_l_prepare_X_y_wgteq` into two functions:
  :py:func:`get_scenario_weights` and :py:func:`stack_predictors_and_targets`
  (`#143 <https://github.com/MESMER-group/mesmer/pull/143>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Moved ``gaspari_cohn`` & ``calc_geodist_exact`` from ``io.load_constant_files`` to ``core.computation``
  (`#158 <https://github.com/MESMER-group/mesmer/issues/158>`_).
  By `Yann Quilcaille <https://github.com/yquilcaille>`_.
- The function ``mask_percentage`` has been renamed to :py:func:`utils.regionmaskcompat.mask_3D_frac_approx`
  (`#202 <https://github.com/MESMER-group/mesmer/pull/202>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Removed :py:func:`mesmer.io.load_constant_files.infer_interval_breaks` and the edges
  from the `lat` and `lon` dictionaries i.e., ``lon["e"]`` and ``lat["e"]``
  (`#233 <https://github.com/MESMER-group/mesmer/pull/233>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Deprecated the ``reg_type`` argument to :py:func:`mesmer.io.load_constant_files.load_regs_ls_wgt_lon_lat`
  and the ``reg_dict`` argument to :py:func:`mesmer.utils.select.extract_land`. These arguments
  no longer have any affect (`#235 <https://github.com/MESMER-group/mesmer/pull/235>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.


Deprecations
^^^^^^^^^^^^

- The function ``mesmer.create_emulations.create_emus_gt`` has been renamed to
  :py:func:`create_emulations.gather_gt_data` (`#246 <https://github.com/MESMER-group/mesmer/pull/246>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

- The function ``mesmer.utils.select.extract_time_period`` is now deprecated and will be
  removed in a future version. Please raise an issue if you use this function (`#243
  <https://github.com/MESMER-group/mesmer/pull/243>`_). By `Mathias Hauser
  <https://github.com/mathause>`_.

Bug fixes
^^^^^^^^^

- Fix three issues with :py:func:`utils.regionmaskcompat.mask_3D_frac_approx`. Note that these
  issues are only relevant if passing xarray objects and/ or masks close to the poles
  (`#202 <https://github.com/MESMER-group/mesmer/pull/202>`_ and `#218 <https://github.com/MESMER-group/mesmer/pull/218>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

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

- Make use of :py:class:`mesmer.stats.linear_regression.LinearRegression` in
   - :py:func:`mesmer.calibrate_mesmer.train_gt_ic_OLSVOLC` (`#145 <https://github.com/MESMER-group/mesmer/pull/145>`_).
     By `Mathias Hauser <https://github.com/mathause>`_.
   - :py:func:`mesmer.create_emulations.create_emus_lv_OLS` and :py:func:`mesmer.create_emulations.create_emus_OLS_each_gp_sep`
     (`#240 <https://github.com/MESMER-group/mesmer/pull/240>`_).By `Mathias Hauser <https://github.com/mathause>`_.

- Add python 3.10 to list of supported versions (`#162 <https://github.com/MESMER-group/mesmer/pull/162>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

- Move contents of setup.py to setup.cfg (`#169 <https://github.com/MESMER-group/mesmer/pull/169>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

- Use pyproject.toml for the build-system and setuptools_scm for the `__version__`
  (`#188 <https://github.com/MESMER-group/mesmer/pull/188>`_).
  By `Mathias Hauser <https://github.com/mathause>`_.

- Added additional tests for the calibration step (`#209 <https://github.com/MESMER-group/mesmer/issues/209>`_):
   - one scenario (SSP5-8.5) and two ensemble members (`#211 <https://github.com/MESMER-group/mesmer/pull/211>`_)
   - two scenarios (SSP1-2.6 and SSP5-8.5) with one and two ensemble members, respectively (`#214 <https://github.com/MESMER-group/mesmer/pull/214>`_)

  By `Mathias Hauser <https://github.com/mathause>`_.


v0.8.3 - 2021-12-23
-------------------

New Features
^^^^^^^^^^^^

- Add ``mesmer.stats._linear_regression`` (renamed to ``mesmer.stats._fit_linear_regression_np``
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
