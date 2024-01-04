Changelog
=========

v0.10.0 - 2024.01.04
--------------------

version 0.10.0 fixes the bug in the legacy calibration and is thus not numerically
backward compatible. It also updated the supported python, pandas and xarray versions.
Updating the pandas version will create an error when trying to load pickled mesmer
bundles, requiring to use mesmer version v0.9.0 for this.

Bug fixes
^^^^^^^^^

Ensure de-duplicating the historical ensemble members conserves their order. Previously,
the legacy calibration routines used `np.unique`, which shuffles them. See `#338
<https://github.com/MESMER-group/mesmer/issues/338>`_ for details.
(`#339 <https://github.com/MESMER-group/mesmer/pull/339>`_).
By `Mathias Hauser`_.

Breaking changes
^^^^^^^^^^^^^^^^

- Removed support for python 3.7 and python 3.8 (
  `#163 <https://github.com/MESMER-group/mesmer/issues/163>`_,
  `#365 <https://github.com/MESMER-group/mesmer/pull/365>`_,
  `#367 <https://github.com/MESMER-group/mesmer/pull/367>`_, and
  `#371 <https://github.com/MESMER-group/mesmer/pull/371>`_).
  By `Mathias Hauser`_.
- The supported versions of some dependencies were changed (`#369 <https://github.com/MESMER-group/mesmer/pull/369>`_):

  ============ ============= =========
  Package      Old           New
  ============ ============= =========
  pandas       <2.0          >=2.0
  xarray       not specified >=2023.04
  ============ ============= =========

New Features
^^^^^^^^^^^^

- Add python 3.12 to list of supported versions (`#368 <https://github.com/MESMER-group/mesmer/pull/368>`_).
  By `Mathias Hauser`_.

v0.9.0 - 2023.12.19
-------------------

version 0.9.0 is a big step towards rewriting mesmer. All statistical functionality was
extracted and works for xarray data objects. It also contains data handling functions to
prepare climate model data using xarray.

- The restructured code is fully functional and can be used to calibrate and emulate
  temperature. However, it is still missing wrappers which encapsulate the full
  chain and helpers to simplify calibrating several scenarios and ensemble members.

- This version still contains the legacy routines to train and emulate temperature. It
  should have no numerical changes, only minimal changes in usage, and offers speed gains
  over v0.8.3.


Known bugs
^^^^^^^^^^

For the legacy training, the influence of the global variability is underestimated,
because the historical ensemble members are shuffled "randomly". This is kept in v0.9.0
for backward compatibility and will be fixed in a follow-up bug fix release. For details
see `#338 <https://github.com/MESMER-group/mesmer/issues/338>`_.


New Features
^^^^^^^^^^^^

- Extracted statistical functionality for linear regression:
   - Create :py:class:`mesmer.stats.LinearRegression` which encapsulates ``fit``, ``predict``,
     etc. methods around linear regression
     (`#134 <https://github.com/MESMER-group/mesmer/pull/134>`_).
     By `Mathias Hauser`_.
   - Add xarray wrapper for fitting a linear regression (
     `#123 <https://github.com/MESMER-group/mesmer/pull/123>`_ and
     `#142 <https://github.com/MESMER-group/mesmer/pull/142>`_).
     By `Mathias Hauser`_.
   - Add add ``fit_intercept`` argument to the ``linear_regression`` fitting methods and
     functions (`#144 <https://github.com/MESMER-group/mesmer/pull/144>`_).
     By `Mathias Hauser`_.
   - Allow to pass 1-dimensional targets to :py:meth:`mesmer.stats.LinearRegression.fit`
     (`#221 <https://github.com/MESMER-group/mesmer/pull/221>`_).
     By `Mathias Hauser`_.
   - Allow to `exclude` predictor variables in :py:meth:`mesmer.stats.LinearRegression.predict`
     (`#354 <https://github.com/MESMER-group/mesmer/pull/354>`_).
     By `Mathias Hauser`_.
   - Fixed two bugs related to (non-dimension) coordinates (
     `#332 <https://github.com/MESMER-group/mesmer/issues/332>`_,
     `#333 <https://github.com/MESMER-group/mesmer/issues/333>`_ and
     `#334 <https://github.com/MESMER-group/mesmer/pull/313>`_).
     By `Mathias Hauser`_.

- Extracted statistical functionality for auto regression:
   - Add ``mesmer.stats.fit_auto_regression``: xarray wrapper to fit an auto regression model
     (`#139 <https://github.com/MESMER-group/mesmer/pull/139>`_).
     By `Mathias Hauser`_.
   - Have ``mesmer.stats.fit_auto_regression`` return the variance instead of the standard deviation (
     `#306 <https://github.com/MESMER-group/mesmer/issues/306>`_, and
     `#318 <https://github.com/MESMER-group/mesmer/pull/318>`_). By `Mathias Hauser`_.
   - Add ``draw_auto_regression_correlated`` and ``draw_auto_regression_uncorrelated``: to draw samples of a
     (spatially-)correlated and uncorrelated auto regression model (
     `#322 <https://github.com/MESMER-group/mesmer/pull/322>`_,
     `#161 <https://github.com/MESMER-group/mesmer/pull/161>`_ and
     `#313 <https://github.com/MESMER-group/mesmer/pull/313>`_).
     By `Mathias Hauser`_.
   - Add ``mesmer.stats.select_ar_order`` to select the order of an auto regressive model
     (`#176 <https://github.com/MESMER-group/mesmer/pull/176>`_).
     By `Mathias Hauser`_.

- Extracted functions dealing with the spatial covariance and its localization:
   - Add xarray wrappers :py:func:`mesmer.stats.adjust_covariance_ar1`
     and :py:func:`mesmer.stats.find_localized_empirical_covariance`
     (`#191 <https://github.com/MESMER-group/mesmer/pull/191>`__).
     By `Mathias Hauser`_.
   - Refactor and extract numpy-based functions dealing with the spatial covariance and its localization
     (`#167 <https://github.com/MESMER-group/mesmer/pull/167>`__ and `#184
     <https://github.com/MESMER-group/mesmer/pull/184>`__).
     By `Mathias Hauser`_.
   - Allow to pass `1 x n` arrays to :py:func:`mesmer.stats.adjust_covariance_ar1`
     (`#224 <https://github.com/MESMER-group/mesmer/pull/224>`__).
     By `Mathias Hauser`_.

- Update LOWESS smoothing:
   - Extract the LOWESS smoothing for xarray objects: :py:func:`mesmer.stats.lowess`.
     (`#193 <https://github.com/MESMER-group/mesmer/pull/193>`_,
     `#283 <https://github.com/MESMER-group/mesmer/pull/283>`_, and
     `#285 <https://github.com/MESMER-group/mesmer/pull/285>`_).
     By `Mathias Hauser`_.
   - Allow to pool data along a dimension to estimate the LOWESS smoothing.
     (`#331 <https://github.com/MESMER-group/mesmer/pull/331>`_).
     By `Mathias Hauser`_.

- Added helper functions to process xarray-based model data:
   - Added functions to stack regular lat-lon grids to 1D grids and unstack them again (`#217
     <https://github.com/MESMER-group/mesmer/pull/217>`_). By `Mathias Hauser`_.
   - Added functions to mask the ocean and Antarctica (
     `#219 <https://github.com/MESMER-group/mesmer/pull/219>`_ and
     `#314 <https://github.com/MESMER-group/mesmer/pull/314>`_). By `Mathias Hauser`_.
   - Added functions to calculate the weighted global mean
     (`#220 <https://github.com/MESMER-group/mesmer/pull/220>`_ and
     `#287 <https://github.com/MESMER-group/mesmer/pull/287>`_). By `Mathias Hauser`_.
   - Added functions to wrap arrays to [-180, 180) and [0, 360), respectively (`#270
     <https://github.com/MESMER-group/mesmer/pull/270>`_ and `#273
     <https://github.com/MESMER-group/mesmer/pull/273>`_). By `Mathias Hauser`_.

- The aerosol data is now automatically downloaded using `pooch <https://www.fatiando.org/pooch/latest/>`__.
  (`#267 <https://github.com/MESMER-group/mesmer/pull/267>`_). By `Mathias Hauser`_.

- Added helper functions to estimate and superimpose volcanic influence
  (`#336 <https://github.com/MESMER-group/mesmer/pull/336>`_). By `Mathias Hauser`_.

- Added additional tests for the calibration step (`#209 <https://github.com/MESMER-group/mesmer/issues/209>`_):
   - one scenario (SSP5-8.5) and two ensemble members (`#211 <https://github.com/MESMER-group/mesmer/pull/211>`_)
   - two scenarios (SSP1-2.6 and SSP5-8.5) with one and two ensemble members, respectively (`#214 <https://github.com/MESMER-group/mesmer/pull/214>`_)
   - different selection of predictor variables (tas**2 and hfds) for different scenarios (`#291 <https://github.com/MESMER-group/mesmer/pull/291>`_)

   By `Mathias Hauser`_.

- Allow passing `xr.DataArray` to ``gaspari_cohn`` (`#298 <https://github.com/MESMER-group/mesmer/pull/298>`__).
  By `Mathias Hauser`_.
- Allow passing `xr.DataArray` to ``geodist_exact`` (`#299 <https://github.com/MESMER-group/mesmer/pull/299>`__).
  By `Zeb Nicholls`_ and `Mathias Hauser`_.
- Add ``calc_gaspari_cohn_correlation_matrices`` a function to calculate Gaspari-Cohn correlation
  matrices for a range of localisation radii (`#300 <https://github.com/MESMER-group/mesmer/pull/300>`__).
  By `Zeb Nicholls`_ and `Mathias Hauser`_.
- Add a helper function to load tas and (potentially) hfds for several ESMs from cmip-ng
  archive at ETHZ (`#326 <https://github.com/MESMER-group/mesmer/pull/326>`__).
  By `Mathias Hauser`_.

Breaking changes
^^^^^^^^^^^^^^^^

- Localization radii that lead to singular matrices are now skipped (`#187 <https://github.com/MESMER-group/mesmer/issues/187>`__).
  By `Mathias Hauser`_.
- Refactor and split :py:func:`train_l_prepare_X_y_wgteq` into two functions:
  :py:func:`get_scenario_weights` and :py:func:`stack_predictors_and_targets`
  (`#143 <https://github.com/MESMER-group/mesmer/pull/143>`_).
  By `Mathias Hauser`_.
- Moved ``gaspari_cohn`` & ``calc_geodist_exact`` from ``io.load_constant_files`` to ``core.computation``
  (`#158 <https://github.com/MESMER-group/mesmer/issues/158>`_).
  By `Yann Quilcaille`_.
- The function ``mask_percentage`` has been renamed to :py:func:`core.regionmaskcompat.mask_3D_frac_approx`
  (`#202 <https://github.com/MESMER-group/mesmer/pull/202>`_).
  By `Mathias Hauser`_.
- Removed :py:func:`mesmer.io.load_constant_files.infer_interval_breaks` and the edges
  from the `lat` and `lon` dictionaries i.e., ``lon["e"]`` and ``lat["e"]``
  (`#233 <https://github.com/MESMER-group/mesmer/pull/233>`_).
  By `Mathias Hauser`_.
- Deprecated the ``reg_type`` argument to :py:func:`mesmer.io.load_constant_files.load_regs_ls_wgt_lon_lat`
  and the ``reg_dict`` argument to :py:func:`mesmer.utils.select.extract_land`. These arguments
  no longer have any affect (`#235 <https://github.com/MESMER-group/mesmer/pull/235>`_).
  By `Mathias Hauser`_.
- Removed ``ref["type"] == "first"``, i.e., calculating the anomaly w.r.t. the first
  ensemble member (`#247 <https://github.com/MESMER-group/mesmer/pull/247>`_).
  By `Mathias Hauser`_.
- Renamed ``mesmer.calibrate_mesmer._calibrate_and_draw_realisations`` to ``mesmer.calibrate_mesmer._calibrate_tas``
  (`#66 <https://github.com/MESMER-group/mesmer/issues/66>`_).
  By `Mathias Hauser`_.

Deprecations
^^^^^^^^^^^^

- The function ``mesmer.create_emulations.create_emus_gt`` has been renamed to
  :py:func:`create_emulations.gather_gt_data` (`#246 <https://github.com/MESMER-group/mesmer/pull/246>`_).
  By `Mathias Hauser`_.

- The function ``mesmer.utils.select.extract_time_period`` is now deprecated and will be
  removed in a future version. Please raise an issue if you use this function (`#243
  <https://github.com/MESMER-group/mesmer/pull/243>`_). By `Mathias Hauser`_.

Bug fixes
^^^^^^^^^

- Fix three issues with :py:func:`core.regionmaskcompat.mask_3D_frac_approx`. Note that these
  issues are only relevant if passing xarray objects and/ or masks close to the poles
  (`#202 <https://github.com/MESMER-group/mesmer/pull/202>`_ and `#218 <https://github.com/MESMER-group/mesmer/pull/218>`_).
  By `Mathias Hauser`_.

Documentation
^^^^^^^^^^^^^

- Add development/contributing docs (`#121 <https://github.com/MESMER-group/mesmer/pull/121>`_).
  By `Zeb Nicholls`_.

Internal Changes
^^^^^^^^^^^^^^^^

- Refactor the mesmer internals to use the new statistical core, employ helper functions etc.:
   - Use :py:func:`mesmer.utils.separate_hist_future` in :py:func:`mesmer.calibrate_mesmer.train_gt`
     (`#281 <https://github.com/MESMER-group/mesmer/pull/281>`_).
   - Use of :py:class:`mesmer.stats.LinearRegression` in

     - :py:func:`mesmer.calibrate_mesmer.train_gt_ic_OLSVOLC` (`#145 <https://github.com/MESMER-group/mesmer/pull/145>`_).
     - :py:func:`mesmer.create_emulations.create_emus_lv_OLS` and :py:func:`mesmer.create_emulations.create_emus_OLS_each_gp_sep`
       (`#240 <https://github.com/MESMER-group/mesmer/pull/240>`_).

  By `Mathias Hauser`_.

- Restore compatibility with regionmask v0.9.0 (`#136 <https://github.com/MESMER-group/mesmer/pull/136>`_).
  By `Mathias Hauser`_.

- Renamed the ``interpolation`` keyword of ``np.quantile`` to ``method`` changed in
  numpy v1.22.0 (`#137 <https://github.com/MESMER-group/mesmer/pull/137>`_).
  By `Mathias Hauser`_.

- Add python 3.10 and python 3.11 to list of supported versions (`#162
  <https://github.com/MESMER-group/mesmer/pull/162>`_ and `#284
  <https://github.com/MESMER-group/mesmer/pull/284>`_).
  By `Mathias Hauser`_.

- Move contents of setup.py to setup.cfg (`#169 <https://github.com/MESMER-group/mesmer/pull/169>`_).
  By `Mathias Hauser`_.

- Use pyproject.toml for the build-system and setuptools_scm for the `__version__`
  (`#188 <https://github.com/MESMER-group/mesmer/pull/188>`_).
  By `Mathias Hauser`_.

- Moved the climate model data manipulation functions (`#237 <https://github.com/MESMER-group/mesmer/issues/237>`_).
  By `Mathias Hauser`_.

v0.8.3 - 2021-12-23
-------------------

New Features
^^^^^^^^^^^^

- Add ``mesmer.stats._linear_regression`` (renamed to ``mesmer.stats._fit_linear_regression_np``
  in `#142 <https://github.com/MESMER-group/mesmer/pull/142>`_). Starts the process of
  refactoring the codebase (`#116 <https://github.com/MESMER-group/mesmer/pull/116>`_).
  By `Zeb Nicholls`_.

Bug fixes
^^^^^^^^^

- Initialize ``llh_max`` to ``-inf`` to ensure the cross validation loop is entered
  (`#110 <https://github.com/MESMER-group/mesmer/pull/110>`_).
  By `Jonas Schwaab`_.

Documentation
^^^^^^^^^^^^^

- Fix copyright notice and release version in documentation
  (`#127 <https://github.com/MESMER-group/mesmer/pull/127>`_).
  By `Zeb Nicholls`_.

Internal Changes
^^^^^^^^^^^^^^^^

- Automatically upload the code coverage to codecov.io after the test suite has run
  (`#99 <https://github.com/MESMER-group/mesmer/pull/99>`_).
  By `Mathias Hauser`_.
- Internal refactor: moved a number of inline comments to their own line (especially if
  this allows to have the code on one line instead of several) and other minor cleanups
  (`#98 <https://github.com/MESMER-group/mesmer/pull/98>`_).
  By `Mathias Hauser`_.
- Refactor ``io.load_cmipng_tas`` and ``io.load_cmipng_hfds`` to
  de-duplicate their code and add tests for them
  (`#55 <https://github.com/MESMER-group/mesmer/pull/55>`_).
  By `Mathias Hauser`_.


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
  By `Mathias Hauser`_.
- Refactor and speed up of the Gaspari-Cohn function and the calculation of the great
  circle distance (`#85 <https://github.com/MESMER-group/mesmer/pull/85>`_,
  `#88 <https://github.com/MESMER-group/mesmer/pull/88>`_).
  By `Mathias Hauser`_.
- The geopy package is no longer a dependency of mesmer
  (`#88 <https://github.com/MESMER-group/mesmer/pull/88>`_).
  By `Mathias Hauser`_.
- Convert README from Markdown to reStructuredText to fix package build errors. Also
  allows to include the README in the docs to avoid duplication
  (`#102 <https://github.com/MESMER-group/mesmer/issues/102>`_).
  By `Mathias Hauser`_.

v0.8.1 - 2021-07-15
-------------------

- Update example script (`#80 <https://github.com/MESMER-group/mesmer/pull/80>`_).

v0.8.0 - 2021-07-13
-------------------

- First release on PyPI and conda
  (`#79 <https://github.com/MESMER-group/mesmer/pull/79>`_).

.. _`Jonas Schwaab`: https://github.com/jschwaab
.. _`Mathias Hauser`: https://github.com/mathause
.. _`Yann Quilcaille`: https://github.com/yquilcaille
.. _`Zeb Nicholls`: https://github.com/znicholls
