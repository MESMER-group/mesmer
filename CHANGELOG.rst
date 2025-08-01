Changelog
=========

v0.11.0 - unreleased
--------------------

New Features
^^^^^^^^^^^^
- Implemented new data structure using ``xr.DataTree``, see `Data structure using DataTree`_.
- Integrated MESMER-X into the code base, see `Integration of MESMER-X`_.
- Integrated MESMER-M into the code base, see `Integration of MESMER-M`_.
- Added number of observations to the output of the AR process (`#395 <https://github.com/MESMER-group/mesmer/pull/395>`_).
  By `Victoria Bauer`_.
- Add python 3.13 to list of supported versions (`#547 <https://github.com/MESMER-group/mesmer/pull/547>`_).
  By `Mathias Hauser`_.
- Passing ``hist_period`` to the volcaninc helper functions is no longer needed (\
  `#649 <https://github.com/MESMER-group/mesmer/pull/649>`_). By `Mathias Hauser`_.
- Can now pass ``only`` to ``LinearRegression.predict`` to select predictors
  (`#702 <https://github.com/MESMER-group/mesmer/issues/702>`_, and
  `#717 <https://github.com/MESMER-group/mesmer/pull/717>`_).
  By `Mathias Hauser`_.
- Added :py:class:`set_options` to mesmer which can, currently, be used to control
  the number of used threads for matrix decomposition
  (`#349 <https://github.com/MESMER-group/mesmer/issues/349>`_, and
  `#713 <https://github.com/MESMER-group/mesmer/pull/713>`_).
  By `Mathias Hauser`_.
- Enable passing data with a dimension without coordinates (i.e. ``sample`` dimension)
  to ``localized_empirical_covariance`` (`#710 <https://github.com/MESMER-group/mesmer/pull/710>`_).
  By `Mathias Hauser`_.
- Added :py:meth:`LinearRegression.from_params` as a one-step way to initialize a linear regression
  (`#761 <https://github.com/MESMER-group/mesmer/pull/761>`_).
  By `Mathias Hauser`_.

Breaking changes
^^^^^^^^^^^^^^^^
- Switch random number generation for drawing emulations from :py:func:`np.random.seed()` to :py:func:`np.random.default_rng()`
  (`#495 <https://github.com/MESMER-group/mesmer/pull/495>`_). By `Victoria Bauer`_.
- Using Cholesky decomposition for finding covariance localization radius and drawing from the multivariate normal distribution (`#408 <https://github.com/MESMER-group/mesmer/pull/408>`_)
  By `Victoria Bauer`_.
- Removed support for python 3.9 and python 3.10
  (`#513 <https://github.com/MESMER-group/mesmer/pull/513>`_, and `#733 <https://github.com/MESMER-group/mesmer/pull/733>`_)
  By `Mathias Hauser`_.
- Removed the deprecated function :py:func:`mask_percentage` (`#654 <https://github.com/MESMER-group/mesmer/pull/654>`_)
  By `Mathias Hauser`_.
- The supported versions of some dependencies were changed
  (`#399 <https://github.com/MESMER-group/mesmer/pull/399>`_,
  `#405 <https://github.com/MESMER-group/mesmer/pull/405>`_,
  `#503 <https://github.com/MESMER-group/mesmer/pull/503>`_,
  `#621 <https://github.com/MESMER-group/mesmer/pull/621>`_,
  `#627 <https://github.com/MESMER-group/mesmer/pull/627>`_,
  `#683 <https://github.com/MESMER-group/mesmer/pull/683>`_,
  `#686 <https://github.com/MESMER-group/mesmer/pull/686>`_, and
  `#740 <https://github.com/MESMER-group/mesmer/pull/740>`_):

  ================= ============= =========
  Package           Old           New
  ================= ============= =========
  **cartopy**       not specified 0.23
  **dask**          not specified 2024.7
  **filefisher**    not required  1.1
  **joblib**        not specified 1.4
  **netcdf4**       not specified 1.7
  **numpy**         not specified 1.26
  **packaging**     not specified 24.1
  **pandas**        2.0           2.2
  **pooch**         not specified 1.8
  **properscoring** not specified 0.1
  **pyproj**        not specified 3.6
  **regionmask**    0.8           0.12
  **scikit-learn**  not specified 1.5
  **scipy**         not specified 1.14
  **shapely**       not specified 2.0
  **statsmodels**   not specified 0.14
  **xarray**        2023.04       2025.03
  ================= ============= =========

Deprecations
^^^^^^^^^^^^

- Deprecated ``mask_3D_frac_approx`` as the functionality is now offered in regionmask
  v0.12.0 (`#451 <https://github.com/MESMER-group/mesmer/pull/451>`_).

Bug fixes
^^^^^^^^^
- Averaging standard deviations for the AR parameters of global variability over several ensemble members and scenarios now averages the
  variances (`#499 <https://github.com/MESMER-group/mesmer/pull/499>`_).
  By `Victoria Bauer`_.

Documentation
^^^^^^^^^^^^^
- Updated and extended the development Guide (`#511 <https://github.com/MESMER-group/mesmer/pull/511>`_, `#523 <https://github.com/MESMER-group/mesmer/pull/523>`_)
- Added example notebooks for calibrating on multiple scenarios and ensemble members and emulating multiple scenarios (`#521 <https://github.com/MESMER-group/mesmer/pull/521>`_).

Internal Changes
^^^^^^^^^^^^^^^^

- Start testing the minimum versions of required dependencies (`#398 <https://github.com/MESMER-group/mesmer/pull/398>`_).
  By `Mathias Hauser`_.
- Restore compatibility with pandas version v2.2 and xarray version v2024.02 (`#404 <https://github.com/MESMER-group/mesmer/pull/404>`_).
  By `Mathias Hauser`_.
- Explicitly include all required dependencies (`#448 <https://github.com/MESMER-group/mesmer/pull/448>`_).
- Unshallow the mesmer git repository on rtd (`#456 <https://github.com/MESMER-group/mesmer/pull/456>`_).
  By `Victoria Bauer`_.
- Use ruff instead of isort and flake8 to lint the code base (`#490 <https://github.com/MESMER-group/mesmer/pull/490>`_).
  By `Mathias Hauser`_.
- Consolidate package metadata and configuration in `pyproject.toml` (`#650 <https://github.com/MESMER-group/mesmer/pull/650>`_).
  By `Mathias Hauser`_.
- Made the :py:func:`create_equal_dim_names` private (`#653 <https://github.com/MESMER-group/mesmer/pull/653>`_).
  By `Mathias Hauser`_.
- Removed the ``regionmaskcompat.py`` module. It is no longer needed after requiring *regionmask* v0.12  (`#683 <https://github.com/MESMER-group/mesmer/pull/683>`_).
  By `Mathias Hauser`_.

Data structure using DataTree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This release uses :py:class:`xr.DataTree` as data structure to handle multiple scenarios.
This was originally done with the prototype `xarray-datatree` package. After the port of
``DataTree`` to xarray stabilized, mesmer briefly supported both ``DataTree`` versions
(the one in xarray-datatree and in xarray) before dropping support for `xarray-datatree`.

- Switch to storing several predictors in one :py:class:`xr.Dataset`` per scenario node in a :py:class:`DataTree` (`#677 <https://github.com/MESMER-group/mesmer/pull/677>`_).
- Enable passing a :py:class:`DataTree` to the auto regression functions (`#570 <https://github.com/MESMER-group/mesmer/pull/570>`_, `#677 <https://github.com/MESMER-group/mesmer/pull/677>`_).
- Enable passing :py:class:`DataTree` and :py:class:`xr.Dataset` to :py:class:`LinearRegression`
  (`#566 <https://github.com/MESMER-group/mesmer/pull/566>`_, and
  `#720 <https://github.com/MESMER-group/mesmer/pull/720>`_).
- Add weighting function for several scenarios (`#567 <https://github.com/MESMER-group/mesmer/pull/567>`_).
- Add function to compute anomalies over several scenarios stored in a :py:class:`DataTree` (`#625 <https://github.com/MESMER-group/mesmer/pull/625>`_).
- Add utility functions for :py:class:`DataTree` (`#556 <https://github.com/MESMER-group/mesmer/pull/556>`_).
- Add a wrapper to allow :py:class:`DataTree` in many data handling functions (\
  `#632 <https://github.com/MESMER-group/mesmer/issues/632>`_,
  `#643 <https://github.com/MESMER-group/mesmer/pull/643>`_,
  `#641 <https://github.com/MESMER-group/mesmer/pull/641>`_,
  `#644 <https://github.com/MESMER-group/mesmer/pull/644>`_, and
  `#682 <https://github.com/MESMER-group/mesmer/pull/682>`_).
- Add calibration integration tests for multiple scenarios and change parameter files to netcdfs with new naming structure (`#537 <https://github.com/MESMER-group/mesmer/pull/537>`_)
- Add new integration tests for drawing realisations (`#599 <https://github.com/MESMER-group/mesmer/pull/599>`_)
- Add helper function to merge ``DataTree`` objects  (`#701 <https://github.com/MESMER-group/mesmer/pull/701>`_)
- PRs related to xarray and xarray-datatree:

  - Add `xarray-datatree` as dependency (`#554 <https://github.com/MESMER-group/mesmer/pull/554>`_)
  - Add upper pin to `xarray` version to support `xarray-datatree` (`#559 <https://github.com/MESMER-group/mesmer/pull/559>`_).
  - Port the functionality to xarray's :py:class:`DataTree` implementation (`#607 <https://github.com/MESMER-group/mesmer/pull/607>`_).
  - Drop support for `xarray-datatree`  (`#627 <https://github.com/MESMER-group/mesmer/pull/627>`_).
- Add `filefisher` as dependency to handle file paths of several scenarios (\
  `#586 <https://github.com/MESMER-group/mesmer/pull/586>`_,
  `#592 <https://github.com/MESMER-group/mesmer/pull/592>`_, and
  `#629 <https://github.com/MESMER-group/mesmer/pull/629>`_).

By `Victoria Bauer`_ and `Mathias Hauser`_.

Integration of MESMER-X
^^^^^^^^^^^^^^^^^^^^^^^

In the release the MESMER-X functionality is integrated into the MESMER Codebase.

- Add MESMER-X functionality to the code base (`#432 <https://github.com/MESMER-group/mesmer/pull/432>`_)
- Overall restructuring and refactoring of the MESMER-X code base (`#645 <https://github.com/MESMER-group/mesmer/pull/645>`_)
- Some general refactoring and clean-up (`#437 <https://github.com/MESMER-group/mesmer/pull/437>`_,
  `#465 <https://github.com/MESMER-group/mesmer/pull/465>`_,
  `#466 <https://github.com/MESMER-group/mesmer/pull/466>`_,
  `#467 <https://github.com/MESMER-group/mesmer/pull/467>`_,
  `#468 <https://github.com/MESMER-group/mesmer/pull/468>`_,
  `#469 <https://github.com/MESMER-group/mesmer/pull/469>`_,
  `#470 <https://github.com/MESMER-group/mesmer/pull/470>`_,
  `#502 <https://github.com/MESMER-group/mesmer/pull/502>`_)
- Add unit tests (`#526 <https://github.com/MESMER-group/mesmer/pull/526>`_,
  `#533 <https://github.com/MESMER-group/mesmer/pull/533>`_,
  `#534 <https://github.com/MESMER-group/mesmer/pull/534>`_,
  `#540 <https://github.com/MESMER-group/mesmer/pull/540>`_,
  `#577 <https://github.com/MESMER-group/mesmer/pull/577>`_)
- Add integration tests (`#524 <https://github.com/MESMER-group/mesmer/pull/524>`_,
  `#550 <https://github.com/MESMER-group/mesmer/pull/550>`_,
  `#553 <https://github.com/MESMER-group/mesmer/pull/553>`_)
- Enable to pass set values for loc and scale (only integers) and make scale parameter optional (`#597 <https://github.com/MESMER-group/mesmer/pull/597>`_).
- Enable ``threshold_min_proba`` to be ``None`` in :py:class:`distrib_cov` (`#598 <https://github.com/MESMER-group/mesmer/pull/598>`_).
- Also use Nelder-Mead fit in :py:meth:`distrib_cov._minimize` for ``option_NelderMead == "best_run"`` when Powell fit was not successful (`#600 <https://github.com/MESMER-group/mesmer/pull/600>`_).
- Return `logpmf` for discrete distributions in :py:meth:`distrib_cov._fg_fun_nll_cubed()` (`#602 <https://github.com/MESMER-group/mesmer/pull/602>`_).
- Speed-up MESMER-X

  - add method to calculate params of a distribution (`#539 <https://github.com/MESMER-group/mesmer/pull/539>`_)
  - avoiding frozen distributions (`#532 <https://github.com/MESMER-group/mesmer/issues/532>`_)
  - not broadcasting scalars (`#613 <https://github.com/MESMER-group/mesmer/pull/613>`_)
  - compiling the expression (`#614 <https://github.com/MESMER-group/mesmer/pull/614>`_).

By `Yann Quilcaille`_ with `Victoria Bauer`_ and `Mathias Hauser`_.

Integration of MESMER-M
^^^^^^^^^^^^^^^^^^^^^^^

This release integrates MESMER-M into the existing MESMER infrastructure. This includes
some refactoring, bugfixes and enhancements of the MESMER-M functionality. Note
that this led to some numerical changes compared to the MESMER-M publication
(Nath et al., `2022 <https://doi.org/10.5194/esd-13-851-2022>`_).

- move MESMER-M scripts into mesmer (\
  `#419 <https://github.com/MESMER-group/mesmer/pull/419>`_, and
  `#421 <https://github.com/MESMER-group/mesmer/pull/421>`_).
- move the harmonic model and power transformer functionalities to the stats module (\
  `#484 <https://github.com/MESMER-group/mesmer/pull/484>`_).
- add example script for MESMER-M workflow (`#491 <https://github.com/MESMER-group/mesmer/pull/491>`_)
- add integration tests for MESMER-M (`#501 <https://github.com/MESMER-group/mesmer/pull/501>`_)
- enable calibrating MESMER-M on several scenarios and ensemble members (`#678 <https://github.com/MESMER-group/mesmer/issues/678>`_)
  and add an example (`#572 <https://github.com/MESMER-group/mesmer/pull/572>`_).

Auto-Regression
~~~~~~~~~~~~~~~

- Implement functions performing the monthly (cyclo-stationary) auto-regression and adapt these functions to
  work with xarray. This includes extracting the drawing of spatially correlated innovations to a
  stand-alone function. (`#473 <https://github.com/MESMER-group/mesmer/pull/473>`_)
- Remove the bounds of -1 and 1 on the slope of the cyclo-stationary AR(1) process. This bound is not necessary
  since cyclo-stationarity is also given if the slopes of a few months are (slightly) larger than one. We
  now return the residuals of the cyclo-stationary AR(1) process to fit the covariance matrix on these residuals.
  As a consequence, adjustment of the covariance matrix with the AR slopes is no longer necessary.
  (`#480 <https://github.com/MESMER-group/mesmer/pull/480>`_)
  Compare discussion in `#472 <https://github.com/MESMER-group/mesmer/issues/472>`_.
- Implement function to localize the empirical covarince matrix for each month individually to use in drawing
  of spatially correlated noise in the AR process. (`#479 <https://github.com/MESMER-group/mesmer/pull/479>`_)
- Enable passing data with a dimension without coordinates (i.e. ``sample`` dimension)
  to ``fit_auto_regression_monthly`` (`#706 <https://github.com/MESMER-group/mesmer/pull/706>`_).
- Ensure residuals are ordered correctly in `fit_auto_regression_monthly` (`#708 <https://github.com/MESMER-group/mesmer/pull/708>`_).

Yeo-Johnson power transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Ensure the power transformer yields the correct normalization for more cases (\
   `#440 <https://github.com/MESMER-group/mesmer/issues/440>`_):

   -  expand the upper bound of the first coefficient from :math:`1` to :math:`\infty`,
      i.e. to 1e10  (\
      `#446 <https://github.com/MESMER-group/mesmer/pull/446>`_, `#501 <https://github.com/MESMER-group/mesmer/pull/501>`_)
   -  remove jacobian ``rosen_der`` from fit (\
      `#447 <https://github.com/MESMER-group/mesmer/pull/447>`_)
   -  change optimization method from *SLSQP* to *Nelder-Mead* (\
      `#455 <https://github.com/MESMER-group/mesmer/pull/455>`_)
-  adjust the first guess to assume the data is normally distributed (\
   `#429 <https://github.com/MESMER-group/mesmer/pull/429>`_)
-  make (back-) transformations more stable by using :py:func:`np.expm1` and :py:func:`np.log1p`
   (`#494 <https://github.com/MESMER-group/mesmer/pull/494>`_)
-  rewrite power transformer to work with xarray, and refactor from a class structure to functions (\
   `#442 <https://github.com/MESMER-group/mesmer/pull/442>`_, and
   `#474 <https://github.com/MESMER-group/mesmer/pull/474>`_)
-  fix small code issues and clean the docstrings (\
   `#436 <https://github.com/MESMER-group/mesmer/pull/436>`_,
   `#444 <https://github.com/MESMER-group/mesmer/pull/444>`_,
   `#439 <https://github.com/MESMER-group/mesmer/pull/439>`_,
   `#475 <https://github.com/MESMER-group/mesmer/pull/475>`_, and
   `#425 <https://github.com/MESMER-group/mesmer/pull/425>`_)
- add tests (`#430 <https://github.com/MESMER-group/mesmer/pull/430>`_)
- Converted Yeo-Johnson power transformer functions back into a class, which allows to
  add additional lambda functions (`#716 <https://github.com/MESMER-group/mesmer/pull/716>`_).
- Added a constant lambda function (`#718 <https://github.com/MESMER-group/mesmer/pull/718>`_).
- Enable passing data with a dimension without coordinates (i.e. ``sample`` dimension)
  to power transformer functions (`#703 <https://github.com/MESMER-group/mesmer/pull/703>`_).

Harmonic model
~~~~~~~~~~~~~~

-  Performance and other optimizations:

   - only fit orders up to local minimum and use coeffs from previous order as first guess (`#443 <https://github.com/MESMER-group/mesmer/pull/443>`_)
   - infer the harmonic model order from the coefficients (`#434 <https://github.com/MESMER-group/mesmer/pull/434>`_)
   - optimization of `_generate_fourier_series_order_np()` (`#516 <https://github.com/MESMER-group/mesmer/pull/516>`_ and `#578 <https://github.com/MESMER-group/mesmer/pull/578>`_)
-  return residuals instead of the loss for the optimization (`#460 <https://github.com/MESMER-group/mesmer/pull/460>`_)
-  remove fitting of linear regression with yearly temperature (`#415 <https://github.com/MESMER-group/mesmer/pull/415>`_ and
   `#488 <https://github.com/MESMER-group/mesmer/pull/488>`_) in line with (`Nath et al. 2022 <https://doi.org/10.5194/esd-13-851-2022>`_).
-  add helper function to upsample yearly data to monthly resolution (\
   `#418 <https://github.com/MESMER-group/mesmer/pull/418>`_,
   `#435 <https://github.com/MESMER-group/mesmer/pull/435>`_, and
   `#688 <https://github.com/MESMER-group/mesmer/pull/688>`_).
- de-duplicate the expression of months in their harmonic form (`#415 <https://github.com/MESMER-group/mesmer/pull/415>`_)
  move creation of the month array to the deepest level (`#487 <https://github.com/MESMER-group/mesmer/pull/487>`_).
- fix indexing of harmonic model coefficients (`#415 <https://github.com/MESMER-group/mesmer/pull/415>`_)
-  Refactor variable names, small code improvements, optimization, fixes and clean docstring
   (`#415 <https://github.com/MESMER-group/mesmer/pull/415>`_,
   `#424 <https://github.com/MESMER-group/mesmer/pull/424>`_,
   `#433 <https://github.com/MESMER-group/mesmer/pull/433>`_,
   `#512 <https://github.com/MESMER-group/mesmer/pull/512>`_,
   `#574 <https://github.com/MESMER-group/mesmer/pull/574>`_, and
   `#589 <https://github.com/MESMER-group/mesmer/issues/589>`_).
- add tests (\
  `#431 <https://github.com/MESMER-group/mesmer/pull/431>`_, and
  `#458 <https://github.com/MESMER-group/mesmer/pull/458>`_)
- add function to generate fourier series using xarray (`#478 <https://github.com/MESMER-group/mesmer/pull/478>`_)
- Enable passing data with a dimension without coordinates (i.e. ``sample`` dimension)
  to harmonic model functions (`#705 <https://github.com/MESMER-group/mesmer/pull/705>`_).

By `Victoria Bauer`_ and `Mathias Hauser`_.

Data
^^^^

- Directly source the stratospheric aerosol optical depth data from NASA instead of using
  the version from Climate Explorer (`#665 <https://github.com/MESMER-group/mesmer/pull/665>`_).
  By `Mathias Hauser`_.

v0.10.0 - 2024.01.04
--------------------

version 0.10.0 fixes the bug in the legacy calibration and is thus not numerically
backward compatible. It also updated the supported python, pandas and xarray versions.
Updating the pandas version will create an error when trying to load pickled mesmer
bundles, requiring to use mesmer version v0.9.0 for this.

Bug fixes
^^^^^^^^^

Ensure de-duplicating the historical ensemble members conserves their order. Previously,
the legacy calibration routines used ``np.unique``, which shuffles them. See `#338
<https://github.com/MESMER-group/mesmer/issues/338>`_ for details.
(`#339 <https://github.com/MESMER-group/mesmer/pull/339>`_).
By `Mathias Hauser`_.

Breaking changes
^^^^^^^^^^^^^^^^

- Removed support for python 3.7 and python 3.8 (\
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
   - Add xarray wrapper for fitting a linear regression (\
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
   - Fixed two bugs related to (non-dimension) coordinates (\
     `#332 <https://github.com/MESMER-group/mesmer/issues/332>`_,
     `#333 <https://github.com/MESMER-group/mesmer/issues/333>`_ and
     `#334 <https://github.com/MESMER-group/mesmer/pull/313>`_).
     By `Mathias Hauser`_.

- Extracted statistical functionality for auto regression:
   - Add ``mesmer.stats.fit_auto_regression``: xarray wrapper to fit an auto regression model
     (`#139 <https://github.com/MESMER-group/mesmer/pull/139>`_).
     By `Mathias Hauser`_.
   - Have ``mesmer.stats.fit_auto_regression`` return the variance instead of the standard deviation (\
     `#306 <https://github.com/MESMER-group/mesmer/issues/306>`_, and
     `#318 <https://github.com/MESMER-group/mesmer/pull/318>`_). By `Mathias Hauser`_.
   - Add ``draw_auto_regression_correlated`` and ``draw_auto_regression_uncorrelated``: to draw samples of a
     (spatially-)correlated and uncorrelated auto regression model (\
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
   - Added functions to mask the ocean and Antarctica (\
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
.. _`Victoria Bauer`: https://github.com/veni-vidi-vici-dormivi
