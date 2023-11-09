.. currentmodule:: mesmer

API reference
#############

This page provides an auto-generated summary of mesmers' API.


Statistical functions
=====================

Linear regression
-----------------

.. autosummary::
   :toctree: generated/

   ~stats.linear_regression.LinearRegression
   ~stats.linear_regression.LinearRegression.fit
   ~stats.linear_regression.LinearRegression.predict
   ~stats.linear_regression.LinearRegression.residuals
   ~stats.linear_regression.LinearRegression.to_netcdf
   ~stats.linear_regression.LinearRegression.from_netcdf

Auto regression
---------------

.. autosummary::
   :toctree: generated/

   ~stats.auto_regression._select_ar_order_xr
   ~stats.auto_regression._fit_auto_regression_xr
   ~stats.auto_regression._draw_auto_regression_correlated_np

Localized covariance
--------------------

.. autosummary::
   :toctree: generated/

   ~stats.localized_covariance.adjust_covariance_ar1
   ~stats.localized_covariance.find_localized_empirical_covariance

Smoothing
---------

.. autosummary::
   :toctree: generated/

   ~stats.smoothing.lowess

Geo-spatial
-----------

.. autosummary::
   :toctree: generated/

   ~core.computation.calc_geodist_exact
   ~core.computation.gaspari_cohn

Data handling
=============

Grid manipulation
-----------------

.. autosummary::
   :toctree: generated/

   ~core.grid.wrap_to_180
   ~core.grid.wrap_to_360
   ~core.grid.stack_lat_lon
   ~core.grid.unstack_lat_lon_and_align
   ~core.grid.unstack_lat_lon
   ~core.grid.align_to_coords

Masking regions
---------------

.. autosummary::
   :toctree: generated/

   ~core.mask.mask_ocean_fraction
   ~core.mask.mask_ocean
   ~core.mask.mask_antarctica
   ~core.regionmaskcompat.mask_3D_frac_approx

Weighted operations: calculate global mean
------------------------------------------

.. autosummary::
   :toctree: generated/

   ~core.weighted.global_mean
   ~core.weighted.lat_weights
   ~core.weighted.weighted_mean
