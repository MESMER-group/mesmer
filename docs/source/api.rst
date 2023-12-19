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

   ~stats.LinearRegression
   ~stats.LinearRegression.fit
   ~stats.LinearRegression.predict
   ~stats.LinearRegression.residuals
   ~stats.LinearRegression.to_netcdf
   ~stats.LinearRegression.from_netcdf

Auto regression
---------------

.. autosummary::
   :toctree: generated/

   ~stats._select_ar_order_scen_ens
   ~stats._fit_auto_regression_scen_ens
   ~stats.select_ar_order
   ~stats.fit_auto_regression
   ~stats.draw_auto_regression_uncorrelated
   ~stats.draw_auto_regression_correlated

Localized covariance
--------------------

.. autosummary::
   :toctree: generated/

   ~stats.adjust_covariance_ar1
   ~stats.find_localized_empirical_covariance

Smoothing
---------

.. autosummary::
   :toctree: generated/

   ~stats.lowess


Gaspari-Cohn correlation matrix
-------------------------------

.. autosummary::
   :toctree: generated/

   ~stats.gaspari_cohn_correlation_matrices
   ~stats.gaspari_cohn

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

Geospatial
----------

.. autosummary::
   :toctree: generated/

   ~core.geospatial.geodist_exact


Emulator functions
==================

Volcanic influence
------------------

.. autosummary::
   :toctree: generated/

   ~core.volc.fit_volcanic_influence
   ~core.volc.superimpose_volcanic_influence
