.. currentmodule:: mesmer

API reference
#############

This page provides an auto-generated summary of mesmers' API.

Top-level functions
===================

.. autosummary::
   :toctree: generated/

   set_options
   get_options

Statistical functions
=====================

Linear regression
-----------------

.. autosummary::
   :toctree: generated/

   ~stats.LinearRegression
   ~stats.LinearRegression.from_params
   ~stats.LinearRegression.fit
   ~stats.LinearRegression.predict
   ~stats.LinearRegression.residuals
   ~stats.LinearRegression.to_netcdf
   ~stats.LinearRegression.from_netcdf

Auto regression
---------------

.. autosummary::
   :toctree: generated/

   ~stats.select_ar_order_scen_ens
   ~stats.fit_auto_regression_scen_ens
   ~stats.select_ar_order
   ~stats.fit_auto_regression
   ~stats.fit_auto_regression_monthly
   ~stats.draw_auto_regression_uncorrelated
   ~stats.draw_auto_regression_correlated
   ~stats.draw_auto_regression_monthly

Harmonic Model
--------------

.. autosummary::
   :toctree: generated/

   ~stats.predict_harmonic_model
   ~stats.fit_harmonic_model

Power Transformer
-----------------

.. autosummary::
   :toctree: generated/

.. autosummary::
   :toctree: generated/

   ~stats.YeoJohnsonTransformer
   ~stats.YeoJohnsonTransformer.lambda_function
   ~stats.YeoJohnsonTransformer.get_lambdas_from_covariates
   ~stats.YeoJohnsonTransformer.fit
   ~stats.YeoJohnsonTransformer.transform
   ~stats.YeoJohnsonTransformer.inverse_transform

   ~stats._power_transformer.logistic_lambda_function

Localized covariance
--------------------

.. autosummary::
   :toctree: generated/

   ~stats.adjust_covariance_ar1
   ~stats.find_localized_empirical_covariance
   ~stats.find_localized_empirical_covariance_monthly

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

Conditional distribution
========================

Define covariance structure of conditional distribution
-------------------------------------------------------

.. autosummary::
   :toctree: generated/

   ~distrib.Expression
   ~distrib.Expression.evaluate_params

Fit conditional distribution
----------------------------

.. autosummary::
   :toctree: generated/

   ~distrib.ConditionalDistribution
   ~distrib.ConditionalDistribution.find_first_guess
   ~distrib.ConditionalDistribution.fit
   ~distrib.ConditionalDistribution.compute_quality_scores
   ~distrib.ConditionalDistribution.coefficients
   ~distrib.ConditionalDistribution.from_netcdf
   ~distrib.ConditionalDistribution.to_netcdf

.. autosummary::
   :toctree: generated/


   ~distrib.MinimizeOptions

Transform conditional distribution
----------------------------------

.. autosummary::
   :toctree: generated/

   ~distrib.ProbabilityIntegralTransform
   ~distrib.ProbabilityIntegralTransform.transform


Data handling
=============

Example and test data
---------------------

.. autosummary::
   :toctree: generated/

   ~example_data.cmip6_ng_path

Grid manipulation
-----------------

.. autosummary::
   :toctree: generated/

   ~grid.wrap_to_180
   ~grid.wrap_to_360
   ~grid.stack_lat_lon
   ~grid.unstack_lat_lon_and_align
   ~grid.unstack_lat_lon
   ~grid.align_to_coords

Masking regions
---------------

.. autosummary::
   :toctree: generated/

   ~mask.mask_ocean_fraction
   ~mask.mask_ocean
   ~mask.mask_antarctica

Weighted operations: calculate global mean
------------------------------------------

.. autosummary::
   :toctree: generated/

   ~weighted.global_mean
   ~weighted.lat_weights
   ~weighted.weighted_mean
   ~weighted.equal_scenario_weights_from_datatree

Geospatial
----------

.. autosummary::
   :toctree: generated/

   ~geospatial.geodist_exact


Anomalies
---------

.. autosummary::
   :toctree: generated/

   ~anomaly.calc_anomaly

Resampling
----------

.. autosummary::
   :toctree: generated/

   ~resample.upsample_yearly_data

Emulator functions
==================

Volcanic influence
------------------

.. autosummary::
   :toctree: generated/

   ~volc.fit_volcanic_influence
   ~volc.superimpose_volcanic_influence
