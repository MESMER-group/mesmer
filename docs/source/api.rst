.. currentmodule:: mesmer

#############
API reference
#############

This page provides an auto-generated summary of mesmers' API.


Top-level functions
===================

Statistical functions
---------------------

.. autosummary::
   :toctree: generated/

   ~stats.linear_regression.LinearRegression
   ~stats.linear_regression.LinearRegression.fit
   ~stats.linear_regression.LinearRegression.predict
   ~stats.linear_regression.LinearRegression.residuals
   ~stats.linear_regression.LinearRegression.to_netcdf
   ~stats.linear_regression.LinearRegression.from_netcdf
   ~stats.auto_regression._select_ar_order_xr
   ~stats.auto_regression._fit_auto_regression_xr
   ~stats.auto_regression._draw_auto_regression_correlated_np
   ~stats.smoothing.lowess

Computation
-----------

.. autosummary::
   :toctree: generated/

   ~core.computation.calc_geodist_exact
   ~core.computation.gaspari_cohn

Data manipulation
-----------------

.. autosummary::
   :toctree: generated/

   ~xarray_utils.grid.stack_lat_lon
   ~xarray_utils.grid.unstack_lat_lon_and_align
   ~xarray_utils.grid.unstack_lat_lon
   ~xarray_utils.grid.align_to_coords
   ~xarray_utils.mask.mask_ocean_fraction
   ~xarray_utils.mask.mask_ocean
   ~xarray_utils.mask.mask_antarctica
   ~xarray_utils.global_mean.lat_weights
   ~xarray_utils.global_mean.weighted_mean

Train mesmer
------------

.. autosummary::
   :toctree: generated/

   ~calibrate_mesmer.train_gt
   ~calibrate_mesmer.train_gv
   ~calibrate_mesmer.train_lt
   ~calibrate_mesmer.train_lv

Create emulations
-----------------

.. autosummary::
   :toctree: generated/

   ~create_emulations.create_emus_gt
   ~create_emulations.create_emus_gv
   ~create_emulations.create_emus_lt
   ~create_emulations.create_emus_lv
   ~create_emulations.create_emus_g
   ~create_emulations.create_emus_l
   ~create_emulations.make_realisations


Individual methods and utils
----------------------------

Train mesmer
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ~calibrate_mesmer.train_gv_AR
   ~calibrate_mesmer.train_gt_ic_LOWESS
   ~calibrate_mesmer.train_gt_ic_OLSVOLC
   ~calibrate_mesmer.train_lv_AR1_sci
   ~calibrate_mesmer.train_lv_find_localized_ecov
   ~calibrate_mesmer.get_scenario_weights
   ~calibrate_mesmer.stack_predictors_and_targets

Create emulations
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ~create_emulations.create_emus_gv_AR
   ~create_emulations.create_emus_OLS_each_gp_sep
   ~create_emulations.create_emus_lv_AR1_sci
   ~create_emulations.create_emus_lv_OLS


IO
--

Load constant files
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ~io.load_constant_files.infer_interval_breaks
   ~io.load_constant_files.load_phi_gc
   ~io.load_constant_files.load_regs_ls_wgt_lon_lat


Load output
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ~io.load_mesmer.load_mesmer_output


Load observations
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ~io.load_obs.load_obs
   ~io.load_obs.load_obs_tblend
   ~io.load_obs.load_strat_aod

Save mesmer bundle
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ~io.save_mesmer_bundle.save_mesmer_bundle


Utils
-----

.. autosummary::
   :toctree: generated/

   ~utils.convert.convert_dict_to_arr
   ~utils.convert.separate_hist_future
   ~utils.select.extract_land
   ~utils.select.extract_time_period
   ~utils.regionmaskcompat import mask_3D_frac_approx
