.. currentmodule:: mesmer

#############
API reference
#############

This page provides an auto-generated summary of mesmers' API.


Top-level functions
===================

Statistical core functions
--------------------------

.. autosummary::
   :toctree: generated/

   ~core.linear_regression.LinearRegression
   ~core.linear_regression.LinearRegression.fit
   ~core.linear_regression.LinearRegression.predict
   ~core.linear_regression.LinearRegression.residuals
   ~core.linear_regression.LinearRegression.to_netcdf
   ~core.linear_regression.LinearRegression.from_netcdf
   ~core.auto_regression._select_ar_order_xr
   ~core.auto_regression._fit_auto_regression_xr
   ~core.auto_regression._draw_auto_regression_correlated_np

Computation
-----------

.. autosummary::
   :toctree: generated/

   ~core.computation.calc_geodist_exact
   ~core.computation.gaspari_cohn

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
   ~io.load_constant_files.mask_percentage


Load constant files
^^^^^^^^^^^^^^^^^^^

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
