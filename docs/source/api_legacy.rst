.. currentmodule:: mesmer

Legacy API reference
####################

This page provides an auto-generated summary of mesmers' API.

Train mesmer
============

.. autosummary::
   :toctree: generated/

   ~calibrate_mesmer.train_gt
   ~calibrate_mesmer.train_gv
   ~calibrate_mesmer.train_lt
   ~calibrate_mesmer.train_lv

Create emulations
=================

.. autosummary::
   :toctree: generated/

   ~create_emulations.gather_gt_data
   ~create_emulations.create_emus_gv
   ~create_emulations.create_emus_lt
   ~create_emulations.create_emus_lv
   ~create_emulations.create_emus_g
   ~create_emulations.create_emus_l
   ~create_emulations.make_realisations
   ~create_emulations.create_seed_dict


Individual methods and utils
============================

Train mesmer
------------

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
-----------------

.. autosummary::
   :toctree: generated/

   ~create_emulations.create_emus_gv_AR
   ~create_emulations.create_emus_OLS_each_gp_sep
   ~create_emulations.create_emus_lv_AR1_sci
   ~create_emulations.create_emus_lv_OLS


IO
==

Load cmip data
--------------

.. autosummary::
   :toctree: generated/

   ~io.load_cmip_data_all_esms
   ~io.load_cmipng
   ~io.load_cmipng_hfds
   ~io.load_cmipng_tas


Load constant files
-------------------

.. autosummary::
   :toctree: generated/

   ~io.load_constant_files.load_phi_gc
   ~io.load_constant_files.load_regs_ls_wgt_lon_lat


Load output
-----------

.. autosummary::
   :toctree: generated/

   ~io.load_mesmer.load_mesmer_output


Load observations
-----------------

.. autosummary::
   :toctree: generated/

   ~io.load_obs.load_obs
   ~io.load_obs.load_obs_tblend
   ~io.load_obs.load_strat_aod

Save mesmer bundle
------------------

.. autosummary::
   :toctree: generated/

   ~io.save_mesmer_bundle.save_mesmer_bundle


Utils
=====

.. autosummary::
   :toctree: generated/

   ~utils.convert.convert_dict_to_arr
   ~utils.separate_hist_future
   ~utils.select.extract_land
   ~utils.select.extract_time_period

