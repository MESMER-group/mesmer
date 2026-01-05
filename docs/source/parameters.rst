Pre-calibrated Parameters
=========================

We provide pre-calibrated parameters for the emulation of near surface temperature (see tutorial `Emulating near surface temperature on land with MESMER`_) for 58 CMIP6 models.
The parameters can be downloaded from Zenodo: https://doi.org/10.5281/zenodo.17250327.

Calibration
-----------

The parameters have been calibrated for 58 Earth System Models contributing to the Coupled Model Intercomparison Project Phase 6 (Eyring et al., `2016 <https://doi.org/10.5194/GMD-9-1937-2016>`__), using the near surface temperature of all available initial condition members of the shared socioeconomic pathways (SSPs) SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0 and SSP5-8.5 and their matching historical members. The model data has been preprocessed according to Brunner et al. (`2020 <https://doi.org/10.5281/zenodo.3734128>`__). The version of MESMER used to calibrate the parameters is v1.0.0rc1.
The script to calibrate the parameters is available on Zenodo: https://doi.org/10.5281/zenodo.17264436 as `calibrate_mesmer_CMIP6.py`.

References
----------
* Eyring, V., Bony, S., Meehl, G. A., Senior, C. A., Stevens, B., Stouffer, R. J., & Taylor, K. E. (2016). Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization. Geoscientific Model Development, 9(5), 1937–1958. https://doi.org/10.5194/GMD-9-1937-2016
* Brunner, L., Hauser, M., Lorenz, R., & Beyerle, U. (2020). The ETH Zurich CMIP6 next generation archive: Technical documentation (Version v1.0-final). Zenodo. https://doi.org/10.5281/zenodo.3734128


.. _Emulating near surface temperature on land with MESMER: tutorials/tutorial_mesmer_emulating_multiple_scenarios.html
