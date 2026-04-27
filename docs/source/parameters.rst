Calibrated Parameters
=========================

We provide calibrated parameters for several variables:

* MESMER:
    * annual mean air temperature (tas)
* MESMER-M:
    * monthly mean air temperature (tas)
* MESMER-X:
    * annual maximum of daily maximum air temperature (txx)
    * annual average of soil moisture (mrso)
    * annual minimum of the monthly average soil moisture (mrso_minmon)
    * annual maximum of the Fire Weather Index (fwixx)
    * seasonal average of the Fire Weather Index (fwisa)
    * annual number of days with extreme fire weather (fwixd)
    * length of the fire season (fwils)

The parameters can be downloaded from the ETH Research Collection (Quillcaille et al., `2026 <https://doi.org/10.3929/ethz-c-000798034>`__) and used under the `Creative Commons Attribution 4.0 International License <http://creativecommons.org/licenses/by/4.0/>`__.

Calibration
-----------

The parameters have been calibrated for 58 Earth System Models contributing to the Coupled Model Intercomparison Project Phase 6 (Eyring et al., `2016 <https://doi.org/10.5194/GMD-9-1937-2016>`__), using the near surface temperature of all available initial condition members of the shared socioeconomic pathways (SSPs) SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP4-6.0, SSP5-8.5, and SSP5-3.4-OS and their matching historical members as predictor. The model data has been preprocessed according to Brunner et al. (`2020 <https://doi.org/10.5281/zenodo.3734128>`__). The version of MESMER used to calibrate the parameters is v1.0.0rc1.
The scripts to calibrate the parameters are available on Zenodo: (Bauer et al., `2026 <https://doi.org/10.5281/zenodo.19697078>`__).

References
----------
* Quilcaille, Y., Bauer, V., Hauser, M., Schöngart, S., Gudmundsson, L., & Seneviratne, S. I. (2026). Parametrizations for MESMER v1.0.0 [Application/zip,application/netcdf]. ETH Zurich. https://doi.org/10.3929/ETHZ-C-000798034
* Eyring, V., Bony, S., Meehl, G. A., Senior, C. A., Stevens, B., Stouffer, R. J., & Taylor, K. E. (2016). Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization. Geoscientific Model Development, 9(5), 1937–1958. https://doi.org/10.5194/GMD-9-1937-2016
* Brunner, L., Hauser, M., Lorenz, R., & Beyerle, U. (2020). The ETH Zurich CMIP6 next generation archive: Technical documentation (Version v1.0-final). Zenodo. https://doi.org/10.5281/zenodo.3734128
* Bauer, V., Quilcaille, Y., & Hauser, M. (2026). Analysis scripts for MESMER v1.0.0: Consolidating the Modular Earth System Model Emulator into a Sustainable Research Software Package (Version v4). Zenodo. https://doi.org/10.5281/zenodo.19697078
