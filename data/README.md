# Data Provenance

Auxiliary data for mesmer.

## Stratospheric aerosol optical depth

The stratospheric aerosol optical depth data is sourced from [NASA](https://data.giss.nasa.gov/modelforce/strataer/)
and licensed as [Creative Commons Zero](https://creativecommons.org/public-domain/cc0/) (see [NASA - Science Data Licenses](https://science.data.nasa.gov/about/license), Point 2)

The dataset is an extenstion of Sato et al. ([1993](https://doi.org/10.1029/93JD02553)), see also references on the NASA web page.


## CMIP6

For integration tests and example notebooks we use data from the sixth phase of the Coupled
Model Intercomparison Project (CMIP6) (Eyring et al., [2016](https://doi.org/10.5194/gmd-9-1937-2016)).
Historical simulations (1850-2014) are combined with the shared socioeconomic pathways
(SSPs) projections (O'Neill et al., [2016](https://doi.org/10.5194/gmd-9-3461-2016))
for the years 2015 to 2100.
We disribute adapted data from the model IPSL-CM6A-LR model (Boucher et al., [2018](https://doi.org/10.22033/ESGF/CMIP6.1534), [2019](https://doi.org/10.22033/ESGF/CMIP6.1532)) under the terms of its license. This data was initially (2018-03-14) published under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, which was later (2022-06-03) relaxed to a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license (overview of [CMIP6 models'](https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_source_id_licenses.html) license).
The data has undergone two processing steps (1) it was quality controlled, harmonized and regridded to a common 2.5° grid according to the cmip6-next-generation (cmip6_ng) protocal
(Brunner et al., [2020](https://doi.org/10.5281/zenodo.3734128)), and then (2) regridded to a very coarse resolution of 18° x 9° (i.e. 20 x 20 grid points).


## Acknowledgement

We acknowledge the World Climate Research Programme, which, through its Working Group on
Coupled Modelling, coordinated and promoted CMIP6. We thank the climate modeling groups
for producing and making available their model output, the Earth System Grid Federation
(ESGF) for archiving the data and providing access, and the multiple funding agencies
who support CMIP6 and ESGF. We thank Urs Beyerle for downloading and curating the
CMIP6 data at ETH Zurich, and Lukas Brunner and Ruth Lorenz for processing the data.


## References

* Boucher, O., Denvil, S., Levavasseur, G., Cozic, A., Caubel, A., Foujols, MA., Meurdesoif, Y., Cadule, P., Devilliers, M., Dupont, E., Lurton, T.,  (2019). *IPSL IPSL-CM6A-LR model output prepared for CMIP6 ScenarioMIP.* Version YYYYMMDD[1].Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.1532
* Boucher, O., Denvil, S., Levavasseur, G., Cozic, A., Caubel, A., Foujols, MA., Meurdesoif, Y., Cadule, P., Devilliers, M., Ghattas, J., Lebas, N., Lurton, T., Mellul, L., Musat, I., Mignot, J., Cheruy, F., (2018). *IPSL IPSL-CM6A-LR model output prepared for CMIP6 CMIP.* Version YYYYMMDD[1].Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.1534
* Brunner, L., Hauser, M., Lorenz, R., & Beyerle, U. (2020). *The ETH Zurich CMIP6 next generation archive: Technical documentation.* https://doi.org/10.5281/zenodo.3734128
* Eyring, V., Bony, S., Meehl, G. A., Senior, C. A., Stevens, B., Stouffer, R. J., & Taylor, K. E. (2016). Overview of the coupled model intercomparison project phase 6 (CMIP6) experimental design and organization. Geoscientific Model Development, 9(5), 1937–1958. https://doi.org/10.5194/gmd-9-1937-2016
* O'Neill, B. C., Tebaldi, C., van Vuuren, D. P., Eyring, V., Friedlingstein, P., Hurtt, G., Knutti, R., Kriegler, E., Lamarque, J.-F., Lowe, J., Meehl, G. A., Moss, R., Riahi, K., & Sanderson, B. M. (2016). The scenario model intercomparison project (ScenarioMIP) for CMIP6. Geoscientific Model Development, 9(9), 3461–3482. https://doi.org/10.5194/gmd-9-3461-2016
* Sato, M., Hansen, J.E., McCormick, M.P. and Pollack, J.B., 1993. Stratospheric aerosol optical depths, 1850–1990. Journal of Geophysical Research: Atmospheres, 98(D12), pp.22987-22994. https://doi.org/10.1002/2013MS000266
