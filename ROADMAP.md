# MESMER - TECHNICAL ROADMAP

The MESMER package grew organically as a science project without a strong emphasis on software design.
Therefore, there are a number of small and large improvements that should be envisioned to pave the way for future growth of MESMER.
Here, we outline desired changes which could help to put MESMER on a good foundation.
Many of these changes are intricately connected with each other, therefore we think that rewriting large parts of MESMER is the most efficient way forward.

## Envisioned updates

### User experience

Running MESMER currently involves a mixture of data loading, data preprocessing, defining the statistical model for calibration, storing the determined model parameters, as well as drawing realizations. We hope to improve on this by addressing the following points:

- Clearer separation of individual steps from data loading and preprocessing.
- More flexible and powerful array structures.
- Better documentation, including examples and tutorials.


### Internal data structures

There are two levels of internal data structures used within MESMER: (i) an array data structure, and (ii) a high level structure. The array data corresponds to climate data from individual models, with dimensions, while the high level data structure comprises data from different models, variables and scenarios.

#### Array data structure

MESMER currently uses [numpy](https://numpy.org/) to represent its array data structure.
While numpy is the de-facto standard of array computing today, MESMER can profit from labels which encode information about how the array values map to locations in space, time, etc..
For example a 3 dimensional array can be augmented with named dimensions (`time`, `latitude`, `longitude`) and these dimensions can be augmented with coordinates (e.g. the first time step corresponds to 1850).
Such data structures are provided by the well established [xarray](http://xarray.pydata.org/en/stable/) library, which adds named dimensions, coordinates and attributes on top of raw numpy-like arrays and is particularly tailored to working with netCDF files.
We therefore plan to replace the array data structure with xarray data structures.

Specific tasks
- Rewrite the statistical calibration and emulation functions such that they work with xarray arrays.
- Define how the internal arrays in MESMER should be represented within xarray ([#105](https://github.com/MESMER-group/mesmer/issues/105)).
- Rewrite the internal handling of coordinates and dimensions. Currently time, latitude, longitude etc are passed as individual variables - using xarray this will no longer be necessary.

#### High level data structure

Currently MESMER uses nested dictionaries as high level data structure. For example global mean temperature for a specific model and scenario is available as `GSAT["IPSL-CM6A-LR"]["ssp126"]`. This data structure is not very flexible (e.g. if there are no scenarios one level is superfluous while emulating a GCM-RCM chain might need another level, see [#113](https://github.com/MESMER-group/mesmer/issues/113)). A flat high level data structure would allow to circumvent these problems. However, it is currently not entirely clear how this high level data structure should look like, see discussion in [#106](https://github.com/MESMER-group/mesmer/issues/106).

Specific tasks
- Decide on a (preliminary) high level structure.
- Implement a prototype to see how the data handling within MESMER would work with new structure to assess how this makes using MESMER easier.

### Disentangle the data handling from the statistical functions

Currently, most functions in MESMER combine data handling (e.g. splitting historical data and projections), statistical calibration (e.g. estimating linear regression coefficients), and saving the estimated parameters. These three steps should be split into individual functions, which will offer several benefits: (i) It should become easier to understand what happens in each step of the calibration/ emulation pipeline. (ii) The individual code parts become easier to reuse and update. (iii) Each part can be tested individually.

Specific tasks
- Extract common data handling tasks and write individual functions for each of them ([#109](https://github.com/MESMER-group/mesmer/pull/109)).
- Extract statistical methods (e.g. linear regression at each grid point) and ensure interoperability with the array data format.
- Define an appropriate output format for each statistical task (see also the section _Persisting calibrated parameters_).

Note that these tasks are highly dependent on the _Internal data structures_.

### Persisting calibrated parameters and emulations

Currently, calibrated parameters and emulations are stored in a python-internal format (the [pickle](https://docs.python.org/3/library/pickle.html) format via [`joblib.dump`](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html)). The current format is problematic from a security perspective (see the [pickle documentation](https://docs.python.org/3/library/pickle.html)) and can pose problems with backward compatibility ([#50](https://github.com/MESMER-group/mesmer/issues/50)).

The calibrated parameters (e.g. the spatially-explicit linear regression coefficients) and need to be saved in a more standard format, i.e. as netCDFs ([#65](https://github.com/MESMER-group/mesmer/issues/65)). Note that switching the internal _Array data structure_ to xarray will greatly simplify this task because they can be directly saved as netCDF files.

Specific tasks
- Return emulations and calibrated parameters as xarray Dataset, which can directly be stored as netCDF.
- Define how calibration-specific configuration should be saved, e.g. within the netCDF or external to the netCDF in a file.
- Define how to save global parameters that are calibration-independent, e.g. the configuration, see also _Configuration infrastructure_ see and [#65 (comment)](https://github.com/MESMER-group/mesmer/issues/65#issuecomment-959123361)).

### Configuration infrastructure

The current configuration relies on a python file which includes all available settings. The settings are passed to each function, there is no default configuration, and the configuration file also calculates some values ([#34](https://github.com/MESMER-group/mesmer/issues/34)).

It would be beneficial to re work the current setup of the configuration. The configuration should be a static file (e.g. in yaml format) and global default values should be set where appropriate.


### Increase code test coverage

Adding tests to the code ensures that the behavior of the code is as expected. It also guards against unintentional and unexpected changes when updating the code. Further, code tests can help to think about the code structure and design to make individual parts as independent from each other as possible (else it's difficult to test). The current structure of MESMER makes it difficult to add tests because all the parts are very tightly linked.

### Remove ETH specific code paths

Code to load data is currently written to assuming the ETH data structure ([#61](https://github.com/MESMER-group/mesmer/issues/61)). At some point we should allow the users to hold the data however they want, load it how they want and then have interfaces into which they pass the data.
