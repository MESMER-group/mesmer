#!/usr/bin/env bash

# forcibly remove packages to avoid artifacts
conda uninstall -y --force \
  dask \
  nc-time-axis \
  numpy \
  packaging \
  pandas \
  pooch \
  properscoring \
  pyogrio \
  regionmask \
  scikit-learn \
  scipy \
  statsmodels \
  xarray

# keep cartopy & matplotlib: we don't have tests that use them
# keep joblib: we want to move away from pickle files
# keep netcdf4: difficult to build

# to limit the runtime of Upstream CI
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    statsmodels \
    xarray
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/fatiando/pooch \
    git+https://github.com/geopandas/geopandas \
    git+https://github.com/properscoring/properscoring \
    git+https://github.com/pydata/xarray \
    git+https://github.com/pypa/packaging \
    git+https://github.com/pyproj4/pyproj \
    git+https://github.com/regionmask/regionmask \
    git+https://github.com/SciTools/nc-time-axis
