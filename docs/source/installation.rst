Installation
============

Required dependencies
---------------------

- Python (3.11 or later)
- `dask <https://www.dask.org/>`__
- `filefisher <https://filefisher.readthedocs.io/en/latest/>`__
- `joblib <https://joblib.readthedocs.io/en/latest/>`__
- `netcdf4 <https://unidata.github.io/netcdf4-python/>`__
- `numpy <https://numpy.org>`__
- `pandas <https://pandas.pydata.org/>`__
- `pooch <https://www.fatiando.org/pooch/latest/>`__
- `pyproj <https://pyproj4.github.io/pyproj/stable/>`__
- `regionmask <https://regionmask.readthedocs.io/en/stable/>`__
- `scikit-learn <https://scikit-learn.org/stable/>`__
- `scipy <https://scipy.org/>`__
- `statsmodels <https://www.statsmodels.org/stable/index.html>`__
- `xarray <https://docs.xarray.dev/en/stable/>`__

Please note that we only support datatree version 0.0.13 and will switch to the xarray internal datatree module in the future.

Optional dependencies
---------------------

- `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`__
- `matplotlib <https://matplotlib.org/>`__
- `nc-time-axis <https://nc-time-axis.readthedocs.io/en/stable/>`__
- `properscoring <https://pypi.org/project/properscoring/>`__

Instructions
------------

mesmer is a pure Python package, but its dependencies are not. As a result, we recommend
installing mesmer using conda/mamba:

.. code-block:: bash

    conda install -c conda-forge mesmer

Alternately, mesmer can be installed with pip:

.. code-block:: bash

   python -m pip install mesmer-emulator[complete]

Development installation
------------------------

See instructions under `development`_.

.. _development: development.html
