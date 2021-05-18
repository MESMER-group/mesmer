Installation
============

Required dependencies
---------------------

- Python (3.6 or later)
- `dask <https://dask.org/>`__
- `geopy <https://geopy.readthedocs.io/en/stable/>`__
- `numpy <http://www.numpy.org/>`__ (1.17 or later)
- `pandas <https://pandas.pydata.org/>`__
- `scikit-learn <https://scikit-learn.org/stable/>`__
- `statsmodels <https://www.statsmodels.org/stable/index.html>`__
- `regionmask <https://regionmask.readthedocs.io/en/stable/>`__
- `xarray <http://xarray.pydata.org/>`__ (0.15 or later)

Instructions
------------

mesmer is a pure Python package, but its dependencies are not. As mesmer is not yet
available from PyPI it's recommended is to first install the dependencies using conda/ mamba

.. code-block:: bash

    conda install -c conda-forge dask geopy numpy pandas scikit statsmodels regionmask xarray

and afterwards install mesmer directly from github:

.. code-block:: bash

   python -m pip install git+https://github.com/MESMER-group/mesmer.git

To install mesmer in development mode first clone it using git and install it afterwards:

.. code-block:: bash

   git clone git+https://github.com/MESMER-group/mesmer.git
   python -m pip install -e mesmer

.. _conda: http://conda.io/
