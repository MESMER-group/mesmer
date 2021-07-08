Installation
============

Required dependencies
---------------------

- Python (3.7 or later)
- `dask <https://dask.org/>`__
- `geopy <https://geopy.readthedocs.io/en/stable/>`__
- `numpy <http://www.numpy.org/>`__
- `pandas <https://pandas.pydata.org/>`__
- `scikit-learn <https://scikit-learn.org/stable/>`__
- `statsmodels <https://www.statsmodels.org/stable/index.html>`__
- `regionmask <https://regionmask.readthedocs.io/en/stable/>`__
- `xarray <http://xarray.pydata.org/>`__

Instructions
------------

mesmer is a pure Python package, but its dependencies are not. As a result, we recommend
installing mesmer's dependencies using conda/mamba e.g.

.. code-block:: bash

    conda install -c conda-forge dask geopy numpy pandas scikit statsmodels regionmask xarray pip

and afterwards install mesmer from pypi:

.. code-block:: bash

   pip install mesmer-emulator

or install mesmer directly from github:

.. code-block:: bash

   pip install git+https://github.com/MESMER-group/mesmer.git

To install mesmer in development mode first clone it using git and install it afterwards:

.. code-block:: bash

   git clone git+https://github.com/MESMER-group/mesmer.git
   pip install -e mesmer

.. _conda: http://conda.io/
