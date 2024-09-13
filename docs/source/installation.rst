Installation
============

Required dependencies
---------------------

- Python (3.10 or later)
- `dask <https://dask.org/>`__
- `numpy <http://www.numpy.org/>`__
- `pandas <https://pandas.pydata.org/>`__
- `regionmask <https://regionmask.readthedocs.io/en/stable/>`__
- `pooch <https://www.fatiando.org/pooch/latest/>`__
- `pyproj <https://pyproj4.github.io/pyproj/stable/>`__
- `scikit-learn <https://scikit-learn.org/stable/>`__
- `scipy <https://scipy.org/>`__
- `statsmodels <https://www.statsmodels.org/stable/index.html>`__
- `xarray <http://xarray.pydata.org/>`__

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
