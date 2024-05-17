Installation
============

Required dependencies
---------------------

- Python (3.9 or later)
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

   python -m pip install mesmer-emulator[full]

Development installation
------------------------

See instructions on github: `#315 <https://github.com/MESMER-group/mesmer/issues/315>`__.
