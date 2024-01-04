Installation
============

Required dependencies
---------------------

- Python (3.9 or later)
- `dask <https://dask.org/>`__
- `numpy <http://www.numpy.org/>`__
- `pandas <https://pandas.pydata.org/>`__
- `scikit-learn <https://scikit-learn.org/stable/>`__
- `statsmodels <https://www.statsmodels.org/stable/index.html>`__
- `regionmask <https://regionmask.readthedocs.io/en/stable/>`__
- `xarray <http://xarray.pydata.org/>`__

Instructions
------------

mesmer is a pure Python package, but its dependencies are not. As a result, we recommend
installing mesmer using conda/mamba:

.. code-block:: bash

    conda install -c conda-forge mesmer

Alternately, mesmer can be installed with pip (but we make no guarantees about
the simplicity of installing mesmer's dependencies using pip)

.. code-block:: bash

   python -m pip install mesmer-emulator

Otherwise, mesmer can be installed directly from github

.. code-block:: bash

   python -m pip install git+https://github.com/MESMER-group/mesmer.git

To install mesmer in development mode, first clone it using git and then
install it in an editable mode afterwards:

.. code-block:: bash

   git clone git+https://github.com/MESMER-group/mesmer.git
   python -m pip install -e mesmer

.. _conda: http://conda.io/
