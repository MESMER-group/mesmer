.. development:

Development
===========

Thanks for your interest in contributing to MESMER, we're excited to have you on board!
This section of the docs details how to get up to contribute and how best to communicate.

.. contents:: :local:

Contributing
------------

All contributions are welcome, some possible suggestions include:

- tutorials (or support questions which, once solved, result in a new tutorial :D)
- blog posts
- improving the documentation
- bug reports
- feature requests
- pull requests

Please report issues or discuss feature requests in the `MEMSER issue tracker`_.
If your issue is a feature request or a bug, please use the templates available, otherwise, simply open a normal issue :)

As a contributor, please follow a couple of conventions:

- Create issues in the `MEMSER issue tracker`_ for changes and enhancements, this ensures that everyone in the community has a chance to comment
- Be welcoming to newcomers and encourage diverse new contributors from all backgrounds: see the `Python Community Code of Conduct <https://www.python.org/psf/codeofconduct/>`_
- Only push to your own branches, this allows people to force push to their own branches as they need without fear or causing others headaches
- Start all pull requests as draft pull requests and only mark them as ready for review once they've been rebased onto master, this makes it much simpler for reviewers
- Several small pull requests are preferred over one large PR, this makes it easier for reviewers and faster for everyone as review time grows exponentially with the number of lines in a pull request

Development setup
-----------------

To get setup as a developer, we recommend the following steps (if any of these tools are unfamiliar, please see the resources we recommend in `Development tools`_):

#. Install conda and the Make tool
#. Run ``make conda-environment``, if that fails you can try doing it manually

    #. Change your current directory to MESMER's root directory (i.e. the one which contains ``README.rst``), ``cd mesmer``
    #. Create a conda environment to use with MESMER ``conda create --name mesmer``
    #. Activate your conda environment ``conda activate mesmer``
    #. Install the development dependencies (very important, make sure your conda environment is active before doing this) ``conda install -y --file environment.yml && pip install --upgrade pip wheel && pip install -e .[dev]``

#. Make sure the tests pass by running ``make test``, if that fails the commands are

    #. Activate your conda environment ``conda activate mesmer``
    #. Run the unit and integration tests ``pytest --cov -r a --cov-report term-missing``

Getting help
~~~~~~~~~~~~

Whilst developing, unexpected things can go wrong (that's why it's called 'developing', if we knew what we were doing, it would already be 'developed').
Normally, the fastest way to solve an issue is to contact us via the `MEMSER issue tracker`_.
The other option is to debug yourself.
For this purpose, we provide a list of the tools we use during our development as starting points for your search to find what has gone wrong.

Development tools
+++++++++++++++++

This list of development tools is what we rely on to develop MESMER reliably and reproducibly.
It gives you a few starting points in case things do go inexplicably wrong and you want to work out why.
We include links with each of these tools to starting points that we think are useful, in case you want to learn more.

- `Git <http://swcarpentry.github.io/git-novice/>`_

- `Make <https://swcarpentry.github.io/make-novice/>`_

- `Conda virtual environments <https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c>`_

- `Tests <https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest>`_

    - we use a blend of `pytest <https://docs.pytest.org/en/latest/>`_ and the inbuilt Python testing capabilities for our tests so checkout what we've already done in ``tests`` to get a feel for how it works

- `Continuous integration (CI) <https://docs.travis-ci.com/user/for-beginners/>`_

    - we use `Travis CI <https://travis-ci.com/>`_ for our CI but there are a number of good providers

- `Jupyter Notebooks <https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46>`_

    - Jupyter is automatically included in your virtual environment if you follow our `Getting setup`_ instructions

- Sphinx_

Other tools
+++++++++++

We also use some other tools which aren't necessarily the most familiar.
Here we provide a list of these along with useful resources.

- Mocking in tests (see e.g. `this intro <https://www.toptal.com/python/an-introduction-to-mocking-in-python>`_, there are many more other good resources out there if you simply google "python intro to mocking")

    - note that mocking can take some time to get used to, feel free to raise questions in issues or the relevant PR

- `Regular expressions <https://www.oreilly.com/ideas/an-introduction-to-regular-expressions>`_

    - you can use `regex101.com <regex101.com>`_ to help write and check regular expressions, make sure the language is set to Python to make your life easy!

Testing philosophy
------------------

When writing tests, we try to put them in one of two categories: integration and regression.
Integration tests run bits of the code and assert the correct behaviour was achived.
Some of the integration tests might run fairly big bits of code, others will be more targeted.
Try to keep integration test files targeted and fairly small.
We can always create fixtures to aid code reuse.
The aim is to avoid testing files with thousands of lines of code as such files quickly become hard to rationalise or understand.

In contrast, regression tests run bits of the code and assert the output matches a saved, known output.
Regression tests are there to ensure that we know when outputs will change (sometimes they should change, we just want to make sure that this change is deliberate not accidental).
Regression tests don't require too much code generally, but they may run lots of the code base and hence take a little while to run.

(We are in the process of making the distinction between regression and integration tests clearer, see `#120 <https://github.com/MESMER-group/mesmer/issues/120>`_).


Formatting
----------

To help us focus on what the code does, not how it looks, we use a couple of automatic formatting tools.
We use the following tools:
- `isort <https://github.com/PyCQA/isort>`_ to sort import statements
- `black <https://github.com/psf/black>`_ to auto-format the code
- `flake8 <https://flake8.pycqa.org/en/latest/>`_ to check the format and small errors

These automatically format the code for us and tell use where the errors are.
To use them, after setting yourself up (see `Getting setup`_), simply run ``make format``.
Note that ``make format`` can only be run if you have committed all your work i.e. your working directory is 'clean'.
This restriction is made to ensure that you don't format code without being able to undo it, just in case something goes wrong.


Buiding the docs
----------------

After setting yourself up (see `Getting setup`_), building the docs is as simple as running ``make docs`` (note, run ``make -B docs`` to force the docs to rebuild and ignore make when it says '... index.html is up to date').
This will build the docs for you.
You can preview them by opening ``docs/build/html/index.html`` in a browser.

For documentation we use Sphinx_.
To get ourselves started with Sphinx, we started with `this example <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ then used `Sphinx's getting started guide <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.


Docstring style
~~~~~~~~~~~~~~~

For our docstrings we use numpy style docstrings.
For more information on these, `here is the full guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and `the quick reference we also use <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

Why is there a ``Makefile`` in a pure Python repository?
--------------------------------------------------------

Whilst it may not be standard practice, a ``Makefile`` is a simple way to automate general setup (environment setup in particular).
Hence we have one here which basically acts as a notes file for how to do all those little jobs which we often forget e.g. setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxillary bits and pieces.

.. _Sphinx: http://www.sphinx-doc.org/en/master/
.. _MEMSER issue tracker: https://github.com/MESMER-group/mesmer/issues
