.. development:

Contributing to MESMER
======================
.. contents:: :local:

Overview
--------
Thanks for your interest in contributing to MESMER, we're excited to have you on board!
This section of the docs details how to get up to contribute and how best to communicate.

All contributions are welcome, some possible suggestions include:

- tutorials (or support questions which, once solved, result in a new tutorial :D)
- blog posts
- improving the documentation
- bug reports
- feature requests
- pull requests

Please report issues or discuss feature requests in the `MESMER issue tracker`_.

As a contributor, please follow a couple of conventions:

- Create issues in the `MESMER issue tracker`_ for changes and enhancements, this ensures that everyone in the community has a chance to comment
- Be welcoming to newcomers and encourage diverse new contributors from all backgrounds: see the `Python Community Code of Conduct <https://www.python.org/psf/codeofconduct/>`_
- Only push to your own branches, this allows people to force push to their own branches as they need without fear or causing others headaches
- Start all pull requests as draft pull requests and only mark them as ready for review once they've had main merged into them, this makes it much simpler for reviewers
- Several small pull requests are preferred over one large PR, this makes it easier for reviewers and faster for everyone as review time grows exponentially with the number of lines in a pull request

Development Workflow
--------------------

Getting started
~~~~~~~~~~~~~~~
We are using Github to manage the MESMER codebase. If you are new to git or github, go and checkout the resources linked in the `Development tools`_ section.
Here is what you need to do to get started with MESMER:

#. **Fork the Repository**: Fork the `MESMER repository <https://github.com/MESMER-group/mesmer>`_ to your GitHub account (click the `fork` button on the MESMER landing page). Now you have a personal copy of the MESMER repository in your GitHub account.
#. **Clone the Repository**: Clone your forked repository to your local machine. For this step you must be able to establish a connection from your local machine to your GitHub account. To this end, it might be necessary to set up an SSH key, please consult the `GitHub documentation <https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh>`_ for more details.

    .. code-block:: bash
        
      git clone https://github.com/yourusername/mesmer

You should now have a `mesmer` folder in your current directory.
- Switch to the `mesmer` directory: `cd mesmer`
#. **Create an environment and activate it** to work with MESMER, the steps for this are listed in the `Development setup`_ section.
This installs MESMER in development mode and all dependencies needed to use and develop MESMER.

Now you are ready to work on MESMER. If you want to contribute to the codebase, you can sumbit a pull request (PR) to the MESMER repository. On how to do this,
pleas consult the section `Pull Request Process`_.

Development setup
~~~~~~~~~~~~~~~~~
To get setup as a developer, we recommend the following steps (if any of these tools are unfamiliar, please see the resources we recommend in `Development tools`_):

#. Install conda and the Make tool
#. Run ``make conda-environment``, if that fails you can try doing it manually

    #. Change your current directory to MESMER's root directory (i.e. the one which contains ``README.rst``), ``cd mesmer``
    #. Create a conda environment to use with MESMER ``conda create --name mesmer``
    #. Activate your conda environment ``conda activate mesmer``
    #. Install the development dependencies (very important, make sure your conda environment is active before doing this) ``conda install -y --file environment.yml && pip install -e .[dev]``

#. Make sure that mesmer was correctly installed and the tests pass by running ``make test``, if that fails the commands are

    #. Activate your conda environment ``conda activate mesmer``
    #. Run the unit and integration tests ``pytest --cov -r a --cov-report term-missing``

Pull Request Process
~~~~~~~~~~~~~~~~~~~~
If you want to contribute new features, fixes or other changes to the MESMER codebase, you can do so by submitting a pull request (PR) to the MESMER repository.
Please follow the steps below to submit a PR after having set up MESMER lcoally, following the steps in `Getting started`_:

#. **Create a Branch**: Create a new branch for your feature or bugfix.

    .. code-block:: bash

      git checkout -b your-feature

#. **Make Changes**: Implement your changes in the new branch.
#. **Commit Changes**: Commit your changes with a clear and descriptive message.

    .. code-block:: bash

      git commit -m "Description of your changes"

#. **Push to GitHub**: Push your changes to your forked repository.

    .. code-block:: bash

      git push origin your-feature

    `origin` is the default name of the remote repository you cloned from, so in this case your forked repository.
#. **Create a Pull Request**: Open a pull request on the `MESMER repository <https://github.com/MESMER-group/mesmer>`_ 
    on GitHub by clicking on "Compare and pull request" on the PR page.
#. **Review Process**: each pull request needs approval from a core contributer.
    Please be available for comments and discussion about your contribution to make sure your changes
    can be implemented.
    
    ​Potentially, some things change in the main repository change while your PR is reviewed / you are
    working on it. Please regularly update your main remotely and locally. Remotely you can do this
    by clicking on `sync` in your fork. Afterwards go to you local main and do:

    .. code-block:: shell

      git pull --rebase origin main
      git switch your-feature
      git merge main

#. **Merge**. After a successful review your request can be merged (by clicking on the merge button under
    the pull request webpage) :tada: :tada:
#. After the merge, **delete** the PR from your remote and local repository.
    For your remote you can just klick delete under your merged PR, locally you should switch to main and

    .. code-block:: shell

      git branch -D your-feature

    and update your main remotely (go onto your fork and click `sync`, and then to this locally:

    .. code-block:: shell
        
      git pull --rebase origin main


If you want to contribute more, please open a **new** branch and reiterate the steps above.

Getting help
~~~~~~~~~~~~

Whilst developing, unexpected things can go wrong (that's why it's called 'developing', if we knew what we were doing, it would already be 'developed').
Normally, the fastest way to solve an issue is to contact us via the `MESMER issue tracker`_.
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

    - we use `GitHub actions <https://docs.github.com/en/actions/quickstart>`_ for our CI but there are a number of good options

- `Jupyter Notebooks <https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46>`_

    - Jupyter is automatically included in your virtual environment if you follow our `Development setup`_ instructions

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

Please ensure that any new functionality is covered by tests.
When writing tests, we try to put them in one of two categories: integration and unit tests.
- unit tests test the functionality of each function - check you function actually does what you intend it to do by testing on (simple) examples
- integration tests test for numerical reproducability - write tests that will flag when someone makes numerically alterning changes to your code.
Note that we want to keep the data needed to be shipped with MESMER to a minimum. Please consider reusing the datasets already included in MESMER
to test numerical stability.

Try to keep the test files targeted and fairly small. You can always create
`fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`__ to aid code reuse.
The aim is to avoid testing files with thousands of lines of code as such files quickly become hard to
rationalise or understand.
Please frquently run the tests to ensure your changes do not break existing functionality.

.. code-block:: shell
    
    pytest tests/unit/test_yourtest.py

Formatting
----------

To help us focus on what the code does, not how it looks, we use a couple of automatic formatting tools.
We use the following tools:
- `ruff check <https://docs.astral.sh/ruff/>`_ to check and fix small code errors
- `black <https://github.com/psf/black>`_ to auto-format the code

These automatically format the code for us and tell use where the errors are.
To use them, after setting yourself up (see `Development setup`_), simply run ``make format``.
Note that ``make format`` can only be run if you have committed all your work i.e. your working directory is 'clean'.
This restriction is made to ensure that you don't format code without being able to undo it, just in case something goes wrong.


Building the docs
-----------------

After setting yourself up (see `Development setup`_), building the docs is as simple as running ``make docs`` (note, run ``make -B docs`` to force the docs to rebuild and ignore make when it says '... index.html is up to date').
This will build the docs for you.
You can preview them by opening ``docs/build/html/index.html`` in a browser.

For documentation we use Sphinx_.
To get ourselves started with Sphinx, we started with `this example <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ then used `Sphinx's getting started guide <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.

Please update the documentation to reflect any changes or additions to the code. Follow the structure and style of the existing documentation and lastly,
update the `CHANGELOG` with your changes.

Docstring style
~~~~~~~~~~~~~~~

For our docstrings we use numpy style docstrings.
For more information on these, `here is the full guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and `the quick reference we also use <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

Why is there a ``Makefile`` in a pure Python repository?
--------------------------------------------------------

Whilst it may not be standard practice, a ``Makefile`` is a simple way to automate general setup (environment setup in particular).
Hence we have one here which basically acts as a notes file for how to do all those little jobs which we often forget e.g. setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxiliary bits and pieces.

.. _Sphinx: http://www.sphinx-doc.org
.. _MESMER issue tracker: https://github.com/MESMER-group/mesmer/issues
