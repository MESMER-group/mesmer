.. development:

Contributing to MESMER
======================
.. contents::
   :local:

Overview
--------
Thanks for your interest in contributing to MESMER! We're excited to have you on board! This section of the documentation details how to get started with contributing and how best to communicate. If something is unclear or you are stuck, never hestiate to contact the maintainers (`Mathias Hauser`_ and `Victoria Bauer`_) or open an issue on the `MESMER issue tracker`_.

All contributions are welcome. Some possible suggestions include:

- Tutorials (or support questions which, once solved, result in a new tutorial :D)
- Blog posts
- Improving the documentation
- Bug reports
- Feature requests
- Pull requests

Please report issues or discuss feature requests in the `MESMER issue tracker`_.

As a contributor, please follow a few conventions:

- Create issues in the `MESMER issue tracker`_ for changes and enhancements. This ensures that everyone in the community has a chance to comment.
- Be welcoming to newcomers and encourage diverse new contributors from all backgrounds: see the `Python Community Code of Conduct <https://www.python.org/psf/codeofconduct/>`_.
- Only push to your own branches. This allows people to force push to their own branches as they need without fear of causing others headaches.
- Never commit directly to the main branch (neither the one of main MESMER repo nor your own fork's). This is to ensure that the main branch is always stable and that all changes are reviewed.
- Start all pull requests as draft pull requests and only mark them as ready for review once they've had `main` merged into them. This makes it easier for reviewers to manage their time. If you are afraid your PR has been overlooked you can also actively assign a reviewer on the Github page of your PR.
- Several small pull requests are preferred over one large PR. This makes it easier for reviewers and faster for everyone, as review time grows exponentially with the number of lines in a pull request.

Development Workflow
--------------------

Getting started
~~~~~~~~~~~~~~~
We are using GitHub to manage the MESMER codebase. If you are new to git or GitHub, check out the resources linked in the `Development tools`_ section. Here is what you need to do to get started with MESMER:

1. **Fork the Repository**: Fork the `MESMER repository <https://github.com/MESMER-group/mesmer>`_ to your GitHub account (click the `fork` button on the MESMER landing page). Now you have a personal copy of the MESMER repository in your GitHub account.
2. **Clone the Repository**: Clone your forked repository to your local machine. For this step, you must be able to establish a connection from your local machine to your GitHub account. It might be necessary to set up an SSH key; please consult the `GitHub documentation <https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh>`_ for more details.

   .. code-block:: bash

      git clone git@github.com:yourusername/mesmer.git

   You should now have a `mesmer` folder in your current directory.
   - Switch to the `mesmer` directory: `cd mesmer`
3. **Create an environment and activate it** to work with MESMER. The steps for this are listed in the `Development setup`_ section.
   This installs MESMER in development mode and all dependencies needed to use and develop MESMER.

Now you are ready to work on MESMER. If you want to contribute to the codebase, you can submit a pull request (PR) to the MESMER repository. For instructions on how to do this, please consult the section `Pull Request Process`_.

Development setup
~~~~~~~~~~~~~~~~~
To get set up as a developer, we recommend setting up an environment that holds all the tools for developing the MESMER Codebase.
Follow the steps below (if any of these tools are unfamiliar, please see the resources we recommend in `Development tools`_):

1. Install conda (see a Guide for this `here <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_)
2. Create the environment:

   a. Change your current directory to MESMER's root directory (i.e., the one that contains ``README.rst``), ``cd mesmer``.

   b. Create a conda environment to use with MESMER: ``conda create --name mesmer``.

   c. Activate your conda environment: ``conda activate mesmer``. This is important for the next step, otherwise the packages will be installed in your base environment.

   d. Install the all the dependencies for running mesmer: ``conda install -y --file environment.yml```

   e. Install mesmer itself and packages needed for development (pytest, sphinx, etc.) ``pip install -e .[dev]``.
      The flag ``-e`` installs MESMER in development (**e**\ ditable) mode, which means that changes to the code are immediately reflected in the environment and you don't need to reload your environment to use/test your changes.

3. Make sure that MESMER was correctly installed by running the test suite:

   a. (Re)activate your conda environment just to be sure: ``conda activate mesmer``.

   b. Run the unit and integration tests: ``pytest --cov -r a --cov-report term-missing``.

Pull Request Process
~~~~~~~~~~~~~~~~~~~~
If you want to contribute new features, fixes, or other changes to the MESMER codebase, you can do so by submitting a pull request (PR) to the MESMER repository. Please follow the steps below to submit a PR after having set up MESMER locally, following the steps in `Getting started`_:

1. **Create a Branch**: Create a new branch for your feature or bugfix.

   .. code-block:: bash

      git checkout -b your-feature

2. **Make Changes**: Implement your changes in the new branch.
3. **Commit Changes**: Add and commit your changes with a clear and descriptive message.

   .. code-block:: bash
      git add changed_file
      git commit -m "Description of your changes"

4. **Push to GitHub**: Push your changes to your forked repository.

   .. code-block:: bash

      git push origin your-feature

   `origin` is the default name of the remote repository you cloned from, so in this case, your forked repository.
5. **Create a Pull Request**: Open a pull request on the `MESMER repository <https://github.com/MESMER-group/mesmer>`_ on GitHub by clicking on "Compare and pull request" on the PR page.
6. **Review Process**: Each pull request needs approval from a core contributor. Please be available for comments and discussion about your contribution to ensure your changes can be implemented.

   ​Potentially, some things change in the main repository while your PR is reviewed/you are working on it. Please regularly update your main remotely and locally. Remotely, you can do this by clicking on `sync` in your fork. Afterwards, go to your local main branch and do:

   .. code-block:: shell

      git pull origin main
      git switch your-feature
      git merge main

   Moreover, reviewers or our precommit checks might push changes to your pull request. You can pull these into your local branch by doing:

   .. code-block:: shell

      git pull --rebase origin your-feature

7. **Merge**: After a successful review, your request can be merged (by clicking on the merge button under the pull request webpage) :tada: :tada:
8. After the merge, **delete** the PR from your remote and local repository. For your remote, you can just click delete under your merged PR. Locally, you should switch to main and:

   .. code-block:: shell

      git branch -D your-feature

   And update your main remotely (go onto your fork and click `sync`, and then do this locally):

   .. code-block:: shell

      git pull origin main

If you want to contribute more, please open a **new** branch and repeat the steps above.

Getting help
~~~~~~~~~~~~
While developing, unexpected things can go wrong. Normally, the fastest way to solve an issue is to contact us via the `MESMER issue tracker`_. The other option is to debug yourself. For this purpose, we provide a list of the tools we use during our development as starting points for your search to find what has gone wrong.

Development tools
+++++++++++++++++
This list of development tools is what we rely on to develop MESMER reliably and reproducibly. It gives you a few starting points in case things do go wrong and you want to work out why. We include links with each of these tools to starting points that we think are useful, in case you want to learn more.

- `Git <http://swcarpentry.github.io/git-novice/>`_
- `Conda virtual environments <https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c>`_
- `Tests <https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest>`_ - We use a blend of `pytest <https://docs.pytest.org/en/latest/>`_ and the inbuilt Python testing capabilities for our tests. Check out what we've already done in ``tests`` to get a feel for how it works.

- `Continuous integration (CI) <https://docs.travis-ci.com/user/for-beginners/>`_ - We use `GitHub actions <https://docs.github.com/en/actions/quickstart>`_ for our CI, but there are a number of good options.

- `Jupyter Notebooks <https://medium.com/velotio-perspectives/the-ultimate-beginners-guide-to-jupyter-notebooks-6b00846ed2af>`_ - Jupyter is automatically included in your virtual environment if you follow our `Development setup`_ instructions.

- Sphinx_

- Mocking in tests (see e.g., `this intro <https://www.toptal.com/python/an-introduction-to-mocking-in-python>`_, there are many other good resources out there if you simply Google "python intro to mocking"). Note that mocking can take some time to get used to. Feel free to raise questions in issues or the relevant PR.


Testing philosophy
------------------
Please ensure that any new functionality is covered by tests. When writing tests, we try to put them in one of two categories: integration and unit tests.

- Unit tests check the functionality of each function - ensure your function actually does what you intend it to do by testing on small examples.
- Integration tests test for numerical reproducibility - write tests that will flag when someone makes numerically altering changes to your code. Note that we want to keep the data needed to be shipped with MESMER to a minimum. Please consider reusing the datasets already included in MESMER to test numerical stability.

Try to keep the test files targeted and fairly small. You can always create `fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`__ to aid code reuse. The aim is to avoid testing files with thousands of lines of code as such files quickly become hard to rationalize or understand. Please frequently run the tests to ensure your changes do not break existing functionality.

.. code-block:: shell

   pytest tests/unit/test_yourtest.py

Formatting
----------
To help us focus on what the code does, not how it looks, we use a couple of automatic formatting tools. We use the following tools:

- `ruff check <https://docs.astral.sh/ruff/>`_ to check and fix small code errors.
- `black <https://black.readthedocs.io/en/stable/>`_ to auto-format the code.

These tools automatically format the code for us and tell us where the errors are. To use them, after setting yourself up (see `Development setup`_), simply run ``make format``. Note that ``make format`` can only be run if you have committed all your work, i.e., your working directory is 'clean'. This restriction ensures that you don't format code without being able to undo it, just in case something goes wrong.

Building the docs
-----------------
After setting yourself up (see `Development setup`_), building the docs is done by running ``make docs`` (note, run ``make -B docs`` to force the docs to rebuild and ignore make when it says '... index.html is up to date'). This will build the docs for you. You can preview them by opening ``docs/build/html/index.html`` in a browser.

For documentation, we use Sphinx_. To get started with Sphinx, we began with `this example <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ and then used `Sphinx's getting started guide <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.

Please update the documentation to reflect any changes or additions to the code. Follow the structure and style of the existing documentation, and lastly, update the `CHANGELOG` with your changes.

Docstring style
~~~~~~~~~~~~~~~
For our docstrings, we use numpy style docstrings. For more information on these, `here is the full guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and `the quick reference we also use <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

Why is there a ``Makefile`` in a pure Python repository?
--------------------------------------------------------
While it may not be standard practice, a ``Makefile`` is a way to automate general setup (environment setup in particular). Hence, we have one here, which basically acts as a notes file for how to do all those little jobs we often forget, e.g., setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxiliary bits and pieces.

.. _Sphinx: http://www.sphinx-doc.org
.. _MESMER issue tracker: https://github.com/MESMER-group/mesmer/issues
.. _`Mathias Hauser`: https://github.com/mathause
.. _`Victoria Bauer`: https://github.com/veni-vidi-vici-dormivi