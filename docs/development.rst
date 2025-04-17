.. development:

Contributing to MESMER
======================
.. contents::
   :local:

Overview
--------
Thanks for your interest in contributing to MESMER! We're excited to have you on board! This section of the documentation details how to get started with contributing and how to communicate. If something is unclear or you are stuck, never hesitate to contact the maintainers (`Mathias Hauser`_ or `Victoria Bauer`_) or open an issue on the `MESMER issue tracker`_.

All contributions are welcome. Some possible suggestions include:

- Tutorials (or support questions which, once solved, result in a new tutorial)
- Examples
- Improving the documentation
- Bug reports
- Feature requests
- Pull requests

Please report issues or discuss feature requests in the `MESMER issue tracker`_.

As a contributor, please follow a few conventions:

- Create issues in the `MESMER issue tracker`_ for changes and enhancements. This ensures that everyone in the community has a chance to comment.
- Be welcoming to newcomers and encourage diverse new contributors from all backgrounds: see the `Python Community Code of Conduct <https://www.python.org/psf/codeofconduct/>`_.
- Only push to your own branches. This allows people to force push to their own branches as they need without fear of causing others headaches.
- Never commit directly to the main branch (neither the one of the main MESMER repo nor your own fork's). This is to ensure that the main branch is always stable and that all changes are reviewed. Always create a new branch for your changes and commit to that.
- Several small pull requests are preferred over one large PR. This makes it easier for reviewers and faster for everyone, as review time grows exponentially with the number of lines in a pull request.

Development Workflow
--------------------
This section details the development workflow to contribute features or changes to the codebase of MESMER.

Getting started
~~~~~~~~~~~~~~~
We are using GitHub to manage the MESMER codebase. If you are new to git or GitHub, check out the resources linked in the `Development tools`_ section. Here is what you need to do to get started with MESMER:

1. **Fork the Repository**: Fork the MESMER repository to your GitHub account by clicking the ``fork`` button on the `MESMER landing page <https://github.com/MESMER-group/mesmer>`_. Now you have a personal copy of the MESMER repository in your GitHub account.
2. **Clone the Repository**: Clone your forked repository to your local machine. For this step, you must be able to establish a connection from your local machine to your GitHub account. It might be necessary to set up an SSH key; please consult the documentation on `connection to GitHub with ssh <https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh>`_ for more details. Once set up you should be able to clone the repository with the following command:

   .. code-block:: shell

      git clone git@github.com:yourusername/mesmer.git

   You should now have a ``mesmer`` folder in your current directory. Switch to the ``mesmer`` directory: ``cd mesmer``
3. **Create an environment and activate it** to work with MESMER. The steps for this are listed in the `Development setup`_ section.
   This installs MESMER in development mode and all dependencies needed to use and develop MESMER.

Now you are ready to work with MESMER. If you want to contribute to the codebase, you can submit a pull request (PR) to the MESMER repository. For instructions on how to do this, please consult the section `Pull Request Process`_.

Development setup
~~~~~~~~~~~~~~~~~
To get set up as a developer, we recommend setting up an environment that holds all the tools for developing the MESMER codebase.
Follow the steps below (if any of these tools are unfamiliar, please see the resources we recommend in `Development tools`_):

1. Install conda (we recommend using the `conda-forge installer <https://conda-forge.org/download/>`_)
2. Create the environment:

   a. Change your current directory to MESMER's root directory (i.e., the one that contains ``README.rst``), ``cd mesmer``.

   b. Create a conda environment to use with MESMER and install all the dependencies: ``conda env create -n mesmer_dev -f environment.yml``.

   c. Activate your conda environment: ``conda activate mesmer_dev``. This is important for the next step, otherwise the packages will be installed in your base environment. Make sure the command line now says: ``(mesmer_dev) username@host:~/mesmer>``.

   d. (optional) Install additional dependencies which are not strictly necessary but useful for MESMER:

      .. code-block:: shell

         conda install ipykernel matplotlib
         python -m pip install git+https://github.com/mathause/filefinder/

   d. Install mesmer itself and packages needed for development (pytest, sphinx, etc.) ``python -m pip install -e .``.
      The flag ``-e`` installs MESMER in development (**e**\ ditable) mode, which means that changes to the code are immediately reflected in the environment and you don't need to reload your environment to use/test your changes.

3. Make sure that MESMER was correctly installed by running the test suite ``pytest . --all`` in the mesmer folder.

Pull Request Process
~~~~~~~~~~~~~~~~~~~~
If you want to contribute new features, fixes, or other changes to the MESMER codebase, you can do so by submitting a pull request (PR) to the MESMER repository. Please follow the steps below to submit a PR after having set up MESMER locally, following the steps in `Getting started`_:

1. **Create a Branch**: Create a new branch for your feature or bugfix.

   .. code-block:: shell

      git checkout -b your-feature

   Replace ``your-feature`` with a descriptive name for your branch. This name should be short and descriptive of the changes you are making. Moreover, we advise that you branch each feature branch from your main branch, so you can easily update your main branch and merge it into your feature branch if necessary and there are less conflicts than when branching a branch from another feature branch.

2. **Make Changes**: Implement your changes in the new branch.
   If you want to make sure your files are clean and adhere to our pre-commit hooks, run ``pre-commit run --all-files``. This will run all the checks we have set up for you. For an intro to pre-commit, see the `pre-commit documentation <https://pre-commit.com/>`_ and our .pre-commit-config.yml. Please also make sure that your changes are tested (see `Testing philosophy`_) and documented (see `Documenting`_).
3. **Commit Changes**: Add and commit your changes with a clear and descriptive message.

   .. code-block:: shell

      git add changed_file
      git commit -m "Description of your changes"

   You can keep adding commits until you think your feature is ready to be merged. If you are unsure about how to write a good commit message, `here is a guide <https://chris.beams.io/posts/git-commit/>`_.

4. **Push to GitHub**: Push your changes to your forked repository.

   .. code-block:: shell

      git push origin your-feature

   `origin` is the default name of the remote repository you cloned from, so in this case, your forked repository. Your changes are now on GitHub.
5. **Create a Pull Request**: Open a pull request on the `MESMER repository <https://github.com/MESMER-group/mesmer>`_ on GitHub by clicking on "Compare and pull request" either on the PR page of MESMER itself or in your own fork (a message should appear on the top of the page after you pushed). You will be prompted to give your PR a name and a short description, explaining what you did. There is also a small check list for you to fill out, asking if your PR solves any known issues from the `MESMER Issue Tracker`_, if you added test and documentation to your PR and added an entry to the `CHANGELOG`_.
6. **Review Process**: Each pull request needs approval from a core contributor. You can mark your PR as a draft if you are not ready for the review yet and actively request a review in the side bar of your PR when you are ready. Before you request a review please make sure your changes pass all tests and pre-commit checks (you will see a green check mark under your PR if they do). If you need help with this (or anything else), don't hesitate to reach out to the team by writing a comment and tagging either `Mathias Hauser`_ or `Victoria Bauer`_. Please also be available for comments and discussion about your contribution to ensure your changes can be implemented.

   ​Potentially, some things change in the main repository while your PR is reviewed/you are working on it. Please regularly update your main remotely and locally. Remotely, you can do this by clicking on ``sync`` in your fork. Afterwards, go to your local main branch and do:

   .. code-block:: shell

      git pull origin main
      git switch your-feature
      git merge main

   Moreover, reviewers or our pre-commit checks might push changes to your pull request. You can pull these into your local branch by doing:

   .. code-block:: shell

      git pull --rebase origin your-feature

7. **Merge**: After a successful review, your request can be merged (by clicking on the merge button on the pull request webpage). Yay! Your changes are now part of MESMER.
8. After the merge, **delete** the PR from your remote and local repository. For your remote, you can just click delete under your merged PR. Locally, you should switch to main and:

   .. code-block:: shell

      git branch -D your-feature

   And update your main remotely (go onto your fork and click ``sync``, and then do this locally):

   .. code-block:: shell

      git pull origin main

If you want to contribute more, please open a **new** branch and repeat the steps above. Thanks for contributing!

Getting help
~~~~~~~~~~~~
While developing, unexpected things can go wrong. Normally, the fastest way to solve an issue is to contact us via the `MESMER issue tracker`_. The other option is to debug yourself. For this purpose, we provide a list of the tools we use during our development as starting points for your search to find what has gone wrong.

Development tools
+++++++++++++++++
This list of development tools is what we rely on to develop MESMER reliably and reproducibly. It gives you a few starting points in case things do go wrong and you want to work out why. We include links with each of these tools to starting points that we think are useful, in case you want to learn more.

- `Git <http://swcarpentry.github.io/git-novice/>`_
- `Conda environments <https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c>`_
- `Tests <https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest>`_ - We use a blend of `pytest <https://docs.pytest.org/en/latest/>`_ and the inbuilt Python testing capabilities for our tests. Check out what we've already done in the ``tests/unit`` folder to get a feel for how it works.

- `Continuous integration (CI) <https://docs.travis-ci.com/user/for-beginners/>`_ - We use `GitHub actions <https://docs.github.com/en/actions/quickstart>`_ for our CI, but there are a number of good options.

- `Jupyter Notebooks <https://medium.com/velotio-perspectives/the-ultimate-beginners-guide-to-jupyter-notebooks-6b00846ed2af>`_ - Jupyter is automatically included in your virtual environment if you follow our `Development setup`_ instructions.
- Notebook Debugging: Some IDEs have debbunging capabilities for Jupyter Notebooks built in like `Visual Studio Code <https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_debug-a-jupyter-notebook>`_, `pycharm <https://www.jetbrains.com/help/pycharm/running-jupyter-notebook-cells.html>`_ or `Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/user/debugger.html>`_, but you can also use `pdb <https://docs.python.org/3/library/pdb.html>`_ or `ipdb <https://pypi.org/project/ipdb/>`_ in the terminal on python scripts.

- Sphinx_

- Mocking in tests (see e.g., `this intro <https://www.toptal.com/python/an-introduction-to-mocking-in-python>`_, there are many other good resources out there if you simply Google "python intro to mocking"). Note that mocking can take some time to get used to. Feel free to raise questions in issues or the relevant PR.


Testing philosophy
------------------
Please ensure that any new functionality is covered by tests. When writing tests, we try to put them in one of two categories: integration and unit tests.

- **Unit tests** check the functionality of each function - ensure your function actually does what you intend it to do by testing on small examples. You can look at examples of this in the `tests/unit` folder.
- **Integration tests** test for numerical reproducibility - write tests that will flag when someone makes numerically altering changes to your code. Note that we want to keep the data needed to be shipped with MESMER to a minimum. Please consider reusing the datasets already included in MESMER to test numerical stability. Have a look at the alreaedy available tests in the ``tests/integration`` folder.

Try to keep the test files targeted and fairly small. You can always create `fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`__ to aid code reuse. The aim is to avoid testing files with thousands of lines of code as such files quickly become hard to rationalize or understand. Please frequently run the tests to ensure your changes do not break existing functionality.

.. code-block:: shell

   pytest tests/unit/test_feature.py

Formatting
----------
To help us focus on what the code does, not how it looks, we use a couple of automatic formatting tools. We use the following tools:

- `ruff check <https://docs.astral.sh/ruff/>`_ to check and fix small code errors.
- `black <https://black.readthedocs.io/en/stable/>`_ to auto-format the code.

These tools automatically format the code for us and tell us where the errors are. To use them, after setting up the development environment (see `Development setup`_), run ``ruff check . --fix ; black .;``. If you run these commands after committing all your work, i.e., your working directory is 'clean'. This ensures that you don't format code without being able to undo it, just in case something goes wrong.

Documenting
-----------
We strongly encourage you to document your code. By this we mean mainting a transparent workflow via git and github and commenting your code lines but above all we want to encourage documenting your new functions via a docstring, explaining what the function does and how it can be used. This makes it easier for others to understand what you have done and how to use it.

We use Sphinx_ to generate our documentation. To get started with Sphinx, we began with `this example <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ and then used `Sphinx's getting started guide <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.
After setting up the development environment (see `Development setup`_) and adding your documentation, building the docs is done by running ``make docs`` (note, run ``make -B docs`` to force the docs to rebuild and ignore make when it says '... index.html is up to date'). This will build the docs for you. You can preview them by opening ``docs/build/html/index.html`` in a browser.

Please update the documentation to reflect any changes or additions to the code. Follow the structure and style of the existing documentation, and lastly, update the `CHANGELOG` with your changes.

For our docstrings, we use numpy style docstrings. For more information on these, `here is the full guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and `the quick reference we also use <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

.. _Sphinx: http://www.sphinx-doc.org
.. _MESMER issue tracker: https://github.com/MESMER-group/mesmer/issues
.. _`Mathias Hauser`: https://github.com/mathause
.. _`Victoria Bauer`: https://github.com/veni-vidi-vici-dormivi
.. _CHANGELOG: changelog.html
