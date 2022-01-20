# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Import packages ---------------------------------------------------------

import datetime

from mesmer._version import get_versions

# -- Project information -----------------------------------------------------

project = "mesmer"
copyright_year = datetime.date.today().year
copyright = "(c) 2021-{} ETH Zurich (Land-climate dynamics group, Prof. S.I. Seneviratne), MESMER contributors listed in AUTHORS".format(
    copyright_year
)
authors = "Authors, see AUTHORS"
author = authors

# The short X.Y version
version = get_versions()["version"].split("+")[0]
# The full version, including alpha/beta/rc tags
release = get_versions()["version"]


# -- General configuration ---------------------------------------------------

# add sphinx extension modules
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
]

extlinks = {
    "issue": ("https://github.com/mesmer-group/mesmer/issues/%s", "GH"),
    "pull": ("https://github.com/mesmer-group/mesmer/pull/%s", "PR"),
}

autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

# napoleon_use_ivar = True
# napoleon_use_admonition_for_notes = True

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []

pygments_style = "sphinx"
