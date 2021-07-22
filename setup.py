from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

import versioneer

PACKAGE_NAME = "mesmer-emulator"
DESCRIPTION = "Modular Earth System Model Emulator with spatially Resolved output"
KEYWORDS = [
    "climate",
    "atmosphere",
    "Earth System Model Emulator",
]

AUTHOR = "mesmer developpers"
EMAIL = "mesmer@env.ethz.ch"
URL = "https://github.com/MESMER-group/mesmer"  # use documentation url?
PROJECT_URLS = {
    "Source": "https://github.com/MESMER-group/mesmer",
    "Bug Reports": "https://github.com/MESMER-group/mesmer/issues",
    # "Documentation": "TBD",
}

LICENSE = "GPLv3+"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]


REQUIREMENTS_INSTALL = [
    "dask[complete]",
    "geopy",
    "numpy",
    "pandas",
    "scikit-learn",
    "statsmodels",
    "regionmask",
    "xarray",
]
REQUIREMENTS_TESTS = [
    "pytest",
    "pytest-cov",
    "pytest-xdist",
]
REQUIREMENTS_DEV = [
    "black",
    "flake8",
    "isort",
    "setuptools",
    "twine",
    "wheel",
    *REQUIREMENTS_TESTS,
]

REQUIREMENTS_DOCS = [
    "sphinx-book-theme",
    "numpydoc",
]

REQUIREMENTS_EXTRAS = {
    "dev": REQUIREMENTS_DEV,
    "tests": REQUIREMENTS_TESTS,
    "docs": REQUIREMENTS_DOCS,
}


SOURCE_DIR = "mesmer"

PACKAGES = find_packages()
PACKAGE_DATA = {}


README = "README.md"

with open(README, "r") as readme_file:
    README_TEXT = readme_file.read()


class Mesmer(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": Mesmer})

setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description=README_TEXT,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    include_package_data=True,
    install_requires=REQUIREMENTS_INSTALL,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=cmdclass,
    # entry_points=ENTRY_POINTS,
)
