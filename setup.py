from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

PACKAGE_NAME = "mesmer"
DESCRIPTION = "TBD Lea"
KEYWORDS = [
    "TBD",
    "Lea",
    "python",
    "climate",
    # "atmosphere",
    # "simple climate model",
    # "reduced complexity climate model",
    # "data processing",
]

AUTHOR = "Lea Beusch"
EMAIL = "TBD Lea"
URL = "TBD Lea"
PROJECT_URLS = {
    "Bug Reports": "TBD Lea https://github.com/znicholls/netcdf-scm/issues",
    # "Documentation": "TBD Lea https://openscm.readthedocs.io/en/latest",
    "Source": "TBD Lea https://github.com/znicholls/netcdf-scm",
}

LICENSE = "TBD Lea"
CLASSIFIERS = [
    "TBD Lea",
    "Development Status :: 4 - Beta",
    # "License :: OSI Approved :: BSD License",
    # "Intended Audience :: Developers",
    # "Operating System :: OS Independent",
    # "Programming Language :: Python :: 3.7",
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
]
requirements_dev = [
    *[
        "black",
        "flake8",
        "isort",
    ],
    *REQUIREMENTS_TESTS,
]

REQUIREMENTS_EXTRAS = {
    "dev": requirements_dev,
    "tests": REQUIREMENTS_TESTS,
}


SOURCE_DIR = "mesmer"

PACKAGES = find_packages(exclude=["tests"])
PACKAGE_DATA = {}


README = "README.md"

with open(README, "r") as readme_file:
    README_TEXT = readme_file.read()


# class Mesmer(TestCommand):
#     def finalize_options(self):
#         TestCommand.finalize_options(self)
#         self.test_args = []
#         self.test_suite = True

#     def run_tests(self):
#         import pytest

#         pytest.main(self.test_args)


# cmdclass = versioneer.get_cmdclass()
# cmdclass.update({"test": Mesmer})

setup(
    name=PACKAGE_NAME,
    # version=versioneer.get_version(),
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
    # cmdclass=cmdclass,
    # entry_points=ENTRY_POINTS,
)
