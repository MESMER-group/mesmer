.DEFAULT_GOAL := help

OS=`uname`
SHELL=/bin/bash

CONDA_ENV_YML=environment.yml

FILES_TO_FORMAT_PYTHON=setup.py examples mesmer tests

N_JOBS ?= 1

ifndef CONDA_PREFIX
$(error Conda environment not active. Activate your conda environment before using this Makefile.)
else
ifeq ($(CONDA_DEFAULT_ENV),base)
$(error Do not install to conda base environment. Activate a different conda environment and rerun make. A new environment can be created with e.g. `conda create --name mesmer`))
endif
VENV_DIR=$(CONDA_PREFIX)
endif

# use mamba if available
MAMBA_EXE := $(shell command -v mamba 2> /dev/null)
ifndef MAMBA_EXE
MAMBA_OR_CONDA=$(CONDA_EXE)
else
MAMBA_OR_CONDA=$(MAMBA_EXE)
endif

PYTHON=$(VENV_DIR)/bin/python

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: format
format: $(VENV_DIR)  ## auto-format the code using relevant tools
	make isort
	make black
	make flake8

.PHONY: black
black: $(VENV_DIR)  ## auto-format the code using black
	$(VENV_DIR)/bin/black $(FILES_TO_FORMAT_PYTHON) docs/source/conf.py

.PHONY: flake8
flake8: $(VENV_DIR)  ## lint the code using flake8
	$(VENV_DIR)/bin/flake8 $(FILES_TO_FORMAT_PYTHON)

.PHONY: isort
isort: $(VENV_DIR)  ## lint the code using flake8
	$(VENV_DIR)/bin/isort $(FILES_TO_FORMAT_PYTHON)

.PHONY: docs
docs: $(VENV_DIR)  ## build the docs
	$(VENV_DIR)/bin/sphinx-build -M html docs/source docs/build

.PHONY: test
test: $(VENV_DIR)  ## run the testsuite
	$(VENV_DIR)/bin/pytest -r a -v --cov=mesmer --cov-report=term-missing

.PHONY: test_cov_xml
test_cov_xml: $(VENV_DIR)  ## run the testsuite with xml report for codecov
	$(VENV_DIR)/bin/pytest -r a -v --cov=mesmer --cov-report=xml

.PHONY: test-install
test-install: $(VENV_DIR)  ## test whether installing locally in a fresh env works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install wheel pip --upgrade
	$(TEMPVENV)/bin/pip install .
	$(TEMPVENV)/bin/python scripts/test_install.py

.PHONY: conda-environment
conda-environment:  $(VENV_DIR)  ## make virtual environment for development
$(VENV_DIR): $(CONDA_ENV_YML) setup.py
	$(MAMBA_OR_CONDA) env update --file $(CONDA_ENV_YML)
	# Install the remainder of the dependencies using pip
	$(PYTHON) -m pip install --upgrade pip wheel
	$(PYTHON) -m pip install -e .[dev]
	touch $(VENV_DIR)
