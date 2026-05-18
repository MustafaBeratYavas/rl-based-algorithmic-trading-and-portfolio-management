PYTHON ?= python
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
PYTEST := $(PYTHON) -m pytest

.PHONY: setup lint format type test test-unit test-integration test-all coverage ci pre-commit build-data train evaluate optimize

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check src scripts tests
	$(RUFF) format --check src scripts tests

format:
	$(RUFF) check src scripts tests --fix
	$(RUFF) format src scripts tests

type:
	$(MYPY) src scripts

test: test-unit

test-unit:
	$(PYTEST) tests/unit --cov=src --cov-report=term-missing --cov-report=xml

test-integration:
	RUN_INTEGRATION=1 $(PYTEST) tests/integration -ra --strict-config --strict-markers --tb=short -o addopts=

test-all:
	RUN_INTEGRATION=1 $(PYTEST) tests -ra --strict-config --strict-markers --tb=short -o addopts=

coverage: test

ci: lint type test

pre-commit:
	$(PYTHON) -m pre_commit run --all-files

build-data:
	$(PYTHON) -m scripts.build_dataset

train:
	$(PYTHON) -m scripts.train_agent

evaluate:
	$(PYTHON) -m scripts.evaluate_agent

optimize:
	$(PYTHON) -m scripts.optimize_hyperparams
