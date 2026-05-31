COMPOSE ?= docker compose
SERVICE ?= pipeline
GPU_SERVICE ?= pipeline-gpu
DOCKER_UID ?= 1000
DOCKER_GID ?= 1000
COMPOSE_ENV := DOCKER_BUILDKIT=1 DOCKER_UID=$(DOCKER_UID) DOCKER_GID=$(DOCKER_GID)
RUN := $(COMPOSE_ENV) $(COMPOSE) run --rm $(SERVICE)

.DEFAULT_GOAL := help

.PHONY: help build rebuild shell lint format type typecheck test test-unit test-integration test-all coverage ci pre-commit build-data train evaluate optimize tensorboard compose-config gpu-build gpu-train

help:
	@echo "Docker-first workflow"
	@echo "  make build             Build the pipeline image"
	@echo "  make ci                Run compile, lint, typecheck, and unit tests"
	@echo "  make lint|format|type  Run quality tools in the container"
	@echo "  make test-unit         Run unit tests with coverage"
	@echo "  make build-data        Build or refresh datasets"
	@echo "  make train             Train the configured agent"
	@echo "  make evaluate          Evaluate the saved agent"
	@echo "  make optimize          Run Optuna tuning"
	@echo "  make tensorboard       Serve TensorBoard"
	@echo "  make gpu-build|gpu-train  Build or run the optional GPU image"

build:
	$(COMPOSE_ENV) $(COMPOSE) build $(SERVICE)

rebuild:
	$(COMPOSE_ENV) $(COMPOSE) build --pull --no-cache $(SERVICE)

shell:
	$(RUN) shell

lint:
	$(RUN) lint

format:
	$(RUN) format

type: typecheck

typecheck:
	$(RUN) typecheck

test: test-unit

test-unit:
	$(RUN) test-unit

test-integration:
	$(RUN) test-integration

test-all:
	$(RUN) test tests -ra --strict-config --strict-markers --tb=short -o addopts=

coverage: test

ci:
	$(RUN) ci

pre-commit:
	$(RUN) pre-commit

build-data:
	$(RUN) build-data

train:
	$(RUN) train

evaluate:
	$(RUN) evaluate

optimize:
	$(RUN) optimize

tensorboard:
	$(COMPOSE_ENV) $(COMPOSE) up tensorboard

compose-config:
	$(COMPOSE) config
	$(COMPOSE) --profile gpu config

gpu-build:
	$(COMPOSE_ENV) $(COMPOSE) --profile gpu build $(GPU_SERVICE)

gpu-train:
	$(COMPOSE_ENV) $(COMPOSE) --profile gpu run --rm $(GPU_SERVICE) train
