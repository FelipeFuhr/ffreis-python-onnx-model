.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

SHELL := /usr/bin/env bash

PYTHON ?= python3
VENV ?= .venv
CONTAINER_COMMAND ?= podman

PREFIX ?= ffreis
BASE_DIR ?= .
CONTAINER_DIR ?= container

# ------------------------------------------------------------------------------
# Image names
# ------------------------------------------------------------------------------

BASE_IMAGE := $(PREFIX)/base
BASE_BUILDER_IMAGE := $(PREFIX)/base-builder
BUILDER_IMAGE := $(PREFIX)/builder
BASE_RUNNER_IMAGE := $(PREFIX)/base-runner
RUNNER_IMAGE := $(PREFIX)/runner

# ------------------------------------------------------------------------------
# Derived values
# ------------------------------------------------------------------------------

# Extract digests from digests.env (computed once)
BASE_IMAGE_VALUE := $(shell grep '^BASE_IMAGE=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)
BASE_DIGEST_VALUE := $(shell grep '^BASE_DIGEST=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Meta targets
# ------------------------------------------------------------------------------

.PHONY: all
all: lint build run ## Lint, build, and run

# ------------------------------------------------------------------------------
# Tooling / setup
# ------------------------------------------------------------------------------

.PHONY: install-python-local
install-python-local: ## Install Python locally if missing
	@if command -v python3 >/dev/null 2>&1; then \
		echo "python3 already installed: $$(command -v python3)"; \
		exit 0; \
	fi
	sudo apt-get update
	sudo apt-get install -y python3 python3-pip

.PHONY: install-podman-local
install-podman-local: ## Install Podman locally if missing
	@if command -v podman >/dev/null 2>&1; then \
		echo "podman already installed: $$(command -v podman)"; \
		exit 0; \
	fi
	sudo apt-get update
	sudo apt-get install -y podman

.PHONY: local-setup
local-setup: install-python-local install-podman-local ## Install local dev prerequisites

# ------------------------------------------------------------------------------
# Container builds
# ------------------------------------------------------------------------------

.PHONY: build-base
build-base: ## Build base image (pinned by digest env)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base -t $(BASE_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE_VALUE)" \
		--build-arg BASE_DIGEST="$(BASE_DIGEST_VALUE)"

.PHONY: build-base-builder
build-base-builder: build-base ## Build base-builder image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-builder -t $(BASE_BUILDER_IMAGE) $(BASE_DIR)

.PHONY: build-builder
build-builder: build-base build-base-builder ## Build builder image (installs deps, runs tests, generates lock file)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.builder -t $(BUILDER_IMAGE) $(BASE_DIR)

.PHONY: build-base-runner
build-base-runner: build-base ## Build base-runner image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(BASE_RUNNER_IMAGE) $(BASE_DIR)

.PHONY: build-runner
build-runner: build-base-runner build-builder ## Build runner image (minimal Python runtime)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.runner -t $(RUNNER_IMAGE) $(BASE_DIR) \
		--build-arg BUILDER_IMAGE="$(BUILDER_IMAGE)"

.PHONY: build-images
build-images: build-base build-base-builder build-builder build-base-runner build-runner ## Build all images (may be slow)

.PHONY: build
build: build-images ## Build all container images

# ------------------------------------------------------------------------------
# Python (local) targets
# ------------------------------------------------------------------------------

.PHONY: env
env: ## Create local virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: . $(VENV)/bin/activate"

.PHONY: build-local
build-local: env ## Install project and dev dependencies
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

.PHONY: run-app
run-app: ## Run the runner container
	$(CONTAINER_COMMAND) run $(RUNNER_IMAGE)

.PHONY: run
run: ## Run app locally
	$(VENV)/bin/python main.py

.PHONY: run-container
run-container: run-app ## Alias: run the app in container

.PHONY: fmt
fmt: ## Format Python code
	$(VENV)/bin/black .
	$(VENV)/bin/ruff format .

.PHONY: fmt-check
fmt-check: ## Check Python formatting
	$(VENV)/bin/black --check .
	$(VENV)/bin/ruff format --check .

.PHONY: lint
lint: fmt-check ## Run linting + static typing
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy src

.PHONY: test
test: ## Run all tests
	$(VENV)/bin/pytest -q

.PHONY: test-unit
test-unit: ## Run unit tests
	$(VENV)/bin/pytest -q -m "unit"

.PHONY: test-integration
test-integration: ## Run integration tests
	$(VENV)/bin/pytest -q -m "integration and not e2e"

.PHONY: test-e2e
test-e2e: ## Run e2e tests
	$(VENV)/bin/pytest -q -m "e2e"

.PHONY: coverage
coverage: ## Run tests with coverage output
	$(VENV)/bin/pytest \
		-q \
		--cov=onnx_model_serving \
		--cov-report=term \
		--cov-report=xml:coverage.xml

# ------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------

.PHONY: clean-repo
clean-repo: ## Clean repo build outputs
	rm -rf $(VENV) build __pycache__ .pytest_cache .coverage htmlcov *.pyc coverage.xml

.PHONY: clean-base
clean-base: ## Remove base image
	$(CONTAINER_COMMAND) rmi $(BASE_IMAGE) || true

.PHONY: clean-base-builder
clean-base-builder: ## Remove base-builder image
	$(CONTAINER_COMMAND) rmi $(BASE_BUILDER_IMAGE) || true

.PHONY: clean-builder
clean-builder: ## Remove builder image
	$(CONTAINER_COMMAND) rmi $(BUILDER_IMAGE) || true

.PHONY: clean-base-runner
clean-base-runner: ## Remove base-runner image
	$(CONTAINER_COMMAND) rmi $(BASE_RUNNER_IMAGE) || true

.PHONY: clean-runner
clean-runner: ## Remove runner image
	$(CONTAINER_COMMAND) rmi $(RUNNER_IMAGE) || true

.PHONY: clean-all
clean-all: clean-repo clean-base clean-base-builder clean-builder clean-base-runner clean-runner ## Clean everything
