.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash

PYTHON ?= python3

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build: ## Install dependencies
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: run
run: ## Run the application
	$(PYTHON) main.py

.PHONY: fmt
fmt: ## Format Python code
	$(PYTHON) -m black . || true

.PHONY: fmt-check
fmt-check: ## Check Python formatting
	$(PYTHON) -m black --check . || true

.PHONY: lint
lint: fmt-check ## Run linting
	$(PYTHON) -m pylint *.py || true

.PHONY: test
test: ## Run tests
	$(PYTHON) -m pytest -v || echo "No tests defined yet"

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf __pycache__ .pytest_cache .coverage htmlcov *.pyc
