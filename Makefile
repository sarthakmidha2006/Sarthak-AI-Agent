# =============================================================================
# AI Persona System ("brain") — developer task runner
# Run `make help` for the list of targets.
# =============================================================================

# Use bash with strict-ish flags for recipe shells.
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# Allow overriding the interpreter, e.g. `make run PYTHON=python3.11`.
PYTHON ?= python
PIP ?= $(PYTHON) -m pip

# Image / container naming (kept in sync with docker-compose.yml).
IMAGE ?= scaler-ai-persona:latest

.DEFAULT_GOAL := help

.PHONY: help install ingest run eval test fmt lint docker-build docker-up docker-down clean

help: ## Show this help.
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies (+ dev extras for fmt/lint/test).
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install ruff black

ingest: ## Ingest the corpus (resume + GitHub) into the vector / BM25 indexes.
	./scripts/ingest.sh $(ARGS)

run: ## Run the FastAPI app locally with autoreload (host/port from .env).
	./scripts/run.sh

eval: ## Run the evaluation harness and write eval/report.json.
	$(PYTHON) -m eval.run_eval $(ARGS)

test: ## Run the test suite (no network; OpenAI is mocked).
	$(PYTHON) -m pytest

fmt: ## Auto-format with black + ruff (apply fixes).
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .

lint: ## Lint without modifying files.
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

docker-build: ## Build the Docker image.
	docker build -t $(IMAGE) .

docker-up: ## Start the stack via docker compose (foreground, with build).
	docker compose up --build

docker-down: ## Stop and remove the docker compose stack.
	docker compose down

clean: ## Remove caches and build artifacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
