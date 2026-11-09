.DEFAULT_GOAL := help
UV ?= uv
# Host interpreter downloads require an explicit, separately explained user approval.
export UV_PYTHON_DOWNLOADS := never
APP = $(UV) run --locked python -m llm_inference
NAME ?= copilot
SERVICE ?=

.PHONY: help setup doctor up down restart model-start model-stop status logs check key revoke test lint typecheck

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  make %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install dependencies and create .env with generated credentials; starts nothing
	$(UV) sync --locked
	$(APP) setup

doctor: ## Validate local prerequisites and configuration; starts nothing
	$(APP) doctor

up: ## Start LLM containers connected to your existing Ollama server
	$(APP) up

down: ## Stop LLM containers and unload the selected model; leave Ollama running
	$(APP) down

restart: ## Stop and start with the current .env
	$(APP) restart

model-start: ## Preload the selected model through the existing Ollama API
	$(APP) model-start

model-stop: ## Unload only the selected model; leave Ollama and downloaded files intact
	$(APP) model-stop

status: ## Show service state and loaded models without loading a model
	$(APP) status

logs: ## Follow container logs; optionally use SERVICE=gateway, db, or edge
	$(APP) logs $(SERVICE)

check: ## Test auth, route isolation, inference, and streaming; loads the selected model
	$(APP) check

key: ## Create/reuse and display a client key; NAME=copilot by default
	$(APP) key $(NAME)

revoke: ## Revoke a client key; use NAME=client-name
	$(APP) revoke $(NAME)

test: ## Run lifecycle, configuration, and protocol tests without a model or Docker
	$(UV) run --locked pytest -q

lint: ## Check Python style
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .

typecheck: ## Run strict mypy checks on application source
	$(UV) run --locked mypy
