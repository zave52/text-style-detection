.DEFAULT_GOAL := help
.PHONY: help install install-min kernel api ui notebook docs dataset test lint format check

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
UI_PORT  ?= 8501

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv (Python 3.13) and install all dependencies (CPU)
	uv sync

install-min: ## Install only the runtime dependencies (API and UI)
	uv sync --no-default-groups --group api --group ui

kernel: ## Register the Jupyter kernel "text-style-detection"
	uv run python -m ipykernel install --user --name text-style-detection --display-name "Python 3.13 (text-style-detection)"

api: ## Run the FastAPI backend with auto-reload on http://127.0.0.1:8000
	uv run uvicorn app.main:app --reload --host $(API_HOST) --port $(API_PORT)

ui: ## Run the Streamlit interface on http://localhost:8501 (talks to the API)
	API_URL=http://localhost:$(API_PORT) uv run streamlit run frontend/app.py --server.port $(UI_PORT)

notebook: ## Start JupyterLab
	uv run jupyter lab

docs: ## Serve the MkDocs documentation locally
	uv run --group docs mkdocs serve

dataset: ## Compile, audit, and clean the corpus into data/prepared/
	uv run python -m scripts.build_dataset

test: ## Run the test suite
	uv run pytest

lint: ## Check code style with Ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Format code with Ruff
	uv run ruff check --fix .
	uv run ruff format .

check: lint test ## Run lint and tests
