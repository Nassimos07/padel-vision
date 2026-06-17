.PHONY: help venv install install-cpu detect app lint format test clean
PY ?= python3
SOURCE ?= data/raw/match.mp4

help:
	@echo "Padel Match Analytics - make targets:"
	@echo "  make venv         Create a .venv virtual environment"
	@echo "  make install      Install GPU torch (CUDA 12.1) + project extras"
	@echo "  make install-cpu  Install CPU-only torch + project extras"
	@echo "  make detect       Run detection (override with SOURCE=path/to/video.mp4)"
	@echo "  make app          Launch the Streamlit dashboard"
	@echo "  make lint         Lint with ruff"
	@echo "  make format       Format with black + ruff"
	@echo "  make test         Run the test suite"
	@echo "  make clean        Remove caches and build artifacts"

venv:
	$(PY) -m venv .venv
	@echo "Activate it with:  source .venv/bin/activate"

install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip install -e ".[app,notebook,dev]"

install-cpu:
	pip install torch torchvision
	pip install -e ".[app,notebook,dev]"

detect:
	padel detect $(SOURCE)

app:
	streamlit run app/streamlit_app.py

lint:
	ruff check src tests app

format:
	black src tests app
	ruff check --fix src tests app

test:
	pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
