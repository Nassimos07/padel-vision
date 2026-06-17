.PHONY: help venv install install-cpu roi court detect lint format test clean
PY ?= python3
SOURCE ?= data/raw/match.mp4

help:
	@echo "padel-vision - make targets:"
	@echo "  make venv         Create a .venv virtual environment"
	@echo "  make install      Install GPU torch (CUDA 12.1) + the package + extras"
	@echo "  make install-cpu  Install CPU-only torch + the package + extras"
	@echo "  make roi          Draw the detection ROI (override with SOURCE=path/to/video.mp4)"
	@echo "  make court        Pick the 4 court corners (override with SOURCE=...)"
	@echo "  make detect       Detect players in real time (override with SOURCE=...)"
	@echo "  make lint         Lint with ruff"
	@echo "  make format       Format with black + ruff"
	@echo "  make test         Run the test suite"
	@echo "  make clean        Remove caches and build artifacts"

venv:
	$(PY) -m venv .venv
	@echo "Activate it with:  source .venv/bin/activate"

install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip install -e ".[notebook,dev]"

install-cpu:
	pip install torch torchvision
	pip install -e ".[notebook,dev]"

roi:
	padel-vision roi adjust $(SOURCE)

court:
	padel-vision court adjust $(SOURCE)

detect:
	padel-vision detect players $(SOURCE)

lint:
	ruff check src tests scripts

format:
	black src tests scripts
	ruff check --fix src tests scripts

test:
	pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
