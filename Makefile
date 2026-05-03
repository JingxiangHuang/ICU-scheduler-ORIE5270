# Makefile for icu_scheduler — one-liner reproducibility

.PHONY: help install test lint format docs demo clean

help:
	@echo "ICU Scheduler — available targets:"
	@echo "  make install   Install package with dev dependencies"
	@echo "  make test      Run pytest with coverage"
	@echo "  make lint      Run flake8 + mypy"
	@echo "  make format    Run black"
	@echo "  make docs      Build Sphinx HTML docs"
	@echo "  make demo      Run the end-to-end demo pipeline"
	@echo "  make clean     Remove build/coverage artifacts"

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/

docs:
	cd docs && make html

demo:
	icu-scheduler run-demo

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} +
