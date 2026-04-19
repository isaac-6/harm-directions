# Convenience targets for local development.
# Mirrors what CI runs; use `make check` before pushing.

.PHONY: help install check lint format typecheck test clean

help:
	@echo "Available targets:"
	@echo "  install     Install package with dev dependencies"
	@echo "  check       Run all checks (lint, format, typecheck, test)"
	@echo "  lint        Run ruff check"
	@echo "  format      Run ruff format (modifies files)"
	@echo "  typecheck   Run mypy"
	@echo "  test        Run pytest"
	@echo "  clean       Remove caches and build artefacts"

install:
	pip install -e ".[dev]"

check: lint format-check typecheck test
	@echo "All checks passed."

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

format-check:
	ruff format --check src tests scripts

typecheck:
	mypy src

test:
	pytest

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
