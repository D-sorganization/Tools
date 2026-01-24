# Tools Repository Makefile
# Provides common development tasks for the Tools monorepo
#
# Usage:
#   make help     - Show available targets
#   make lint     - Run all linters
#   make format   - Format code
#   make test     - Run tests
#   make clean    - Clean build artifacts

.PHONY: help lint format test clean install check all

# Default target
help:
	@echo "Tools Repository - Available targets:"
	@echo ""
	@echo "  make install   - Install dependencies"
	@echo "  make lint      - Run linters (ruff, mypy)"
	@echo "  make format    - Format code (black, ruff)"
	@echo "  make test      - Run pytest"
	@echo "  make check     - Run all checks (lint + test)"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make all       - Install, format, lint, test"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -r python/requirements.txt || true

# Run linters
lint:
	@echo "Running ruff check..."
	ruff check .
	@echo "Running mypy (errors are advisory; see CONTRIBUTING.md)..."
	mypy . --config-file mypy.ini || true

# Format code
format:
	@echo "Running black..."
	black .
	@echo "Running ruff format..."
	ruff format .
	@echo "Running ruff fix..."
	ruff check . --fix || true

# Run tests
test:
	@echo "Running pytest..."
	pytest python/tests/ -v --tb=short

# Run all checks
check: lint test
	@echo "All checks complete."

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name "*.egg-info" \) -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*_output.txt" -o -name "*_temp.txt" \) -delete 2>/dev/null || true
	@echo "Clean complete."

# Run everything
all: install format lint test
	@echo "All tasks complete."
