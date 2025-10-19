# Makefile for Virtual Try-On Project
# Note: On Windows, use 'make.exe' from GNU Make or use make.ps1 (PowerShell alternative)

.PHONY: help setup install run test lint format clean dev

# Default Python executable
PYTHON ?= python
VENV_DIR ?= .venv
VENV_BIN ?= $(VENV_DIR)/Scripts
PYTHON_VENV = $(VENV_BIN)/python.exe
PIP_VENV = $(VENV_BIN)/pip.exe
STREAMLIT_VENV = $(VENV_BIN)/streamlit.exe

# Default target
help:
	@echo "Virtual Try-On - Available targets:"
	@echo ""
	@echo "  make setup     - Create virtual environment and install dependencies"
	@echo "  make install   - Install/update dependencies"
	@echo "  make run       - Run the Streamlit application"
	@echo "  make test      - Run pytest tests"
	@echo "  make lint      - Run code linting (flake8)"
	@echo "  make format    - Format code with black"
	@echo "  make clean     - Remove generated files and caches"
	@echo "  make dev       - Install development dependencies"
	@echo ""

# Create virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt
	@echo "Setup complete! Activate with: $(VENV_BIN)\Activate.ps1"

# Install/update dependencies
install:
	@echo "Installing dependencies..."
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt

# Install development dependencies
dev: install
	@echo "Installing development dependencies..."
	$(PIP_VENV) install pytest pytest-cov black flake8 mypy isort

# Run the Streamlit application
run:
	@echo "Starting Streamlit application..."
	$(STREAMLIT_VENV) run app.py

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON_VENV) -m pytest -v

# Run linting
lint:
	@echo "Running flake8..."
	-$(PYTHON_VENV) -m flake8 src/ app.py --max-line-length=100 --extend-ignore=E203,W503

# Format code
format:
	@echo "Formatting code with black..."
	-$(PYTHON_VENV) -m black src/ app.py --line-length=100
	@echo "Sorting imports with isort..."
	-$(PYTHON_VENV) -m isort src/ app.py --profile black

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	-rm -rf __pycache__/
	-rm -rf src/__pycache__/
	-rm -rf tests/__pycache__/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf htmlcov/
	-rm -rf .coverage
	-rm -rf dist/
	-rm -rf build/
	-rm -rf *.egg-info/
	@echo "Clean complete!"

# Run all checks (lint + test)
check: lint test
	@echo "All checks passed!"
