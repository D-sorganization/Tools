# Quick Start Guide

Get up and running with the Tools repository in minutes.

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- **Git** with LFS support
- **MATLAB R2020a+** (optional, for MATLAB-based tools)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/D-sorganization/Tools.git
cd Tools
git lfs install
git lfs pull
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10 or higher

# Test launcher
python UnifiedToolsLauncher.py
```

## Using the Launcher

The **UnifiedToolsLauncher.py** is the primary entry point:

```bash
python UnifiedToolsLauncher.py
```

This opens a GUI with all available tools organized by category:

- **Media Processing**: Audio/Video tools
- **Data Processing**: CSV/Parquet analyzers
- **Scientific Modeling**: Solar system simulations, path planners
- **Web Applications**: Calculator, Unit Converter
- **Development Tools**: Folder management utilities

## Common Tasks

### Launch a Python Tool

1. Open UnifiedToolsLauncher.py
2. Navigate to the tool's category
3. Click the tool's launch button

### Launch a MATLAB Tool

1. Ensure MATLAB is installed and in PATH
2. Use UnifiedToolsLauncher.py to launch MATLAB tools
3. If MATLAB is not found, the launcher will attempt to open the file in your default editor

### Run Tests

```bash
# Run all tests
pytest .

# Run specific test file
pytest tests/test_example.py

# Run with coverage
pytest --cov=.
```

### Code Quality Checks

```bash
# Lint code
ruff check .

# Format code
ruff format .
black .

# Type checking
mypy . --config-file mypy.ini
```

## Troubleshooting

### Python Version Issues

If you see `ImportError: cannot import name 'StrEnum'`:

- Upgrade to Python 3.10+ (compatibility shims included for 3.10)
- Or use Python 3.12 for best compatibility

### MATLAB Tools Not Working

- Install MATLAB R2020a or later
- Add MATLAB to system PATH
- Verify with: `matlab -batch "version"`

### Launcher Won't Start

- Check PyQt6 is installed: `pip install PyQt6>=6.7.0`
- Verify Python version: `python --version`
- Check error messages in the terminal

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Review [AGENTS.md](AGENTS.md) for coding standards

## Getting Help

- Check [GitHub Issues](https://github.com/D-sorganization/Tools/issues)
- Review troubleshooting section in README.md
- Create a new issue for bugs or feature requests
