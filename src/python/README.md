# Python Tools Package

A collection of Python-based utilities, applications, and tools for file management, project packaging, and data processing.

## Overview

This directory serves as the central Python package for the Golf Biomechanics Simulator & Game Engine repository. It contains multiple standalone tools and shared utilities organized for easy access and maintenance.

## Directory Structure

```
python/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment specification
├── folder_tool/           # Folder processing and organization
├── folder_tool_pro/       # Advanced folder management (Pro)
├── folder_packer_pro/     # Professional project packaging with encryption
├── project_packer/        # Project packaging utilities
├── src/                   # Shared source modules
│   ├── __init__.py
│   ├── logger_utils.py    # Logging and seed management
└── tests/                 # Unit tests
```

## Included Tools

### Folder Tool

Advanced folder processing for file combining, organization, deduplication, and archive extraction.

**Features:**

- Multiple processing modes (combine, flatten, deduplicate, analyze)
- File filtering by extension and size
- Bulk archive extraction (.zip, .rar, .7z)
- Preview mode for safe testing
- Automatic backup creation

See [folder_tool/README.md](folder_tool/README.md) for details.

### Folder Tool Pro

Enhanced version with professional features and modern UI.

See [folder_tool_pro/README.md](folder_tool_pro/README.md) for details.

### Folder Packer Pro v2.0

Professional project packaging tool with enterprise-grade security.

**Features:**

- AES-256 encryption
- Flexible compression (None, Fast, Balanced, Best)
- Git integration
- Dark/Light themes
- Real-time preview

See [folder_packer_pro/README.md](folder_packer_pro/README.md) for details.

### Project Packer

Secure project packaging and transport utilities.

See [project_packer/README.md](project_packer/README.md) for details.

## Installation

### Using pip

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using Conda

```bash
# Create environment from specification
conda env create -f environment.yml

# Activate environment
conda activate sim-env
```

## Dependencies

Core dependencies (from requirements.txt):

- `numpy==2.0.1` - Numerical computing
- `pandas==2.2.2` - Data manipulation
- `matplotlib==3.9.0` - Visualization
- `scipy==1.13.1` - Scientific computing
- `pytest==8.2.0` - Testing framework
- `pyyaml==6.0.1` - YAML parsing
- `cryptography` - Encryption support
- `PyQt6==6.7.0` - GUI framework

## Shared Utilities

### Logger Utils (`src/logger_utils.py`)

Provides reproducible random seed management:

```python
from src.logger_utils import set_seeds

# Set reproducibility seed (default: 42)
set_seeds()

# Or use custom seed
set_seeds(12345)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## Development

### Code Quality

Run pre-commit checks before committing:

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Adding New Tools

1. Create a new directory for your tool
2. Add a `README.md` with usage documentation
3. Include tests in the `tests/` directory
4. Update this README with tool information

## Integration

These tools integrate with:

- **UnifiedToolsLauncher.py** - Central launcher for all repository tools
- **Development Tools** (`development_tools/`) - Additional tool versions
- **File Management** (`file_management/`) - Related file utilities

## License

Part of the Tools repository. See main repository license for details.
