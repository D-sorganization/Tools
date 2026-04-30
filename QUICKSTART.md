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

# Install dependencies and developer tooling
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10 or higher

# Confirm the root developer entrypoints are available
make help

# Test launcher
python UnifiedToolsLauncher.py
```

## Using the Launcher

The **UnifiedToolsLauncher.py** is the primary entry point:

```bash
python UnifiedToolsLauncher.py
```

This opens a GUI with all available tools organized by category:

- **Analysis**: Multi-parameter sensitivity analysis
- **Biomechanics**: C3D viewer, motion capture analysis
- **Data Processing**: CSV/Parquet analyzers
- **Development Tools**: Folder management, PDF renaming
- **Engineering Drafting**: P&ID generator
- **Mathematics**: ODE solver
- **Optimization**: Adam optimizer
- **Process Simulation**: Financial calculator, pressure drop analysis
- **Scientific Modeling**: Solar system simulations, path planners
- **Web Applications**: Calculator, Unit Converter

### Launcher Features

#### Error Notifications
When a tool fails to launch, a detailed error notification appears showing:
- The error message
- Error type (ToolNotFoundError, LaunchError, etc.)
- Suggestions for remediation based on the error type
- Example: "Python not found. Install Python 3.11+"

#### Launch Progress Indicator
While a tool is starting:
- A progress dialog shows with animated spinner
- Status message displays: "Starting [Tool Name]..."
- Progress bar indicates elapsed time
- Auto-closes with success message: "✓ Tool launched successfully"
- Timeout after 5 minutes if tool hangs

#### Debug Mode
Toggle "Debug Mode (Verbose Logs)" in the header to:
- See detailed stdout/stderr output
- Monitor tool process IDs
- Troubleshoot launch issues

#### Activity Log
The bottom panel shows:
- Tool launch history
- Success/error messages with timestamps
- Color-coded: green for success, red for errors
- Scrollable history for reference

### Keyboard Shortcuts

Press **Ctrl+?** to view all available shortcuts:

| Shortcut | Action |
|----------|--------|
| Ctrl+F | Search tools (case-insensitive) |
| Esc | Clear search / Close dialog |
| Tab | Navigate to next tool |
| Shift+Tab | Navigate to previous tool |
| Arrow Up/Down | Navigate between tools |
| Arrow Left/Right | Navigate between categories |
| Enter | Launch selected tool |
| F1 | Open User Manual |
| Ctrl+? | Show keyboard shortcuts help |

### Help System

Access help through the **Help** menu:
- **User Manual (F1)**: Complete launcher documentation
- **Tool Help**: Context-sensitive help for selected category
- **Getting Started**: Quick setup guide
- **About**: Launcher version and information
- **Keyboard Shortcuts (Ctrl+?)**: All available shortcuts

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
python -m pytest

# Run specific test file
python -m pytest tests/test_validation.py

# Run with coverage
python -m pytest --cov=.
```

### Code Quality Checks

```bash
# Lint code
python -m ruff check .

# Format code
python -m ruff format .
python -m black .

# Type checking
python -m mypy . --config-file mypy.ini
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
