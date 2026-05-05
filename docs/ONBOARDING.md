# Developer Onboarding Guide

Welcome to the Tools repository! This guide will get you up and running in under 30 minutes.

## Prerequisites

Check you have these installed:

```bash
# Python 3.10+ (3.12 recommended for best compatibility)
python --version  # Should output 3.10 or higher

# Git with LFS support
git --version
git lfs version

# (Optional) MATLAB R2020a+ for MATLAB-based tools
# (Optional) Node.js 16+ for web applications
```

### Installing Prerequisites

**Python 3.10+:**
- **Ubuntu/Debian**: `sudo apt install python3.12 python3.12-venv`
- **macOS**: `brew install python@3.12`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

**Git LFS:**
```bash
# Ubuntu/Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# Then enable LFS in your user config
git lfs install
```

## Local Setup (5 minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/D-sorganization/Tools.git
cd Tools
git lfs install
git lfs pull
```

**Note:** `git lfs pull` downloads large files. If you see warnings, that's normal—LFS files may not exist yet.

### 2. Create a Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

You'll see `(venv)` in your prompt when active.

### 3. Install Dependencies

```bash
# Install core dependencies
python -m pip install -r requirements.txt

# Install in editable mode with dev tools
python -m pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
# Check Python is correct
python --version

# Check key packages are available
python -c "import PyQt6; print('PyQt6 OK')"
python -c "import pytest; print('pytest OK')"
python -c "import ruff; print('ruff OK')"

# Try a quick test
python -m pytest tests/ -k "test_config" -v
```

**Success**: You should see at least one test pass.

## First Run (5 minutes)

### Run the Launcher

```bash
python UnifiedToolsLauncher.py
```

You should see a PyQt6 window with multiple tabs (Data Processing, Scientific Modeling, etc.) and a list of tools.

**If it doesn't open:**

- Check PyQt6: `pip install --upgrade PyQt6>=6.6.0`
- Check Python version: must be 3.10+
- Look for error messages in the terminal
- Try with verbose output: `python UnifiedToolsLauncher.py --verbose 2>&1 | head -50`

### Launch a Tool

1. Click a tool in the launcher (e.g., "Unit Converter" under Web Applications)
2. Click the **Launch** button
3. For Python tools, a new window should open
4. For web tools, your browser opens

**Common Issues:**
- **MATLAB tools fail silently**: MATLAB not installed or not in PATH. See "Troubleshooting" below.
- **Tool doesn't appear**: Update tools.json or use auto-discovery manifest. See "Adding Tools" section.

## Running Tests (3 minutes)

Tests are organized by marker:

```bash
# Run all tests
python -m pytest

# Run unit tests only
python -m pytest -m unit

# Run integration tests
python -m pytest -m integration

# Run with coverage report
python -m pytest --cov=. --cov-report=html
# Open htmlcov/index.html in your browser

# Run a specific test file
python -m pytest tests/test_plugin_manager.py -v

# Run tests matching a pattern
python -m pytest -k "config" -v
```

## Code Quality (3 minutes)

This repo uses **Ruff** for formatting and linting (not Black).

```bash
# Check for style violations
python -m ruff check .

# Auto-fix fixable violations
python -m ruff check --fix .

# Format code (88-char line limit)
python -m ruff format .

# Check formatting without modifying
python -m ruff format --check .

# Type checking
python -m mypy . --config-file mypy.ini
```

**Pre-commit hooks:**
```bash
# Install hooks to run checks before each commit
bash scripts/setup_precommit.sh

# This ensures you never commit code that fails CI
```

## Understanding the Structure

```
Tools/
├── docs/                          # This documentation
│   ├── ONBOARDING.md             # You are here
│   ├── BUILD_A_TOOL.md           # Tutorial for creating tools
│   ├── ARCHITECTURE_OVERVIEW.md  # System architecture
│   └── ...
├── src/
│   ├── python/src/core/          # Plugin system
│   ├── python/src/utils/         # Shared utilities
│   ├── tools/                    # Tool implementations
│   ├── signal_processing/        # Signal processing library
│   ├── urdf/                     # URDF utilities
│   ├── calculators/              # Engineering calculators
│   └── ...
├── tests/                         # Test suite (organized by module)
├── UnifiedToolsLauncher.py       # Main entry point (PyQt6 GUI)
├── tools.json                     # Tool registry (manifest-based auto-discovery)
├── requirements.txt               # Core dependencies
├── pyproject.toml                 # Python package metadata
├── CLAUDE.md                      # Codebase governance rules
├── README.md                      # Repository overview
└── QUICKSTART.md                  # Quick reference
```

## Key Files to Know

| File | Purpose |
|------|---------|
| `CLAUDE.md` | **Governance**: CI requirements, coding standards, shared library constraints |
| `README.md` | Repository overview and basic troubleshooting |
| `QUICKSTART.md` | Quick reference for common tasks |
| `requirements.txt` | Core Python dependencies |
| `pyproject.toml` | Package metadata (name, version, entry points) |
| `tools.json` | Registry of available tools (centralized, explicit ordering) |
| `src/<tool>/tool_manifest.json` | Per-tool auto-discovery manifest |
| `src/<tool>/gui_registration.py` | PyQt6 launcher metadata for a tool |
| `docs/architecture/PLUGIN_SYSTEM.md` | Full plugin system reference |

## Plugin System and Tool Registration

Tools are registered with the launcher in two ways:

### Auto-discovery (recommended for new tools)

Drop a `tool_manifest.json` in the tool directory:

```json
{
  "name": "My Tool",
  "path": "launch_pyqt6.py",
  "type": "python",
  "description": "Brief description",
  "category": "Development Tools"
}
```

The launcher scans `src/` for these files at startup and includes them automatically.

### Centralized registry (`tools.json`)

Edit the root `tools.json` to add a tool with explicit ordering:

```json
{
  "Development Tools": [
    {
      "name": "My Tool",
      "path": "src/my_tool/launch_pyqt6.py",
      "type": "python",
      "desc": "Brief description"
    }
  ]
}
```

### PyQt6 launcher integration (`gui_registration.py`)

Every tool that opens a PyQt6 window must have a `gui_registration.py` in its root directory. This file exposes a `get_gui_info()` function returning the module path, widget class, dependencies, and minimum window size. See the full reference in [`docs/architecture/PLUGIN_SYSTEM.md`](architecture/PLUGIN_SYSTEM.md).

---

## Making Your First Change

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-feature
```

Follow the naming convention: `feature/short-description`.

### 2. Edit Code

Make your changes in the appropriate module:
- **Plugin system**: `src/python/src/core/`
- **Shared utilities**: `src/python/src/utils/`
- **New tool**: `src/tools/my_tool/` (see BUILD_A_TOOL.md)

### 3. Test Your Changes

```bash
# Run tests for the module you modified
python -m pytest tests/contract -v  # API contract tests (critical)
python -m pytest tests/integration -v  # Cross-module tests

# Or run all tests
python -m pytest
```

### 4. Format and Lint

```bash
python -m ruff format .
python -m ruff check --fix .
```

### 5. Commit and Push

```bash
git add .
git commit -m "Brief description of changes"
git push origin feature/my-feature
```

### 6. Open a Pull Request

Go to GitHub and create a PR. CI will:
1. Run linting checks (Ruff)
2. Run formatting checks
3. Run the full test suite (Python 3.10, 3.11, 3.12)
4. Check code coverage
5. Verify manifest changes

All must pass before merge.

## Troubleshooting

### Python Import Errors

**Error**: `ImportError: cannot import name 'StrEnum' from 'enum'`

**Cause**: You're on Python 3.9 or 3.10 (some features need 3.11+).

**Solution**:
```bash
# Check your Python version
python --version

# If <3.10, upgrade:
# Ubuntu: sudo apt install python3.12
# macOS: brew install python@3.12
# Windows: Download from python.org

# Or use the compatibility shim in the launcher
# (It should work on Python 3.10+ with fallbacks)
```

### Launcher Won't Start

**Error**: `ModuleNotFoundError: No module named 'PyQt6'`

**Solution**:
```bash
pip install --upgrade PyQt6>=6.6.0
```

**Error**: `RuntimeError: Could not find the Qt platform plugin`

**Cause**: Display/GUI environment issue (common in headless/SSH sessions).

**Solution**:
```bash
# If running remotely, use X11 forwarding:
ssh -X user@host  # Then run the launcher

# Or use environment variable:
export QT_QPA_PLATFORM=offscreen
# (But the GUI won't be visible)
```

### MATLAB Tools Not Working

**Error**: Tool launches but MATLAB window never opens.

**Cause**: MATLAB not installed or not in system PATH.

**Solution**:
```bash
# Check if MATLAB is installed
which matlab  # Should show path to MATLAB binary

# If not found, add MATLAB to PATH:
# Linux/macOS: Add to ~/.bashrc or ~/.zshrc
export PATH="/usr/local/MATLAB/R2023a/bin:$PATH"

# Windows: Add to System Environment Variables
# C:\Program Files\MATLAB\R2023a\bin

# Verify it works:
matlab -batch "version"
```

### Tests Not Running

**Error**: `pytest: No module named 'pytest'`

**Solution**:
```bash
pip install pytest>=8.2.0
```

**Error**: `FAILED tests/... - ModuleNotFoundError: No module named 'tools'`

**Cause**: Not in repo root or venv not activated.

**Solution**:
```bash
# Verify you're in the repo root
pwd  # Should end with /Tools

# Verify venv is active
which python  # Should show path with /venv/

# If not activated:
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate      # Windows
```

### Git LFS Issues

**Error**: Large files show as text placeholders (e.g., `version https://git-lfs.github.com/spec/v1`)

**Solution**:
```bash
# Install Git LFS
git lfs install

# Pull the actual files
git lfs pull

# Verify it worked
git lfs ls-files  # Shows tracked files
```

### Pre-commit Hook Failures

**Error**: `commit failed: hook declined to update the ref`

**Cause**: Code fails linting or formatting checks.

**Solution**:
```bash
# Auto-fix what you can
python -m ruff check --fix .
python -m ruff format .

# Review remaining errors
python -m ruff check .

# Try committing again
git commit -m "Your message"
```

## Next Steps

1. **Read the architecture**: `docs/ARCHITECTURE_OVERVIEW.md`
2. **Build a simple tool**: Follow `docs/BUILD_A_TOOL.md` tutorial
3. **Understand governance**: Read `CLAUDE.md` (critical for shared library work)
4. **Check the plugin system**: `docs/architecture/PLUGIN_SYSTEM.md`
5. **Review coding standards**: `docs/development/GUARDRAILS_GUIDELINES.md`

## Getting Help

- **GitHub Issues**: [D-sorganization/Tools/issues](https://github.com/D-sorganization/Tools/issues)
- **Discussions**: GitHub Discussions (if enabled)
- **Slack** (if available): Check team workspace
- **README**: Detailed troubleshooting in [README.md](../README.md)
- **CONTRIBUTING**: Guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md)

## Quick Reference

| Task | Command |
|------|---------|
| Activate venv | `source venv/bin/activate` |
| Install deps | `python -m pip install -r requirements.txt` |
| Run launcher | `python UnifiedToolsLauncher.py` |
| Run tests | `python -m pytest` |
| Check style | `python -m ruff check .` |
| Auto-format | `python -m ruff format .` |
| Create branch | `git checkout -b feature/name` |
| Run one test | `python -m pytest tests/file.py::test_name -v` |

## Tips for Success

1. **Always work on a feature branch**—never commit to `main`.
2. **Run tests before pushing**—CI will catch issues, but faster locally.
3. **Use the launcher often**—it's the user-facing interface, good to verify your changes work there.
4. **Read CLAUDE.md**—it's short and contains critical governance rules.
5. **Check the contract tests**—they define the API surface; breaking them breaks downstream repos.

---

**Ready to start?** Check out `BUILD_A_TOOL.md` to create your first tool!
