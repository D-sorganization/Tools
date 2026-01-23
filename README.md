# Tools Monorepo 🛠️

[![CI Standard](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml/badge.svg)](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Welcome to the **Tools Monorepo**. This repository houses a comprehensive collection of utility tools for data processing, file management, scientific computing, and project automation. It features Python-based utilities, MATLAB scientific tools, and web-based interfaces.

## 📂 Repository Structure

The repository is organized into several key areas:

### 🔬 Scientific Computing

- **`matlab/`**: Core scientific code for golf swing modeling and simulations.
- **`scientific_modeling/`**: Additional modeling resources and documentation.

### 🛠️ Python Tools

**Directory Structure:**
- **`python/`**: Core infrastructure and shared utilities
  - **`python/src/core/`**: Plugin system and core launcher functionality
  - **`python/src/utils/`**: Shared utilities (compatibility shims, logger utils)
  - **`python/src/tile_launcher/`**: Tile launcher components
  - **`python/shared/`**: Performance utilities and shared code
  - **`python/tests/`**: Test suite for core functionality

- **`tools/`**: Tool implementations and utilities
  - **`tools/folder_tools/`**: Folder management tools (folder_tool, folder_packer_pro, project_packer)
  - **`tools/matlab_utilities/`**: MATLAB quality checking and testing utilities
  - **`tools/matlab_code_analyzer_gui/`**: MATLAB code analyzer GUI
  - **`tools/scientific_auditor.py`**: Scientific code auditing tool

**Note:** The distinction between `python/` and `tools/` is:
- `python/` = Core infrastructure, plugin system, shared utilities
- `tools/` = Individual tool implementations and standalone utilities

Future consolidation may merge these, but current structure supports the plugin system architecture.

### 🌐 Web Applications

- **`web_applications/`**: Web-based dashboards and interfaces for the simulators and tools.

### 🚀 Launcher

The repository provides a unified launcher system for accessing all tools. The canonical entry point is:

- **`UnifiedToolsLauncher.py`**: **PRIMARY AND RECOMMENDED** - Modern PyQt6-based GUI launcher
  ```bash
  python UnifiedToolsLauncher.py
  ```
  
  **Features:**
  - Full plugin system support via `core/plugin_manager.py`
  - Comprehensive error handling and user feedback
  - Tool path validation and sanitization
  - Output/error capture for launched tools
  - Debug mode for troubleshooting
  - Activity log for monitoring tool launches

#### Launcher Hierarchy

1. **`UnifiedToolsLauncher.py`** (Primary) - Use this for all new development and general usage
   - Location: Repository root
   - Type: PyQt6 GUI application
   - Status: ✅ Active and maintained
   - Entry point: `python UnifiedToolsLauncher.py`

2. **`launch_tools_main.py`** (Legacy CLI) - Deprecated
   - Location: Repository root
   - Type: Command-line interface
   - Status: ⚠️ Deprecated - retained for backwards compatibility only
   - Migration: Use `UnifiedToolsLauncher.py` instead

3. **`Launcher.py`** (Legacy GUI) - No longer maintained
   - Location: Repository root (if exists)
   - Type: Original GUI launcher
   - Status: ❌ Migrated to `UnifiedToolsLauncher.py`
   - Migration: Use `UnifiedToolsLauncher.py` instead

> **Important:** `tools_launcher.py` does not exist and any references to it are outdated. Use `UnifiedToolsLauncher.py` as the canonical entry point.

> **Note:** Legacy launchers will be removed in v2.0. Please migrate to `UnifiedToolsLauncher.py`.

## 🚀 Quick Start

### Prerequisites

- **Git**: Version control (ensure LFS is installed).
- **Python**: Version **3.10+** required (3.12 recommended for best performance).
  - Compatibility shims are included for Python 3.10 support (see Troubleshooting below)
  - Python 3.13+ not yet tested
- **MATLAB**: Required for running the core simulations (R2020a or later).
- **Node.js**: Required for web applications and some dev tools.

### Installation

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    git lfs install
    git lfs pull
    ```

2.  **Set Up Python Environment**

    ```bash
    # Create a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install dependencies
    pip install -r python/requirements.txt
    ```

3.  **Install Pre-commit Hooks** (For developers)
    ```bash
    bash scripts/setup_precommit.sh
    ```

### Running the Tools

The easiest way to explore the available tools is via the unified launcher:

```bash
python UnifiedToolsLauncher.py
```

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Architecture](docs/architecture/JULES_ARCHITECTURE.md)**: Overview of the CI/CD system and "Control Tower" architecture.
- **[Development Guidelines](docs/development/GUARDRAILS_GUIDELINES.md)**: Coding standards, guardrails, and safety protocols.
- **[Branching Strategy](docs/development/BRANCHING_WORKFLOW_RULE.md)**: Mandatory workflow for feature branches and PRs.
- **[Enhanced Tools](docs/tools/ENHANCED_TOOLS.md)**: Documentation for the "Pro" versions of the folder and project tools.
- **[Release Notes](docs/release/CHANGELOG.md)**: History of changes and updates.

## 🤝 Contribution

We follow a strict **"Safety First"** contribution policy.

1.  **Branching**: Always use feature branches (`feature/your-feature`). Direct commits to `main` are blocked.
2.  **Testing**: All new features must be accompanied by tests.
3.  **Linting**: Ensure your code passes all `pre-commit` checks (Ruff, MyPy, etc.).
4.  **Review**: All changes require a Pull Request review.

For more details, please read the [Development Guidelines](docs/development/GUARDRAILS_GUIDELINES.md).

## 🔧 Troubleshooting

### Python Version Issues

**Problem:** `ImportError: cannot import name 'StrEnum' from 'enum'` or `ImportError: cannot import name 'UTC' from 'datetime'`

**Cause:** You're running Python 3.10, which lacks some features introduced in Python 3.11+.

**Solutions:**
1. **Recommended:** Use Python 3.10 or newer (3.12 recommended)
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.12

   # macOS (Homebrew)
   brew install python@3.12
   ```

2. **Note:** The repository includes compatibility shims in `python/src/utils/compatibility.py` that allow running on Python 3.10+. The application will provide a friendly error message if your Python version is incompatible.

### Launcher Won't Start

**Problem:** `UnifiedToolsLauncher.py` fails to launch or crashes immediately.

**Solutions:**
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version: `python --version` (must be 3.11+)
3. Try running with verbose output: `python UnifiedToolsLauncher.py --verbose`
4. Check for missing PyQt6: `pip install PyQt6>=6.6.0`

### MATLAB Tools Not Working

**Problem:** MATLAB-based tools (Audio Processor, RRT Path Planner, Scientific Modeling tools) fail silently or cannot be launched.

**Cause:** These tools require MATLAB to be installed and accessible in your system PATH.

**MATLAB Requirements:**
- **Minimum Version:** MATLAB R2020a or later
- **Required Toolboxes:** 
  - Signal Processing Toolbox (for audio processing tools)
  - Statistics and Machine Learning Toolbox (for some modeling tools)
  - Image Processing Toolbox (for visualization tools)

**Solutions:**

1. **Install MATLAB**
   - Download from [MathWorks](https://www.mathworks.com/products/matlab.html)
   - Ensure R2020a or later is installed
   - Install required toolboxes during setup

2. **Add MATLAB to System PATH**
   ```bash
   # Linux/macOS
   export PATH="/usr/local/MATLAB/R2023a/bin:$PATH"
   # Or for your specific installation:
   export PATH="/path/to/matlab/bin:$PATH"

   # Windows (PowerShell)
   $env:PATH += ";C:\Program Files\MATLAB\R2023a\bin"
   
   # Windows (Command Prompt) - Add to System Environment Variables permanently
   ```

3. **Verify MATLAB Installation**
   ```bash
   # Check MATLAB version
   matlab -batch "version"
   
   # Test MATLAB execution
   matlab -batch "disp('MATLAB is working')"
   ```

4. **Tool Availability**
   - **Audio Processor**: Requires MATLAB + Signal Processing Toolbox
   - **RRT Path Planner**: Requires MATLAB + Statistics Toolbox
   - **Solar System Model**: Requires MATLAB (basic installation sufficient)
   - **Golf Modeling Suite**: Requires MATLAB + Optimization Toolbox

5. **If MATLAB is Not Available**
   - Python-only tools will still work
   - Web applications are independent of MATLAB
   - Some tools have Python alternatives (check individual tool documentation)

**Note:** The launcher will attempt to open MATLAB files in your default editor if MATLAB is not found in PATH, but full functionality requires MATLAB to be properly installed and configured.

### Tests Not Running

**Problem:** `pytest` fails with collection errors or import errors.

**Solutions:**
1. Ensure you're in the repository root: `cd /path/to/Tools`
2. Install test dependencies: `pip install pytest>=8.2.0`
3. Run from virtual environment: `source venv/bin/activate`
4. Check Python version compatibility (3.10+ required, 3.12 recommended)

For more help, see [GitHub Issues](https://github.com/D-sorganization/Tools/issues) or create a new issue.

## 🛡️ License

This project is licensed under the MIT License. See individual tool directories for specific licensing terms where applicable.
