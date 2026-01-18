# Tools Monorepo 🛠️

[![CI Standard](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml/badge.svg)](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Welcome to the **Tools Monorepo**. This repository houses a comprehensive collection of utility tools for data processing, file management, scientific computing, and project automation. It features Python-based utilities, MATLAB scientific tools, and web-based interfaces.

## 📂 Repository Structure

The repository is organized into several key areas:

### 🔬 Scientific Computing

- **`matlab/`**: Core scientific code for golf swing modeling and simulations.
- **`scientific_modeling/`**: Additional modeling resources and documentation.

### 🛠️ Python Tools

- **`python/`**: A collection of Python-based utilities and applications.
  - **`folder_tool/`** & **`folder_tool_pro/`**: Advanced directory management and cleanup tools.
  - **`project_packer/`** & **`folder_packer_pro/`**: Secure project packaging and encryption tools.
  - **`data_processing/`**: Scripts and pipelines for data analysis.
- **`tools/`**: General purpose utility scripts.

### 🌐 Web Applications

- **`web_applications/`**: Web-based dashboards and interfaces for the simulators and tools.

### 🚀 Launcher

- **`UnifiedToolsLauncher.py`**: **Primary entry point** - A modern PyQt6-based launcher for accessing all tools in the repository.
  ```bash
  python UnifiedToolsLauncher.py
  ```
  This is the **recommended launcher** with full plugin support, error handling, and tool management.

#### Alternative Launchers (Legacy)

- **`launch_tools_main.py`**: Command-line launcher with basic tool discovery. **Deprecated** - use `UnifiedToolsLauncher.py` for best experience.
- **`Launcher.py`**: Original GUI launcher. **No longer maintained** - migrated to `UnifiedToolsLauncher.py`.

> **Note:** Legacy launchers are retained for backwards compatibility but will be removed in v2.0. Please migrate to `UnifiedToolsLauncher.py`.

## 🚀 Quick Start

### Prerequisites

- **Git**: Version control (ensure LFS is installed).
- **Python**: Version **3.11+** required (3.12 recommended for best performance).
  - Python 3.10 has limited compatibility (see Troubleshooting below)
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
1. **Recommended:** Upgrade to Python 3.11 or 3.12
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.12

   # macOS (Homebrew)
   brew install python@3.12
   ```

2. **Alternative:** The repository includes compatibility shims for Python 3.10 in `python/src/utils/compatibility.py`, but full compatibility is not guaranteed.

### Launcher Won't Start

**Problem:** `UnifiedToolsLauncher.py` fails to launch or crashes immediately.

**Solutions:**
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version: `python --version` (must be 3.11+)
3. Try running with verbose output: `python UnifiedToolsLauncher.py --verbose`
4. Check for missing PyQt6: `pip install PyQt6>=6.6.0`

### MATLAB Tools Not Working

**Problem:** Audio Processor or RRT Path Planner tools fail silently.

**Cause:** These tools require MATLAB to be installed and in your system PATH.

**Solutions:**
1. Install MATLAB R2020a or later
2. Add MATLAB to PATH:
   ```bash
   # Linux/macOS
   export PATH="/path/to/matlab/bin:$PATH"

   # Windows (PowerShell)
   $env:PATH += ";C:\Program Files\MATLAB\R2023a\bin"
   ```
3. Verify MATLAB is accessible: `matlab -batch "version"`

### Tests Not Running

**Problem:** `pytest` fails with collection errors or import errors.

**Solutions:**
1. Ensure you're in the repository root: `cd /path/to/Tools`
2. Install test dependencies: `pip install pytest>=8.2.0`
3. Run from virtual environment: `source venv/bin/activate`
4. Check Python version compatibility (3.11+ required)

For more help, see [GitHub Issues](https://github.com/D-sorganization/Tools/issues) or create a new issue.

## 🛡️ License

This project is licensed under the MIT License. See individual tool directories for specific licensing terms where applicable.
