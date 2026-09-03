# Tools Monorepo 🛠️

[![CI Standard](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml/badge.svg)](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Welcome to the **Tools Monorepo**. This repository houses a comprehensive collection of utility tools for data processing, file management, scientific computing, and project automation. It features Python-based utilities, MATLAB scientific tools, and web-based interfaces.

## 📂 Repository Structure

Canonical topology policy: `docs/architecture/CANONICAL_TOPOLOGY.md`.

The repository is organized into several key areas:

### 🔬 Scientific Computing

- **`matlab/`**: Core scientific code for golf swing modeling and simulations.
- **`src/scientific_modeling/`**: Additional modeling resources and documentation.

### 🛠️ Python Tools

**Directory Structure:**

- **`src/python/`**: Core infrastructure and shared utilities

  - **`src/python/src/core/`**: Plugin system and core launcher functionality
  - **`src/python/src/utils/`**: Shared utilities (compatibility shims, logger utils)
  - **`src/python/src/tile_launcher/`**: Tile launcher components
  - **`src/shared/python/upstream_drift_tools/`**: **NEW** Centralized shared library for fleet-wide logic (Thermo, Conversion, Robotics)
  - **`tests/`**: Canonical root test suite for the shared monorepo surface

- **`src/tools/`**: Tool implementations and utilities

  - **`src/tools/folder_tools/`**: Folder management tools (folder_tool, folder_packer_pro, project_packer)
  - **`src/tools/matlab_utilities/`**: MATLAB quality checking and testing utilities
  - **`src/tools/matlab_code_analyzer_gui/`**: MATLAB code analyzer GUI
  - **`src/tools/scientific_auditor.py`**: Scientific code auditing tool

- **`src/`**: Major tool categories organized under standardized structure
  - **`src/data_processing/`**: Data processing tools and pipelines
  - **`src/document_processing/`**: Document processing utilities
  - **`src/media_processing/`**: Audio and video processing tools
  - **`src/scientific_modeling/`**: Scientific modeling and simulation tools
  - **`src/web_applications/`**: Web-based dashboards and interfaces
  - **`src/verification/`**: Verification and testing utilities

**Note:** The distinction between `src/python/` and `src/tools/` is:

- `src/python/` = Core infrastructure, plugin system, shared utilities
- `src/tools/` = Individual tool implementations and standalone utilities
- `src/` = Major tool categories following standardized `src/` layout pattern

Future consolidation may merge these, but current structure supports the plugin system architecture.

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

See [Launcher Hierarchy & Guide](docs/LAUNCHERS.md) for detailed documentation.

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
- **Python**: Version **3.11+** required (3.12 recommended for best performance).
  - Compatibility shims are retained only for older embedded helper modules; the package metadata requires Python 3.11 or newer.
  - Python 3.13+ not yet tested
  - **CI Testing**: The repository is tested against Python 3.11 and 3.12 (see CI/CD section)
- **MATLAB**: Required for running the core simulations (R2020a or later).
- **Node.js**: Required for web applications and some dev tools.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/D-sorganization/Tools.git
   cd Tools
   git lfs install
   git lfs pull
   ```

2. **Set Up Python Environment**

   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies and the editable package with dev tools
   python -m pip install -r requirements.txt
   python -m pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks** (For developers)

   ```bash
   bash scripts/setup_precommit.sh
   ```

4. **Use the Makefile** (Optional but recommended)

   ```bash
   make help      # Show available targets
   make install   # Install all dependencies
   make check     # Run linters and tests
   make format    # Format code with black and ruff
   ```

### Running the Tools

The easiest way to explore the available tools is via the unified launcher:

```bash
python UnifiedToolsLauncher.py
```

## Optional Rust Acceleration

Several modules in this repository ship optional Rust extensions (built via
[maturin](https://www.maturin.rs/)) that replace hot Python loops with compiled
native code. The extensions are **not distributed as pre-built wheels today** — there
is currently no maturin CI build job. When the wheel is absent the code falls back to
pure-Python automatically and logs a `WARNING` so you know you are on the slow path.

To build the extensions locally:

```bash
pip install maturin
cd rust_core/tools-core && maturin develop --features python
cd rust_core/ai_backend  && maturin develop --features python
```

For full details — what each crate contains, the missing CI workflow spec, and
per-module performance numbers — see
[docs/development/rust_distribution.md](docs/development/rust_distribution.md).

### Local Embeddings (`ai_backend`)

`ai_backend` supports an optional `local-embeddings` feature for offline
ONNX-based embeddings without a remote API. This requires the ONNX Runtime
shared library and the `ORT_DYLIB_PATH` environment variable — especially on
Windows where the library must be downloaded manually.

```bash
# Build with local embeddings (ORT_DYLIB_PATH must be set first)
cd rust_core/ai_backend && maturin develop --features python,local-embeddings

# Preflight check — verifies ORT_DYLIB_PATH before starting your app
python -m src.shared.python.ai._onnx_preflight
```

See [docs/ai_backend_setup.md](docs/ai_backend_setup.md) for the full setup
guide, per-OS instructions, download links, and troubleshooting.

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Architecture](docs/architecture/JULES_ARCHITECTURE.md)**: Overview of the CI/CD system and "Control Tower" architecture.
- **[Fleet Architecture](docs/architecture/FLEET_ARCHITECTURE.md)**: Shared tools architecture across the repository fleet.
- **[Development Guidelines](docs/development/GUARDRAILS_GUIDELINES.md)**: Coding standards, guardrails, and safety protocols.
- **[Branching Strategy](docs/development/BRANCHING_WORKFLOW_RULE.md)**: Mandatory workflow for feature branches and PRs.
- **[Enhanced Tools](docs/tools/ENHANCED_TOOLS.md)**: Documentation for the "Pro" versions of the folder and project tools.
- **[Visualization Guide](docs/VISUALIZATION_GUIDE.md)**: Colorblind-safe plotting and accessibility guidelines.
- **[Plugin System](docs/PLUGIN_SYSTEM.md)**: Automatic tool discovery via manifest files.
- **[Quick Start Guide](QUICKSTART.md)**: Getting started with the Tools repository.
- **[Release Notes](docs/release/CHANGELOG.md)**: History of changes and updates.
- **[Security Policy](SECURITY.md)**: How to report vulnerabilities responsibly.
- **[AI Backend Setup](docs/ai_backend_setup.md)**: ONNX Runtime setup for `local-embeddings`.

## 🤝 Contribution

We follow a strict **"Safety First"** contribution policy.

1. **Branching**: Always use feature branches (`feature/your-feature`). Direct commits to `main` are blocked.
2. **Testing**: All new features must be accompanied by tests. Tests run on Python 3.11 and 3.12.
3. **Linting**: Ensure your code passes all `pre-commit` checks (Ruff, MyPy, etc.).
4. **Review**: All changes require a Pull Request review.
5. **Security**: Report vulnerabilities through the process documented in [SECURITY.md](SECURITY.md), not through public issues.

### CI/CD Testing

The repository uses GitHub Actions for continuous integration:

- **Quality Gate**: Linting (Ruff), formatting (Black), type checking (Mypy), security scanning (pip-audit)
- **Multi-Version Testing**: Tests run on Python 3.11 and 3.12 to ensure compatibility
- **Code Analysis**: Automated code quality checks and security scanning

For more details, please read the [Development Guidelines](docs/development/GUARDRAILS_GUIDELINES.md).

## 🔧 Troubleshooting

### Python Version Issues

**Problem:** `ImportError: cannot import name 'StrEnum' from 'enum'` or `ImportError: cannot import name 'UTC' from 'datetime'`

**Cause:** You're running Python earlier than 3.11, which lacks features required by this package.

**Solutions:**

1. **Recommended:** Use Python 3.11 or newer (3.12 recommended)

   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.12

   # macOS (Homebrew)
   brew install python@3.12
   ```

2. **Note:** Some legacy helper modules still include compatibility shims, but the installable package requires Python 3.11 or newer.

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
4. Check Python version compatibility (3.11+ required, 3.12 recommended)

For more help, see [GitHub Issues](https://github.com/D-sorganization/Tools/issues) or create a new issue.

## 🛡️ License

This project is licensed under the MIT License. See individual tool directories for specific licensing terms where applicable.
