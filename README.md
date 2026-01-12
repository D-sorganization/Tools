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

- **`UnifiedToolsLauncher.py`**: **Primary entry point** - A centralized PyQt6-based launcher for accessing all tools in the repository.
  ```bash
  python UnifiedToolsLauncher.py
  ```

## 🚀 Quick Start

### Prerequisites

- **Git**: Version control (ensure LFS is installed).
- **Python**: Version 3.11+.
- **MATLAB**: Required for running the core simulations.
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

## 🛡️ License

This project is licensed under the MIT License. See individual tool directories for specific licensing terms where applicable.
