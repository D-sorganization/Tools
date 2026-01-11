# Tools Monorepo 🛠️

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

### 🚀 Launchers

- **`UnifiedToolsLauncher.py`**: A centralized launcher for accessing all tools in the repository.
- **`tools_launcher.py`**: Alternative launcher interface.

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
