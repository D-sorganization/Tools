# Quick Start Guide

Welcome to the Tools Repository! This suite contains professional-grade tools for Data Processing, Scientific Modeling, and Media Processing.

## 🚀 Installation

1. **Clone the repository**:

   ```bash
   git clone <repo_url>
   cd Tools
   ```

2. **Setup Environment**:
   Run the automated setup script to check Python versions and install dependencies.
   ```bash
   python setup_dev.py
   ```
   _Requirements: Python 3.10+, Node.js (optional for web apps)._

## 🎮 Launching the Tools

We use a **Unified Tools Launcher** to access all applications from a single interface.

1. **Run the Launcher**:

   ```bash
   python UnifiedToolsLauncher.py
   ```

2. **Select a Tool**:
   - **Data Processing**: Extract signals from CSV/Parquet files.
   - **Scientific Modeling**: Run the Interactive Solar System simulation.
   - **Media Processing**: Launch video/audio processing suites.

## 🛠️ Common Tasks

### Running Tests

Ensure the system is healthy by running the test suite:

```bash
pytest
```

### Type Checking

Run static analysis to verify code quality:

```bash
mypy .
```

## 🆘 Troubleshooting

- **Python Version Error**: Ensure you are using Python 3.10 or newer (`python --version`).
- **Missing Dependencies**: Re-run `python setup_dev.py` or `pip install -r requirements-lock.txt`.
- **Launcher Crashes**: Check the terminal output for error logs.
