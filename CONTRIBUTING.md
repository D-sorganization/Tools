# Contributing Guide

Welcome to the project! We appreciate your interest in contributing.

## Governance & Standards

Please refer to [AGENTS.md](AGENTS.md) for the authoritative guide on:

- **Coding Standards** (Python, JavaScript, MATLAB, C++)
- **Architecture & Agent Roles**
- **Git Workflow & Commit Conventions**
- **Security Protocols**

## Quick Start

1.  **Environment Setup**:
    - Python 3.10+ required (3.11+ recommended for best compatibility)
    - Install dependencies: `pip install -r requirements.txt`
    - Optional: Run `python setup_dev.py` for additional development setup
2.  **Linting**: Ensure your code passes quality checks before committing.
    - Run `ruff check .` and `ruff format .` before committing
    - Run `black --check .` to verify formatting
    - Run `mypy . --config-file mypy.ini` for type checking (advisory - see note below)

    > **Note on Type Checking**: While `mypy` is part of our quality toolchain, strict type
    > checking is not yet fully enforced across the legacy codebase. New code should include
    > type hints. Existing type errors are tracked in issue #219.

3.  **Testing**: Run relevant tests before submitting a PR.
    - Run `pytest .` to execute all tests
    - Ensure test coverage is maintained or improved
4.  **Tools**: Use `python UnifiedToolsLauncher.py` to access development utilities.
    - This is the canonical entry point (not `tools_launcher.py` which does not exist)

## Pull Requests

- Use **GitHub CLI** (`gh pr create`) for PRs.
- Ensure all CI/CD checks pass.
- Follow the Conventional Commits format (e.g., `feat(scope): description`).
