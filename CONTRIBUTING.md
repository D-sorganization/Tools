# Contributing to Tools Repository

This repository follows a strict "Safety First" contribution policy.

## Quick Reference

- **Policies**: See [AGENTS.md](AGENTS.md) for mandatory standards.
- **CI/CD**: See `.github/workflows/ci-standard.yml`.

## Developer Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Environment Setup**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Pre-commit Hooks**
    ```bash
    # Install development dependencies if not already present
    pip install ruff black mypy pre-commit
    pre-commit install
    ```

## Workflow

1.  **Branching**: `git checkout -b feature/your-feature-name`
2.  **Linting**: Run `pre-commit run --all-files` before committing.
3.  **Testing**: Run `pytest` to ensure no regressions.
4.  **Pull Request**: Submit PR targeting `main`.

**Note**: Direct pushes to `main` are blocked.
