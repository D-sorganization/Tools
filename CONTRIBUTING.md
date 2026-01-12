# Contributing to Tools Monorepo

Thank you for your interest in contributing! This document provides guidelines for contributing to the Tools repository.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your feature: `git checkout -b feature/your-feature-name`
4. **Make changes** following our coding standards
5. **Test** your changes
6. **Commit** with a descriptive message
7. **Push** and create a Pull Request

## 📋 Development Setup

```bash
# Clone the repository
git clone https://github.com/D-sorganization/Tools.git
cd Tools

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install ruff black mypy pytest
```

## ✅ Code Standards

### Python

- **Formatter**: Black (default settings)
- **Linter**: Ruff
- **Type Checker**: MyPy (strict mode)
- Use type hints for all public functions
- Use `logging` instead of `print()` for non-trivial code
- Use `pathlib` over `os.path`

### Before Committing

```bash
# Format code
black .

# Lint
ruff check . --fix

# Type check
mypy .

# Run tests
pytest
```

## 🔒 Security Guidelines

- **Never** use `shell=True` in subprocess calls
- **Never** use `eval()` or `exec()`
- **Never** commit credentials or tokens
- Validate all file paths before operations

## 📝 Commit Messages

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding/updating tests
- `chore:` Maintenance tasks

Example: `feat(launcher): Add dark mode support`

## 🧪 Testing

- Add tests for new functionality
- Ensure existing tests pass
- Tests should be deterministic (no external dependencies)

## 📖 Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md under [Unreleased]
- Add docstrings to all public functions

## 🤝 Pull Request Process

1. Ensure all CI checks pass
2. Update documentation as needed
3. Request review from maintainers
4. Address review feedback promptly

## 📫 Questions?

Open an issue for questions or discussions.
