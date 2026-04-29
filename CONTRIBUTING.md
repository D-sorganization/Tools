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
    - Install dependencies: `python -m pip install -r requirements.txt`
    - Install the editable package and dev tools: `python -m pip install -e ".[dev]"`
    - Optional: Run `python setup_dev.py` for additional development setup
2.  **Linting**: Ensure your code passes quality checks before committing.

    - Run `python -m ruff check .` and `python -m ruff format .` before committing
    - Run `python -m black --check .` to verify formatting
    - Run `python -m mypy . --config-file mypy.ini` for type checking (advisory - see note below)

    > **Note on Type Checking**: While `mypy` is part of our quality toolchain, strict type
    > checking is not yet fully enforced across the legacy codebase. New code should include
    > type hints. Existing type errors are tracked in issue #219.

3.  **Testing**: Run relevant tests before submitting a PR.
    - Run `python -m pytest` to execute the canonical root test suite
    - Ensure test coverage is maintained or improved
4.  **Tools**: Use `python UnifiedToolsLauncher.py` to access development utilities.
    - This is the canonical entry point (not `tools_launcher.py` which does not exist)

## Security: Secrets & Credentials Management (Issue #2356)

**CRITICAL:** Never commit secrets, API keys, passwords, tokens, or credentials to the repository.

### Best Practices

1. **Use Environment Variables**:
   - Store secrets in `.env` files (which are `.gitignore`-excluded).
   - Load at runtime using `python-dotenv`:
     ```python
     from dotenv import load_dotenv
     import os
     
     load_dotenv()
     api_key = os.getenv('API_KEY')
     ```

2. **Create `.env.example` Templates**:
   - Show the required environment variable names with placeholder values (no real secrets).

3. **Use OS Keyring for Interactive Tools**:
   - For GUI apps storing credentials, use `python-keyring` to save to OS keyring (macOS Keychain, Linux Secret Service, Windows Credential Manager).

4. **Exclude Config Files**:
   - `.gitignore` excludes `.env*`, `*.key`, `*.pem`, and `secrets/` directories.
   - Add new config files containing secrets to `.gitignore`.

5. **Scan Before Committing**:
   ```bash
   python3 -m src.python.src.utils.secrets_scanner src/
   ```

6. **Code Review**:
   - Ensure no hardcoded secrets in string literals, docstrings, test fixtures, or logging.
   - Use placeholder values in tests: `"test_password"`, `"mock_api_key"`, etc.

See `SECURITY.md` for detailed guidelines and examples.

## Security Reporting

Do not file public issues for vulnerabilities. Use the process documented in
`SECURITY.md`.

## Pull Requests

- Use **GitHub CLI** (`gh pr create`) for PRs.
- Ensure all CI/CD checks pass.
- Follow the Conventional Commits format (e.g., `feat(scope): description`).
- Verify no secrets are included (see Secrets & Credentials section above).
