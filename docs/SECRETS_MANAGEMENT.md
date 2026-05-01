# Secrets Management Guide

This document outlines how to handle credentials, API keys, and sensitive data in the Tools repository.

## Overview

The Tools repository is a shared library consumed by UpstreamDrift and Gasification_Model. As a critical dependency, it must maintain strict security standards:

- **No real secrets committed** to the repository
- **Test fixtures use OWASP-safe values** only
- **Detect-secrets baseline is locked** to prevent regressions
- **GitHub API rate limiting** is implemented and monitored
- **All path references** use environment variables or platform-agnostic methods

## OWASP-Safe Test Values

When writing tests that require credential-like strings, use the following naming convention:

### Prefixes

- **API Keys**: `OWASP-TEST-API-KEY-{SERVICE}-EXAMPLE`
  - Example: `OWASP-TEST-API-KEY-OPENAI-EXAMPLE`
  - Example: `OWASP-TEST-API-KEY-GITHUB-EXAMPLE`

- **Authentication Tokens**: `OWASP-TEST-TOKEN-{SERVICE}-EXAMPLE`
  - Example: `OWASP-TEST-TOKEN-GITHUB-EXAMPLE`
  - Example: `OWASP-TEST-TOKEN-SAFE-FOR-TESTING-ONLY`

- **Passwords**: `OWASP-TEST-PASSWORD-{SERVICE}-EXAMPLE`
  - Example: `OWASP-TEST-PASSWORD-DB-EXAMPLE`

- **Secrets/Secret Keys**: `OWASP-TEST-SECRET-KEY-SAFE-FOR-TESTING-ONLY`

- **General**: `OWASP-TEST-{THING}-SAFE-VALUE`
  - Example: `OWASP-TEST-JWT-SAFE-VALUE`

### Why These Prefixes?

The `OWASP-TEST-` prefix serves multiple purposes:

1. **Easily searchable**: Any commit containing the prefix is likely a test value
2. **Unmistakable intent**: Developers immediately recognize it as a test value
3. **OWASP standard**: Aligns with OWASP testing guidelines
4. **Whitelisted by detect-secrets**: The baseline recognizes these as non-secrets

## Examples

### Good: Using OWASP-safe test values

```python
# Test file: tests/test_config_loader.py
def test_loads_config_from_file(tmp_path):
    config_data = {"api_key": "OWASP-TEST-API-KEY-SAFE-VALUE", "debug": True}
    config_file = tmp_path / "settings.json"
    config_file.write_text(json.dumps(config_data))

    result = load_config(config_file)
    assert result == config_data
```

```python
# Test file: tests/document_processing/test_pdf_renamer_config_security.py
def test_setup_api_key_interactive_saves_keyring_not_env_file(monkeypatch, tmp_path):
    answers = iter(["y", "OWASP-TEST-API-KEY-SAFE-VALUE"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert config.setup_api_key_interactive() is True
```

### Bad: Hardcoded real-looking secrets

```python
# DON'T DO THIS - will fail detect-secrets
def test_api_call(tmp_path):
    config_data = {"api_key": "sk-proj-a1b2c3d4e5f6g7h8", "debug": True}
    # This will be flagged by detect-secrets!
```

## Environment Variables

### Naming Convention: `TOOLS_*` Prefix

All new optional-service credentials **must** use the `TOOLS_*` prefix (issue #2407):

| Canonical Name            | Purpose                                     | Legacy name (still accepted) |
| ------------------------- | ------------------------------------------- | ----------------------------- |
| `TOOLS_GITHUB_TOKEN`      | GitHub API; raises rate limit 60→5000/hr    | `GITHUB_TOKEN`               |
| `TOOLS_GEMINI_API_KEY`    | Gemini AI for PDF renaming                  | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| `TOOLS_MATLAB_PATH`       | Full path to `matlab` executable            | (none)                        |

The `TOOLS_` prefix:
- Namespaces all Tools variables to avoid collisions with system or third-party env vars
- Makes it clear in shell environment dumps which variables belong to this project
- Enables future tooling to validate/warn on missing service variables at startup

When writing code that reads these variables, always check the canonical `TOOLS_*` name
first, then the legacy name as a fallback:

```python
import os

# Good: canonical first, legacy fallback
token = os.environ.get("TOOLS_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")

# Bad: only legacy name
token = os.environ.get("GITHUB_TOKEN")
```

Use the `.env.example` file as a reference for safe environment variable values:

```bash
# Good: Use OWASP-safe values in .env
TOOLS_GITHUB_TOKEN=OWASP-TEST-TOKEN-GITHUB-EXAMPLE
TOOLS_GEMINI_API_KEY=OWASP-TEST-API-KEY-GEMINI-EXAMPLE
SECRET_KEY=OWASP-TEST-SECRET-KEY-SAFE-FOR-TESTING-ONLY

# Bad: NEVER commit real credentials
# TOOLS_GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TOOLS_GEMINI_API_KEY=AIzaSy-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## GitHub API Hardening

The Tools repository includes hardening for GitHub API interactions:

### Rate Limit Handling

The `model_generation.library._rate_limiter` module provides:

1. **Rate-limit header extraction**: Reads `X-RateLimit-*` headers from GitHub responses
2. **Exponential backoff**: Automatically retries on 429 (Too Many Requests) errors
3. **Status logging**: Logs rate-limit status for monitoring

Usage example:

```python
from model_generation.library._rate_limiter import make_request_with_backoff

# Make a request with automatic rate-limit handling
response = make_request_with_backoff(
    url="https://api.github.com/repos/owner/repo",
    headers={"Authorization": "token GITHUB_TOKEN"},
    max_retries=3,
)

# Extract and log rate-limit info
rate_limit_info = extract_rate_limit_info(response)
log_rate_limit_status(url, rate_limit_info, status_code=200)
```

### Constants

- `DEFAULT_MAX_RETRIES = 3`: Maximum number of retry attempts
- `DEFAULT_INITIAL_BACKOFF = 1.0`: Initial backoff delay in seconds
- `DEFAULT_MAX_BACKOFF = 32.0`: Maximum backoff delay in seconds

## Detect-Secrets Baseline

The `.secrets.baseline` file maintains a whitelist of known-safe secrets in the repository.

### Locking the Baseline

The baseline is version-controlled and should not change except when:

1. **Adding legitimate test fixtures**: Update after adding OWASP-safe test values
2. **Removing exposed secrets**: Update after fixing actual security issues

### Verifying the Baseline

```bash
# Check for new secrets that aren't in the baseline
python3 -m detect_secrets scan

# Update baseline if new OWASP-safe test values are added
python3 -m detect_secrets scan --baseline .secrets.baseline
```

## CI Integration

The Continuous Integration pipeline enforces:

1. **detect-secrets scan**: Fails if non-whitelisted secrets are detected
2. **ruff check**: Enforces code style and security checks
3. **Test coverage**: Minimum 10% coverage on touched files
4. **No `print()` statements in `src/`**: Use logging instead

## Paths and System Information

### MATLAB Path Normalization

Don't hardcode MATLAB paths. Instead:

```python
# Bad: Hardcoded paths expose system information
matlab_root = "/usr/local/MATLAB/R2023b"
matlab_bin = "/Applications/MATLAB_R2023b.app/bin/matlab"

# Good: Use environment variables or which()
import shutil
matlab_cmd = shutil.which("matlab")
if matlab_cmd:
    print(f"Found MATLAB at {matlab_cmd}")
else:
    print("MATLAB not found in PATH")
```

### Environment Variables

```python
import os

# Use environment variables for paths
matlab_root = os.environ.get("MATLAB_ROOT")
if not matlab_root:
    # Fallback to searching PATH
    import shutil
    matlab_cmd = shutil.which("matlab")
```

## Reporting Issues

If you discover exposed secrets:

1. **Do not commit** a fix that simply removes the secret (history is permanent)
2. **Open a GitHub issue** referencing this guide
3. **Contact the maintainers** before making changes to `.secrets.baseline`
4. **Use `git filter-branch`** or similar tools to remove from history if necessary

## Related Documentation

- [GitHub API Rate Limiting](https://docs.github.com/en/rest/overview/rate-limits-for-the-rest-api)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [Detect-Secrets Documentation](https://github.com/Yelp/detect-secrets)
- [CLAUDE.md](../CLAUDE.md) - Project conventions and CI requirements
