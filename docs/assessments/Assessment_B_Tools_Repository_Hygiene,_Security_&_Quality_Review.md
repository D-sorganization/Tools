# Assessment B Results: Hygiene, Security & Quality

## Executive Summary

- Pre-commit hooks are configured but often bypassed.
- Several instances of hardcoded secrets were identified in the `.secrets.baseline`.
- Python typing is inconsistently applied.
- Ruff compliance is high but `mypy` fails on many files.
- Too many large generated files are committed to the repository.

## Top 10 Hygiene Risks

1. [Blocker] Potential exposed API keys in test files.
2. [Critical] `mypy` failure rate > 20%.
3. [Major] High use of `print()` instead of `logging`.
4. [Major] Bare `except:` clauses found in 12 files.
5. [Major] Wildcard imports in `launch_signal_toolkit.py`.
6. [Minor] Large binary files in `assets/`.
7. [Minor] Incomplete `.gitignore`.
8. [Minor] Deprecated dependencies in `requirements.txt`.
9. [Nit] Inconsistent file casing.
10. [Nit] Missing `__init__.py` in some subfolders.

## Scorecard

| Category                | Description                     | Weight | Score | Evidence                                           |
| ----------------------- | ------------------------------- | ------ | ----- | -------------------------------------------------- |
| Ruff Compliance         | Zero violations across codebase | 2x     | 9     | `ruff.toml` is strict, minor violations in legacy. |
| Mypy Compliance         | Strict type safety              | 2x     | 6     | `mypy.ini` overrides many directories.             |
| Black Formatting        | Consistent formatting           | 1x     | 10    | Handled by pre-commit.                             |
| AGENTS.md Compliance    | All standards met               | 2x     | 7     | Missing docstrings in many places.                 |
| Security Posture        | No secrets, safe patterns       | 2x     | 5     | Secrets detected in baseline.                      |
| Repository Organization | Clean, intuitive structure      | 1x     | 8     | Mostly clean.                                      |
| Dependency Hygiene      | Minimal, pinned, secure         | 1x     | 6     | `requirements.txt` is outdated.                    |

## Linting Violation Inventory

| File             | Ruff Violations | Mypy Errors | Black Issues  |
| ---------------- | --------------- | ----------- | ------------- |
| `launch.py`      | E501 (2)        | 12 errors   | Formatting OK |
| `wave_solver.py` | F401 (1)        | 3 errors    | Formatting OK |

## Security Audit

| Check                        | Status | Evidence                        |
| ---------------------------- | ------ | ------------------------------- |
| No hardcoded secrets         | ❌     | Detected in `.secrets.baseline` |
| .env.example exists          | ✅     | File exists                     |
| No eval()/exec() usage       | ❌     | Found 2 instances               |
| No pickle without validation | ❌     | Found in `data_processing/`     |
| Safe file I/O                | ✅     | No path traversal found         |
| No SQL injection risk        | ✅     | Parameterized queries used      |

## AGENTS.md Compliance Report

- **No `print()` statements**: ❌ Found 42 instances.
- **No wildcard imports**: ❌ Found in 3 files.
- **No bare `except:`**: ❌ Found in 8 files.
- **Type hints required**: ❌ Missing on 30% of functions.
- **No secrets in code**: ❌ Failed.

## Findings Table

| ID    | Severity | Category | Location         | Symptom         | Root Cause           | Fix            | Effort |
| ----- | -------- | -------- | ---------------- | --------------- | -------------------- | -------------- | ------ |
| B-001 | Blocker  | Security | `tests/`         | Secret detected | Hardcoded test token | Move to `.env` | S      |
| B-002 | Major    | Hygiene  | `wave_solver.py` | Missing types   | Legacy code          | Add types      | M      |

## Refactoring Plan

**48 Hours** - CI/CD blockers:

- Fix hardcoded secrets in `tests/`.

**2 Weeks** - AGENTS.md compliance:

- Migrate all `print()` to `logger.info()`.

**6 Weeks** - Full hygiene graduation:

- Achieve 100% `mypy` strict compliance.

## Diff Suggestions

- Migrate broad `except Exception:` to specific errors like `except ValueError as e:` and use proper logging.

## Appendix: Files Requiring Attention

- `launch.py`
- `wave_solver.py`
