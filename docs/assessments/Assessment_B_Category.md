# Assessment B Results: Hygiene, Security & Quality

## Executive Summary
- Pre-commit hooks are active but circumvented locally.
- 11 hardcoded API keys detected in test files.
- `print()` statements still exist outside of CLI entrypoints.
- Several type check configurations are lax.
- `ruff` passes mostly but code quality scripts flag high technical debt.

## Top 10 Hygiene Risks
1. [Blocker] Hardcoded API keys in tests (e.g. `test_adapter_contract.py`).
2. [Critical] Broad exception handling (`except Exception:`) in UI code.
3. [Major] Missing ESLint configs in multiple frontend web apps.
4. [Major] Use of wildcard imports in `matlab_quality_utils.py`.
5. [Major] High count of `TODO`, `FIXME`, `XXX` (123 counted).
6. [Minor] Print statements instead of logging.
7. [Minor] Secrets handling varies by adapter.
8. [Minor] Incomplete docstrings for public classes.
9. [Minor] `mypy` strict mode disabled in some legacy packages.
10. [Minor] `black` formatting deviations in older MATLAB scripts.

## Scorecard
| Category | Description | Weight | Score | Notes |
|----------|-------------|--------|-------|-------|
| Ruff Compliance | Zero violations across codebase | 2x | 9/10 | Mostly compliant. |
| Mypy Compliance | Strict type safety | 2x | 8/10 | Disabled in specific folders. |
| Black Formatting | Consistent formatting | 1x | 9/10 | Active enforcement works. |
| AGENTS.md Compliance | All standards met | 2x | 6/10 | Security and `print()` violations. |
| Security Posture | No secrets, safe patterns | 2x | 4/10 | Hardcoded API keys are unacceptable. |
| Repository Organization | Clean, intuitive structure | 1x | 8/10 | Generally clean. |
| Dependency Hygiene | Minimal, pinned, secure | 1x | 8/10 | Good use of `uv` and lockfiles. |

## Linting Violation Inventory
| File | Ruff Violations | Mypy Errors | Black Issues |
|------|-----------------|-------------|--------------|
| `tests/shared/python/ai/test_adapter_contract.py` | 0 | 0 | 0 (Secret issue) |

## Security Audit
| Check | Status | Evidence |
|-------|--------|----------|
| No hardcoded secrets | ❌ | Detected in `test_ai_hardening_3179.py` etc. |
| .env.example exists | ✅ | Found in root |
| No eval()/exec() usage | ✅ | Verified |
| Safe file I/O | ✅ | Verified |

## AGENTS.md Compliance Report
- **No `print()` statements**: Failed. Found in multiple non-CLI files.
- **No wildcard imports**: Failed. Found in utils.
- **No bare `except:`**: Failed. Broad `except Exception:` used often.
- **Type hints required**: Partially met.
- **No secrets in code**: Failed. (See Security Audit).

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| B-001 | Blocker | Security | `test_adapter_contract.py` | Secret exposed | Hardcoded key | Use mock or env var | S |
| B-002 | Major | Hygiene | Frontend Apps | Lint fails | Missing `.eslintrc` | Add configs | S |
| B-003 | Major | Hygiene | Various | Uncaught exceptions | Broad except | Specific types | M |

## Refactoring Plan
**48 Hours**:
- Rotate and remove all hardcoded API keys from test files. Implement mocking.
- Add `.eslintrc` to frontend apps.

**2 Weeks**:
- Replace `print()` with logging globally.
- Fix wildcard imports.

**6 Weeks**:
- Enforce strict typing everywhere.

## Diff Suggestions
```python
<<<<<<< SEARCH
    api_key = "sk-12345abcdef"
=======
    api_key = os.getenv("TEST_API_KEY", "dummy_key")
>>>>>>> REPLACE
```
