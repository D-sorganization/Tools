# Assessment B Results: Hygiene, Security & Quality

## Executive Summary

- **Critical Security Vulnerabilities Detected**: Active usage of `eval()` and `exec()` on potentially untrusted input constitutes a **BLOCKER** severity risk.
- **Pervasive Hygiene Violations**: `print()` statements are used universally for logging, violating AGENTS.md directives.
- **Type Safety is Non-Existent**: Hundreds of missing type definitions (`no-untyped-def`) render static analysis largely ineffective.
- **Dependency Management is Loose**: Multiple `requirements.txt` files without lockfiles create reproducibility risks.
- **Legacy Artifacts**: Files like `Data_Processor_r0.py` indicate a "commit-and-forget" mentality regarding prototype code.

## Top 10 Hygiene Risks

1.  **Arbitrary Code Execution (Blocker)**: `eval()` in `Data_Processor_r0.py` and `fitting.py`.
2.  **Dynamic Execution (Blocker)**: `exec()` in `verify_installation.py`.
3.  **Logging Failure (Critical)**: `print()` used instead of `logging` module in >90% of files.
4.  **Type Blindness (Major)**: 70+ MyPy errors regarding untyped definitions.
5.  **Bare Excepts (Major)**: `except:` found in scripts, masking potential errors.
6.  **Hardcoded Secrets (Potential)**: `API_KEY_QUICK_REFERENCE.txt` suggests secrets might be stored in plain text.
7.  **Dead Code (Minor)**: `r0` and `deprecated` files cluttering the repo.
8.  **Import Sorting (Minor)**: Ruff reports import sorting violations (I001).
9.  **Formatting (Minor)**: Black found unformatted files.
10. **Structure (Minor)**: `__init__.py` files missing in some test directories.

## Scorecard

| Category                | Score | Evidence & Remediation                                                                 |
| ----------------------- | ----- | -------------------------------------------------------------------------------------- |
| Ruff Compliance         | 8/10  | Only ~7 violations found. Code is generally lint-clean.                                |
| Mypy Compliance         | 2/10  | Hundreds of violations. **Fix**: Add type hints to all function signatures.            |
| Black Formatting        | 9/10  | Most files are formatted.                                                              |
| AGENTS.md Compliance    | 1/10  | `print()` usage is rampant. **Fix**: Replace all `print()` with `logger.info()`.       |
| Security Posture        | 1/10  | `eval()` usage is unacceptable. **Fix**: Immediate removal.                            |
| Repository Organization | 5/10  | Cluttered root.                                                                        |
| Dependency Hygiene      | 4/10  | Fragmented requirements.                                                               |

## Linting Violation Inventory

**Ruff**: 7 Violations (I001, UP022, W293, E722, W605, UP015).
**Black**: 2 files need formatting.
**MyPy**: ~70 files with missing type annotations.

## Security Audit

| Check                        | Status | Evidence                                                                 |
| ---------------------------- | ------ | ------------------------------------------------------------------------ |
| No hardcoded secrets         | ⚠️     | `API_KEY_QUICK_REFERENCE.txt` found.                                     |
| .env.example exists          | ✅     | `.env.example` exists.                                                   |
| No eval()/exec() usage       | ❌     | `eval()` in `Data_Processor_r0.py`, `fitting.py`; `exec()` in `verify*`. |
| Safe file I/O                | ⚠️     | Hardcoded paths common.                                                  |
| No SQL injection risk        | ✅     | No SQL usage detected.                                                   |

## AGENTS.md Compliance Report

1.  **Print Statements**: **FAILED**. `print()` is the primary output mechanism.
2.  **Wildcard Imports**: **PASSED**. Few to no wildcard imports found.
3.  **Bare Except Clauses**: **FAILED**. `except:` found in scripts.
4.  **Missing Type Hints**: **FAILED**. Widespread lack of typing.

## Findings Table

| ID    | Severity | Category | Location                 | Symptom            | Root Cause | Fix                  | Effort |
| ----- | -------- | -------- | ------------------------ | ------------------ | ---------- | -------------------- | ------ |
| B-001 | Blocker  | Security | `Data_Processor_r0.py`   | `eval(formula)`    | Unsafe API | Use `numexpr`        | M      |
| B-002 | Blocker  | Security | `fitting.py`             | `eval(expression)` | Unsafe API | Use `ast.literal_eval`| M     |
| B-003 | Critical | Hygiene  | Global                   | `print()` usage    | Laziness   | `logging` migration  | L      |
| B-004 | Major    | Hygiene  | `verify_installation.py` | `exec()`           | Dynamic loading | Importlib       | M      |

## Refactoring Plan

**48 Hours - CI/CD blockers:**
-   Remove `eval()` and `exec()` calls.
-   Delete `API_KEY_QUICK_REFERENCE.txt` if it contains real keys.

**2 Weeks - AGENTS.md compliance:**
-   Mass migration of `print()` to `logging`.
-   Add type hints to core `shared` libraries.

**6 Weeks - Full hygiene graduation:**
-   Enforce strict MyPy in CI.
-   Unify dependencies.

## Diff-Style Suggestions

```python
# Data_Processor_r0.py
<<<<<<< SEARCH
    result = eval(formula, {"__builtins__": {}}, safe_dict)
=======
    import simpleeval
    result = simpleeval.simple_eval(formula, names=safe_dict)
>>>>>>> REPLACE
```
