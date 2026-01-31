# Assessment K Results: Reproducibility & Provenance

## Executive Summary

- **Dependency Chaos**: Lack of lockfiles guarantees that installations will drift over time.
- **Hardcoded Paths**: Scripts containing absolute paths or assumptions about `C:/` drives will fail on other machines.
- **Data Versioning**: No DVC or similar system for data files.

## Scorecard

| Category                 | Score | Evidence & Remediation                                                                 |
| ------------------------ | ----- | -------------------------------------------------------------------------------------- |
| Environment              | 2/10  | No lockfiles. **Fix**: Poetry.                                                         |
| Path Independence        | 4/10  | Some `pathlib` usage, but legacy scripts use strings.                                  |
| Data Provenance          | 1/10  | None.                                                                                  |

## Findings Table

| ID    | Severity | Category | Location                 | Symptom            | Fix                  |
| ----- | -------- | -------- | ------------------------ | ------------------ | -------------------- |
| K-001 | Critical | Env      | Root                     | No lockfile        | Generate lockfile    |
| K-002 | Major    | Code     | `Data_Processor_r0.py`   | Hardcoded paths    | Use `pathlib`        |

## Refactoring Plan

**48 Hours:**
-   Scan for absolute paths and replace with relative.

**2 Weeks:**
-   Implement `poetry.lock`.
