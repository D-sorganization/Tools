# Assessment D Results: Error Handling

## Executive Summary
- Heavy reliance on base `Exception` in UI code.
- Scientific calculators crash when `NotImplementedError` is raised instead of displaying user warnings.
- The unified launcher fails silently when child processes exit non-zero.

## Top 10 Risks
1. [Major] Broad exception handling in `main_window.py` across all GUI tools.
2. [Major] Scientific tools use raw exceptions instead of typed error domains.
3. [Minor] Logging levels are inconsistently applied during failures.

## Scorecard
| Category | Description | Weight | Score | Notes |
|----------|-------------|--------|-------|-------|
| Typed Exceptions | Custom error hierarchies used | 2x | 4/10 | Lacking in UI |
| Graceful Degradation | Does the app recover? | 2x | 5/10 | UI often crashes |

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| D-001 | Major | Exceptions | `psa_gui.py` | Silent failure | Broad except | Specific exception types | M |

## Refactoring Plan
**48 Hours**:
- Replace broad `except Exception:` blocks in top level PyQt6 main files with proper error dialogue handling.
