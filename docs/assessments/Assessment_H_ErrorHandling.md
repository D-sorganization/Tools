# Assessment H Results: Error Handling & Debugging

## Executive Summary

- **"Print" Debugging**: The codebase relies almost exclusively on `print()` for error reporting, which is untrackable in production.
- **Bare Excepts**: Dangerous `except:` clauses swallow errors, making debugging impossible.
- **Inconsistent UX**: Some tools show a GUI popup on error, others crash to console.
- **No Central Logging**: No centralized log file or rotation strategy.

## Top 10 Error Handling Risks

1.  **Silent Failures (Critical)**: Bare `except:` swallows critical bugs.
2.  **Console Noise (Major)**: `print()` spam hides real errors.
3.  **Traceback Loss (Major)**: Exceptions caught without logging `traceback`.
4.  **GUI Crash (Major)**: Uncaught exceptions in PyQt slots crash the app.
5.  **User Confusion (Moderate)**: "Error: 1" type messages.
6.  **Debug Mode (Minor)**: No flag to enable verbose logging.
7.  **Exit Codes (Minor)**: Scripts return 0 even on failure.
8.  **Assertion Abuse (Minor)**: `assert` used for control flow.
9.  **File I/O (Moderate)**: Race conditions not handled.
10. **Network (Minor)**: No retry logic.

## Scorecard

| Category                 | Score | Evidence & Remediation                                                                 |
| ------------------------ | ----- | -------------------------------------------------------------------------------------- |
| Exception Safety         | 3/10  | Bare excepts. **Fix**: Use `except Exception as e`.                                    |
| Logging Quality          | 1/10  | Prints. **Fix**: Use `logging`.                                                        |
| User Feedback            | 4/10  | Inconsistent.                                                                          |
| Debuggability            | 2/10  | Hard.                                                                                  |

## Findings Table

| ID    | Severity | Category | Location          | Symptom                 | Root Cause           | Fix                  | Effort |
| ----- | -------- | -------- | ----------------- | ----------------------- | -------------------- | -------------------- | ------ |
| H-001 | Critical | Code     | Global            | `except:`               | Lazy coding          | Catch specific       | M      |
| H-002 | Major    | Code     | Global            | `print(e)`              | No logging setup     | `logger.error(e)`    | L      |

## Refactoring Plan

**48 Hours - Critical fixes:**
-   Grep and replace all `except:` with `except Exception:`.

**2 Weeks - Major improvements:**
-   Implement `src/shared/python/utils/logging_utils.py`.
-   Migrate `UnifiedToolsLauncher` to use it.

## Diff-Style Suggestions

```python
# src/shared/python/utils/logging_utils.py (New)
import logging
import sys

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
```
