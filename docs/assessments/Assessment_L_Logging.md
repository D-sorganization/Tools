# Assessment L: Logging

## Executive Summary
This assessment evaluates the application telemetry, structured logging practices, and debuggability of the repository.
The repository is suffering from an incomplete transition from basic print-debugging to structured, production-ready logging. While `logging` is configured and used in many shared components, there are still 135 `print()` statements scattered throughout the codebase (e.g., `Launcher.py`, legacy UI modules). These `print()` calls lack severity levels, timestamps, and contextual metadata, making it exceedingly difficult to debug issues in production or when running tools via graphical launchers where stdout is hidden.

## Scorecard
- **Grade: 5.0/10**

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| L-001 | Major | Standardization | Global Codebase | 135 `print()` statements | Leftover debug code | Replace with `logger.debug()` or `logger.info()` | M |
| L-002 | Major | Accessibility | PyQt GUI Applications | Users cannot see errors | UI hides `stderr` and `stdout` | Implement an in-app `QTextEdit` log viewer | M |
| L-003 | Medium | Structure | `src/shared/python/utils/logging.py` | Hard to parse logs | Flat text log formatting | Adopt `structlog` for JSON-formatted logs | M |
| L-004 | Minor | Configuration | `launcher_legacy.log` | Unrotated log files | No `RotatingFileHandler` | Add log rotation to prevent disk space exhaustion | S |

## Refactoring Plan
- **Short Term**: Address L-001 by running a global find-and-replace script to convert all `print(...)` calls into `logging.info(...)` or `logging.debug(...)`, ensuring the standard library logger is instantiated in every file.
- **Medium Term**: Implement a `QPlainTextEdit` widget within a docking panel in the `UnifiedToolsLauncher` that acts as a custom `logging.Handler`, surfacing critical and error logs directly to the user (L-002).
- **Long Term**: Migrate from standard Python `logging` to `structlog` (L-003) to generate machine-readable JSON logs, and ensure that all file handlers use `RotatingFileHandler` to manage log file sizes (L-004).
