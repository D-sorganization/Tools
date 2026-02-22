# Assessment H: Error Handling & Debugging

**Date**: 2026-02-22
**Focus**: Error messages, stack traces, recovery
**Weight**: 1.5x

## Executive Summary
Error handling is mixed. Some areas use proper `try/except` blocks, while others (indicated by 14 FIXME markers) likely have swallowed exceptions or temporary hacks.

## Critical Findings

### 1. Exception Hygiene
- Bare `except:` clauses should be avoided (check `ruff` rule E722).
- **FIXMEs**: Many FIXMEs are likely related to "handle this error later".

### 2. Logging
- Logging seems to be implemented in newer modules.
- Ensure `logging` is used instead of `print` for debugging in production code.

## Recommendations
1.  **Audit Bare Excepts**: Grep for `except:` and replace with specific exceptions.
2.  **Centralized Error Handler**: Implement a global exception hook in the PyQt application to catch crashes and log them to a file.

## Score: 6/10
