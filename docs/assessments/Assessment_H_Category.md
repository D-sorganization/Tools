# Assessment H: Error Handling & Debugging

## Executive Summary
**Score: 4/10**
**Severity: MAJOR**

Error handling is inconsistent. Newer web apps use structured logging and proper HTTP error codes, but legacy desktop tools often swallow exceptions or print stack traces to stdout, which is invisible to the user.

## Key Findings

### 1. Silent Failures
- **Issue**: `try/except: pass` blocks found in `upstream_drift_tools` and `performance_utils.py` hide critical failures (e.g., file access denied).
- **Impact**: Users see "nothing happen" instead of an error message.

### 2. Logging
- **Strengths**: `utils.logging_utils` exists.
- **Weaknesses**: Many scripts still use `print()` for debugging info. This clutters stdout and makes parsing logs difficult.

### 3. User Feedback
- **Issue**: Desktop apps often lack a status bar or error dialog. If a calculation fails, the UI simply stays static.

## Recommendations
1. **Eliminate Silent Failures**: Replace `pass` in except blocks with `logger.warning()` or `logger.error()`.
2. **Standardize Logging**: Replace all `print()` calls with the shared `logging_utils` logger.
3. **UI Error Dialogs**: Implement a global exception handler in `UnifiedToolsLauncher` (and legacy apps) to display a popup dialog with the traceback when a crash occurs.
