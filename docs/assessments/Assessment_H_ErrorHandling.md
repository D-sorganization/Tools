# Assessment H: Error Handling & Debugging
**Date**: 2026-02-05
**Focus**: Error messages, stack traces, recovery

## 1. Findings Table

| Area | Status | Notes |
| :--- | :--- | :--- |
| **Logging** | ✅ IMPROVED | Recent switch from `print` to `logging` (with file output) in `Launcher.py` is a major step forward. |
| **UI Recovery** | ⚠️ BASIC | Most UIs wrap execution in broad `try/except Exception` blocks. While this prevents crashes, it sometimes masks the root cause or leaves the app in an inconsistent state. |
| **Diagnostics** | ✅ GOOD | Error dialogs now include log paths. `tools_launcher.log` is central. |
| **Web Apps** | ⚠️ PENDING | `video_processor` still relies on console logging (TODO: migrate to `pino`). |

## 2. Critical Path Analysis
Broad exception swallowing in UIs makes debugging logic errors difficult for developers, even if it "saves" the user from a crash.

## 3. Score
**Grade**: 6/10
**Justification**: Basic safety nets are in place, and logging has improved, but sophisticated error recovery or reporting (e.g., Sentry integration) is missing.

## 4. Recommendations
1.  **Structured Logging**: Adopt structured logging (JSON) for web apps and key tools to enable easier analysis.
2.  **Narrow Exceptions**: Refactor `try/except` blocks to catch specific errors (e.g., `ValueError`, `IOError`) and handle them gracefully, letting unexpected ones crash (or log stack trace explicitly).
3.  **Telemetry**: Consider an opt-in telemetry system for crash reporting.
