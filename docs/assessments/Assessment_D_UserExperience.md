# Assessment D: User Experience & Developer Journey
**Date**: 2026-02-05
**Focus**: Time-to-value, onboarding, friction points

## 1. Findings Table

| Area | Status | Notes |
| :--- | :--- | :--- |
| **Launcher UX** | ⚠️ INCONSISTENT | Users face two different launcher interfaces (Tkinter vs PyQt6) with different capabilities. The legacy launcher feels dated compared to the unified one. |
| **Error Feedback** | ✅ IMPROVED | Recent updates to `launch_tools_main.py` now include absolute paths to logs in error dialogs, significantly aiding debugging. |
| **Onboarding** | ⚠️ COMPLEX | Setting up the environment requires navigating `setup_dev.py` and potentially manually setting `PYTHONPATH`. No "One Click" installer exists for end-users. |
| **Tool Consistency** | ⚠️ MIXED | Individual tools vary wildly in look and feel (Matplotlib vs PyQt6 charts, dark mode support, etc.). |

## 2. Critical Path Analysis
The inconsistency between tools makes the suite feel like a loose collection of scripts rather than a cohesive product. The requirement to understand `PYTHONPATH` is a major friction point for non-developer users.

## 3. Score
**Grade**: 6/10
**Justification**: Recent error dialog improvements are a plus, but the fragmented UX and complex setup procedure lower the score.

## 4. Recommendations
1.  **Unified Theme**: Apply a consistent stylesheet (e.g., QtDarkStyle) across all PyQt6 tools.
2.  **Simplified Install**: Create a strictly automated installer (e.g., `install.bat` / `install.sh`) that handles venv creation and path setup invisibly.
3.  **Deprecate Legacy**: Hide the Tkinter launcher from the default user workflow.
