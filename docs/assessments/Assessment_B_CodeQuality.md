# Assessment B: Code Quality & Hygiene
**Date**: 2026-02-05
**Focus**: Linting, formatting, type safety

## 1. Findings Table

| Area | Status | Notes |
| :--- | :--- | :--- |
| **DRY (Don't Repeat Yourself)** | ❌ CRITICAL | Pragmatic Programmer review identified 20+ instances of duplicate code, particularly in `UnifiedToolsLauncher.py`, `setup_dev.py`, and calculator UIs. |
| **Orthogonality** | ⚠️ POOR | "God functions" (e.g., `create_plot_left_content` > 190 lines) create strong coupling and difficult testing scenarios in UI files. |
| **Type Safety** | ⚠️ MIXED | Recent efforts (e.g., `humanoid_character_builder`) enforce MyPy, but legacy tools and scripts often lack type hints or use `Any`. |
| **Linting** | ✅ ENFORCED | CI pipelines enforce Black formatting and Ruff linting, catching many potential issues early. |

## 2. Critical Path Analysis
The duplication in `UnifiedToolsLauncher.py` and `setup_dev.py` is a maintenance ticking time bomb. A bug fix in one location (e.g., path handling) is likely to be missed in the duplicated blocks, leading to inconsistent behavior.

## 3. Score
**Grade**: 5/10
**Justification**: Strict linting in CI saves the grade, but the pervasive copy-paste programming (DRY violations) and monolithic functions indicate a need for a major refactoring sprint.

## 4. Recommendations
1.  **Extract Common Logic**: Create a `src/tools/common/` module to house shared logic for Launchers and Setup scripts.
2.  **Refactor God Functions**: Systematically break down functions > 50 lines (identified in Pragmatic Review) into sub-functions.
3.  **Strict Typing**: Expand strict MyPy settings to `src/tools/` and `src/shared/`.
