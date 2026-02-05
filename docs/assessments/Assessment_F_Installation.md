# Assessment F: Installation & Deployment
**Date**: 2026-02-05
**Focus**: pip/conda, cross-platform, CI/CD

## 1. Findings Table

| Area | Status | Notes |
| :--- | :--- | :--- |
| **Dependency Mgmt** | ⚠️ SCATTERED | Multiple `requirements.txt` files exist (root, `pdf_renamer`, etc.). Inline comments in requirements are a nice touch but parser compatibility varies. |
| **Setup Script** | ⚠️ DUPLICATED | `setup_dev.py` contains duplicated logic for path setup and dependency checking, risking divergence. |
| **Environment** | ❌ FRAGILE | Persistent `ModuleNotFoundError` (especially `pandas`) in CI suggests environment isolation issues or PYTHONPATH misconfiguration. |
| **Platform Support** | ✅ GOOD | Code generally aims for cross-platform compatibility (using `os.path.join`, `pathlib`). |

## 2. Critical Path Analysis
The inability to reliably run tests in CI due to environment issues (despite `requirements.txt` existing) is a critical blocker for reliable deployment.

## 3. Score
**Grade**: 7/10
**Justification**: The intent and artifacts (requirements, setup scripts) are there, but the execution (CI failures, duplication) drags it down.

## 4. Recommendations
1.  **Unify Requirements**: Consolidate into a single `pyproject.toml` or a master `requirements.txt` that references others.
2.  **Fix CI Env**: Debug the CI runner to ensure the virtual environment is actually activated and `PYTHONPATH` is set correctly before tests run.
3.  **Deduplicate Setup**: Refactor `setup_dev.py` to use a shared utility module.
