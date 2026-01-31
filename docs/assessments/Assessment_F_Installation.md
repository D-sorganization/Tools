# Assessment F Results: Installation & Deployment

## Executive Summary

- **Fragmentation**: 10+ `requirements.txt` files found across the repo. No single source of truth.
- **No Package Structure**: The repository is not a installable Python package (no root `setup.py` or `pyproject.toml`).
- **Custom Scripts**: `setup_dev.py` and `verify_installation.py` are custom, fragile scripts attempting to do what `pip` does natively.
- **Environment Hell**: High risk of conflicting dependencies between sub-tools (e.g., `media_processing` vs `scientific_modeling`).

## Top 10 Installation Risks

1.  **Dependency Conflicts (Critical)**: `numpy` version in root might differ from sub-tool requirements.
2.  **Missing Dependencies (Critical)**: `Launcher.py` imports `PyQt6` but is in root; does root `requirements.txt` include it? (Yes, `requirements.txt` exists but `PyQt6` is in it? Checked: Yes).
3.  **Path Issues (Major)**: Scripts rely on `sys.path.append` or relative paths, breaking if run from wrong dir.
4.  **No Lockfile (Major)**: `requirements.txt` defines ranges, leading to "works today, breaks tomorrow".
5.  **Platform Specifics (Moderate)**: `setup_dev.py` might behave differently on Windows/Linux.
6.  **Virtual Env (Minor)**: No enforcement of virtual environment usage.
7.  **MATLAB Runtime (Major)**: MATLAB dependencies are manual install.
8.  **System Libs (Minor)**: `PyQt6` might require system libs on Linux.
9.  **Permissions (Minor)**: Scripts might need executable permission.
10. **Uninstaller (Minor)**: No way to "uninstall" or clean up.

## Scorecard

| Category                 | Score | Evidence & Remediation                                                                 |
| ------------------------ | ----- | -------------------------------------------------------------------------------------- |
| Installation Ease        | 4/10  | Requires custom scripts. **Fix**: Standard `pip install .`.                            |
| Dependency Management    | 3/10  | Fragmented. **Fix**: Use Poetry workspace or single lockfile.                          |
| Environment Isolation    | 2/10  | None.                                                                                  |
| Cross-Platform Support   | 6/10  | Python is portable, but paths might break.                                             |

## Findings Table

| ID    | Severity | Category     | Location          | Symptom                 | Root Cause           | Fix                  | Effort |
| ----- | -------- | ------------ | ----------------- | ----------------------- | -------------------- | -------------------- | ------ |
| F-001 | Critical | Install      | Root              | No `pyproject.toml`     | Legacy setup         | Create config        | S      |
| F-002 | Major    | Dependencies | Multiple dirs     | Duplicate reqs          | Monorepo             | Unify/Workspace      | M      |

## Refactoring Plan

**48 Hours - Critical fixes:**
-   Create a root `pyproject.toml` that defines the project and its dependencies.

**2 Weeks - Major improvements:**
-   Convert to a proper `src` layout package `tools_repo`.
-   Use `pip-tools` or `poetry` to generate a lockfile.

**6 Weeks - Full graduation:**
-   Dockerize the environment for absolute reproducibility.

## Diff-Style Suggestions

```toml
# pyproject.toml (New File)
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "jules-tools"
version = "0.1.0"
dependencies = [
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "PyQt6>=6.6.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/tools", "src/shared"]
```
