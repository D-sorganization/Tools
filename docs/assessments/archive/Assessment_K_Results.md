# Assessment K Results: Reproducibility & Provenance

## Executive Summary

-   **Environment**: Python environment reproducible via `requirements.txt`. Node via `package-lock.json`.
-   **Lock Files**: Python lacks a lock file (e.g., `poetry.lock`), meaning builds might drift.
-   **Versioning**: `CHANGELOG.md` tracks changes.
-   **Data**: No large datasets in repo (good), but where does test data come from?
-   **Determinism**: `solar_system_model` simulation likely deterministic if seeded.

## Top 10 Reproducibility Risks

1.  **Python Drift (Severity: Medium)**: `requirements.txt` without pinning transitive dependencies can lead to "it works on my machine" issues.
2.  **MATLAB Version (Severity: Medium)**: MATLAB scripts often depend on specific toolboxes/versions.
3.  **OS Specifics (Severity: Low)**: Paths and shortcuts are Windows-centric.
4.  **Test Data (Severity: Low)**: Is test data generated or static?
5.  **Randomness (Severity: Low)**: RRT path planner is randomized; does it have a seed?
6.  **Documentation (Severity: Low)**: Setup guide needs to be strict.
7.  **Binaries (Severity: Low)**: No compiled binaries committed (good).
8.  **Git LFS (Severity: Low)**: Mentioned in README, need to ensure it's used if needed.
9.  **Time Handling (Severity: Low)**: Timezones in data processor?
10. **Hardware (Severity: Low)**: OpenGL for solar system might behave differently on different GPUs.

## Scorecard

| Category             | Score | Evidence & Remediation                                    |
| -------------------- | ----- | --------------------------------------------------------- |
| Dependency Pinning   | 7/10  | `requirements.txt` used, likely unpinned transitive deps. |
| Build Reproducibility| 7/10  | No binary build artifacts.                                |
| Experiment Tracking  | N/A   | Not an ML repo.                                           |
| Version Control      | 10/10 | Git used effectively.                                     |
| Data Provenance      | 8/10  | Clean separation of code and data.                        |

## Findings Table

| ID    | Severity | Category        | Location | Symptom | Root Cause | Fix | Effort |
| ----- | -------- | --------------- | -------- | ------- | ---------- | --- | ------ |
| K-001 | Medium   | Reproducibility | Root     | No lock file | PIP usage | Use `pip-tools` or `poetry` | M |

## Refactoring Plan

**48 Hours**:
-   None.

**2 Weeks**:
-   Generate `requirements.lock`.

**6 Weeks**:
-   Move to Poetry for Python dependency management.
