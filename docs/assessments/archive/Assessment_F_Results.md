# Assessment F Results: Installation & Deployment

## Executive Summary

-   **Dependencies**: Dependencies are managed via `requirements.txt` (Python) and `package.json` (Node.js).
-   **Setup Scripts**: `scripts/setup_precommit.sh` aids developer onboarding.
-   **Environment**: The requirement for multiple runtimes (Python 3.11+, Node, MATLAB) complicates the "one-click" setup.
-   **Deployment**: No Dockerfiles found for containerized deployment of web apps.
-   **CI/CD**: `ci-standard.yml` suggests an active CI pipeline.

## Top 10 Installation Risks

1.  **Environment Sync (Severity: Medium)**: Root `requirements.txt` vs `python/requirements.txt` could diverge.
2.  **MATLAB Requirement (Severity: High)**: Proprietary dependency is a barrier.
3.  **Path Issues (Severity: Medium)**: Scripts might rely on execution from root vs subfolder.
4.  **Node Version (Severity: Low)**: `unit_converter` needs specific Node version?
5.  **Virtual Env (Severity: Low)**: Instructions rely on user creating venv.
6.  **Pre-commit (Severity: Low)**: Must be installed manually via script.
7.  **System Deps (Severity: Low)**: Some Python packages (e.g., audio) might need system libs.
8.  **Windows/Linux (Severity: Low)**: PowerShell scripts are Windows-only.
9.  **Updates (Severity: Low)**: How to update all tools? `git pull` + `pip install`?
10. **Deployment (Severity: Medium)**: No clear path to deploy `calculator` to a server.

## Scorecard

| Category                  | Score | Evidence & Remediation                                        |
| ------------------------- | ----- | ------------------------------------------------------------- |
| Package Management        | 8/10  | Standard files used.                                          |
| Cross-Platform            | 7/10  | Windows focused (PS1 scripts), but Python is cross-platform.  |
| CI/CD Integration         | 9/10  | Workflows exist.                                              |
| Containerization          | 0/10  | No Dockerfiles found.                                         |
| Documentation             | 8/10  | Installation steps in README.                                 |

## Findings Table

| ID    | Severity | Category     | Location | Symptom | Root Cause | Fix | Effort |
| ----- | -------- | ------------ | -------- | ------- | ---------- | --- | ------ |
| F-001 | Medium   | Installation | Root     | No Docker | N/A        | Add Dockerfile | M |

## Refactoring Plan

**48 Hours**:
-   None.

**2 Weeks**:
-   Consolidate `requirements.txt`.

**6 Weeks**:
-   Create a `docker-compose.yml` for the web applications.
