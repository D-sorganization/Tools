# Assessment F: Tools Repository Deployment & Installation Review

## 1. Executive Summary

- The repository boasts flawless pipeline automation via GitHub Actions (e.g., `ci-standard.yml`, `Jules-Sentinel.yml`), leading to a high score in CI/CD integration.
- Tool installation is relatively straightforward for local scripts using standard Python virtual environments, but web applications (Next.js) have deployment gaps.
- Dependency installation was historically flaky but recent patches to CI workflows removed `|| true` masking, enforcing strict requirements handling.
- **Top Risk**: A persistent `ModuleNotFoundError` for packages like `pandas` and `numpy` when running `PYTHONPATH` tests directly from within `src/data_processing` indicates inconsistent local developer environment configurations compared to CI.

## 2. Scorecard (0-10)

| Category                     | Description                                   | Score |
| ---------------------------- | --------------------------------------------- | ----- |
| Package Management           | Are dependencies clearly defined?             | 9     |
| Cross-Platform Support       | Does it run on Windows/Mac/Linux?             | 8     |
| Virtual Environment Setup    | Are setup scripts reliable?                   | 8     |
| CI/CD Pipeline Build         | Do builds pass reliably?                      | 10    |
| Containerization             | Are Dockerfiles present/working?              | 4     |

*Evidence for Containerization (4)*: The web apps lack robust Dockerfiles for production deployment, relying heavily on local `npm run dev`.

## 3. Deployment Gap Table

| ID    | Severity | Domain/File | Description | Fix Recommendation | Effort |
| ----- | -------- | ----------- | ----------- | ------------------ | ------ |
| F-001 | Major    | Developer Env | `PYTHONPATH` issues | Standardize execution via `tox` or explicit shell scripts | S |
| F-002 | Major    | `media_processing` | Next.js backend deploy | Create a `Dockerfile` for production deployment | M |
| F-003 | Minor    | `Folder Packer` | Build Executables | Ensure PyInstaller configurations handle resource paths correctly | M |
| F-004 | Nit      | Global | Legacy `requirements.txt` | Consider modernizing with `poetry` or `uv` | L |

## 4. Remediation Plan

**Immediate (48 Hours):**
- Fix the `PYTHONPATH` discrepancy causing local `pytest` executions to fail on `pandas/numpy` missing imports while CI passes.

**Short-Term (2 Weeks):**
- Provide a `Dockerfile` and a `docker-compose.yml` for the `media_processing` video processor to streamline deployment.
- Clean up any legacy installation scripts (`setup_dev.py`) that might contain duplicate code (flagged by Pragmatic Programmer).

**Long-Term (6 Weeks):**
- Evaluate moving from standard `pip` + `requirements.txt` to a more robust lockfile system like `poetry` to guarantee exact deterministic installations across the monorepo.
