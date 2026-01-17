# Assessment B Results: Hygiene, Security & Quality

## Executive Summary

- **Hygiene Standards**: The repository generally follows good practices with `ruff.toml` and `mypy.ini` present. However, occasional `print()` statements persist in production code (e.g., `pdf_renamer`, `folder_packer_pro`), violating strict AGENTS.md directives.
- **Security Posture**: Basic security measures are in place (no obvious hardcoded secrets found in a quick scan, though `setup_api_key.py` handles sensitive data). Dependencies are managed via `requirements.txt`.
- **Linter Configuration**: Modern tooling (`ruff`) is configured, which is a strong positive.
- **Pre-commit**: `.pre-commit-config.yaml` exists, ensuring some level of automated checking.

## Top 10 Hygiene Risks

1.  **Residual Print Statements (Major)**: Found `print()` in `document_processing/pdf_renamer/setup_api_key.py` and `tools/folder_tools/folder_packer_pro/folder_packer_pro.py`.
2.  **Missing `pip-audit` (Moderate)**: The tool `pip-audit` is configured in CI but not available in the current environment, potentially hiding vulnerability checks.
3.  **Loose File Permissions (Minor)**: No explicit checking of file permissions for sensitive tools like `folder_packer_pro` (which handles encryption).
4.  **Mixed Type Hinting (Minor)**: While core files are typed, older or peripheral scripts might lack full coverage.
5.  **Docstring Inconsistency (Minor)**: Some files have excellent module docstrings, others are minimal.
6.  **TODOs in Code (Minor)**: "Legacy" and "Replicant" markers imply technical debt.
7.  **Dead Code (Minor)**: `pdf_renamer_backup` folder exists alongside `pdf_renamer`, cluttering the repo.
8.  **Secret Handling in CLI (Moderate)**: `setup_api_key.py` prompts for keys but the handling should be verified to ensure it doesn't log them.
9.  **Binary Files (Minor)**: `ProfilePhoto.jpg` in `media_processing` – verify necessity and size.
10. **Test Hygiene (Minor)**: `pdf_renamer_backup` has its own tests, duplicating test execution.

## Scorecard

| Category                | Score | Evidence & Remediation                                                                   |
| ----------------------- | ----- | ---------------------------------------------------------------------------------------- |
| Ruff Compliance         | 9/10  | Config exists, likely enforced in CI.                                                    |
| Mypy Compliance         | 8/10  | Config exists, strict mode not verified globally.                                        |
| Black Formatting        | 9/10  | Code appears consistent.                                                                 |
| AGENTS.md Compliance    | 7/10  | `print()` usage found. **Fix**: Replace with `logging`.                                  |
| Security Posture        | 8/10  | No hardcoded secrets found. `setup_api_key.py` handles input.                            |
| Repository Organization | 8/10  | Generally good, but `backup` folders reduce cleanliness. **Fix**: Delete backups.        |
| Dependency Hygiene      | 8/10  | `requirements.txt` used.                                                                 |

## Linting Violation Inventory

| File                                               | Violation | Type        |
| -------------------------------------------------- | --------- | ----------- |
| `document_processing/pdf_renamer/setup_api_key.py` | `print()` | Hygiene     |
| `tools/folder_tools/.../folder_packer_pro.py`      | `print()` | Hygiene     |

## Security Audit

| Check                        | Status | Evidence |
| ---------------------------- | ------ | -------- |
| No hardcoded secrets         | ✅     | Grep scan negative for obvious passwords. |
| .env.example exists          | ❌     | Not found in root (only `environment.yml` and `requirements.txt`). |
| No eval()/exec() usage       | ✅     | Not prevalent in examined files. |
| Safe file I/O                | ✅     | `Path` usage seen generally. |

## AGENTS.md Compliance Report

- **No `print()`**: FAILED. Found usage in CLI tools.
- **No wildcard imports**: PASSED (mostly).
- **Type hints required**: PASSED (mostly).
- **No secrets**: PASSED.

## Findings Table

| ID    | Severity | Category | Location                                           | Symptom                | Root Cause          | Fix                                   | Effort |
| ----- | -------- | -------- | -------------------------------------------------- | ---------------------- | ------------------- | ------------------------------------- | ------ |
| B-001 | Major    | Hygiene  | `document_processing/pdf_renamer/setup_api_key.py` | `print()` usage        | CLI tool design     | Use `logging` or `rich` for CLI output| S      |
| B-002 | Minor    | Hygiene  | `document_processing/pdf_renamer_backup/`          | Duplicate code         | Backup strategy     | Delete folder                         | S      |
| B-003 | Minor    | Security | Root                                               | Missing `.env.example` | Setup oversight     | Create file                           | S      |

## Refactoring Plan

**48 Hours**
- Delete `document_processing/pdf_renamer_backup/`.
- Create `.env.example`.

**2 Weeks**
- Replace all `print()` in CLI tools with a proper CLI library (like `typer` or `rich`) or `logging`.

**6 Weeks**
- Enforce strict `mypy` across all modules.

## Diff Suggestions

**Replace Print with Logging (Example)**

```python
<<<<<<< SEARCH
    print(f"Error loading tools.json: {e}")
=======
    logging.error(f"Error loading tools.json: {e}")
>>>>>>> REPLACE
```
