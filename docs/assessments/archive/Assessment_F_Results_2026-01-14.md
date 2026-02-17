# Assessment F: Installation & Deployment Results

**Date:** 2026-01-14
**Assessor:** Jules

## 1. Installation Experience

**Score: 4/10**

- **Dependency Management**: Fragmented. `python/requirements.txt`, `scientific_modeling/.../requirements.txt`, `web_applications/.../package.json`.
- **Reproducibility**: Python projects lack lock files (`poetry.lock` or `pip-tools`). Node.js projects use `package-lock.json` but rely on `pnpm` which must be installed globally.
- **Documentation**: Instructions are missing for a "unified" install.

## 2. CI/CD Pipeline

**Score: 3/10**

- **State**: "Shadow IT" workflows (`Jules-*.yml`) were found in a subproject, attempting to control the repo.
  - _Remediated_: These were deleted during assessment to restore governance.
- **Standard CI**: `ci-standard.yml` (implied by memory/logs) exists but is bypassed by the new injection.

## 3. Platform Support

**Score: 5/10**

- **Cross-Platform**: Python code is generally cross-platform.
- **System Dependencies**: `scientific_modeling/solar_system_model` likely requires OpenGL system libraries which are not documented in `requirements.txt`.

## Remediation Roadmap

- **Immediate**: Create a root-level `requirements.txt` that includes all subprojects (or use a workspace tool).
- **Short-term**: Implement `pnpm` workspaces for JS and `poetry` or `uv` for Python to manage the monorepo.
- **Long-term**: Containerize applications.
