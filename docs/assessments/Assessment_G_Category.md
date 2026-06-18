# Assessment G Results: Dependencies

## Executive Summary
- The repository successfully utilizes `uv` for modern Python package management, reducing resolution times.
- However, multiple older GUI applications lack explicit dependency mapping.
- There is a heavy reliance on outdated or unpinned transitive libraries in legacy packages.

## Top 10 Risks
1. [Major] Web apps (`function_generator/web`, `data_processing/data_processor/web`) have unmanaged node module lockfiles.
2. [Major] The shared python core relies on multiple heavy data-science libraries that inflate the Docker image.
3. [Minor] Rust core dependencies are not version-pinned in `Cargo.toml`.

## Scorecard
| Category | Description | Weight | Score | Notes |
|----------|-------------|--------|-------|-------|
| Lockfiles | Are lockfiles used? | 2x | 6/10 | Node apps frequently suffer from detached lockfiles. |
| Vulnerabilities | Are dependencies secure? | 2x | 7/10 | General safety is okay, but audits are not automated. |

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| G-001 | Major | Node | Web Apps | Missing lockfile context | Pathspec errors | Restore lockfiles globally after install | S |

## Refactoring Plan
**48 Hours**:
- Implement CI checks to verify lockfile integrity across all Node.js projects.
