# Assessment G: Dependencies

## Executive Summary
This assessment reviews dependency management, package isolation, and supply chain security practices within the Tools repository.
Dependency management is generally strong. The root `requirements.txt` is well-commented and explicitly pins dependencies like `PyQt6`, `pandas`, `numpy`, and `matplotlib`. Virtual environments are heavily encouraged. However, as a polyglot monorepo, a single global `requirements.txt` forces all tools to share a massive dependency footprint, leading to potential version conflicts and slow installation times for users who only want a specific calculator.

## Scorecard
- **Grade: 8.5/10**

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| G-001 | Major | Architecture | `requirements.txt` | "Dependency Hell" for specific tools | Monolithic requirement file | Split into `tool/requirements.txt` + `shared/` | M |
| G-002 | Medium | Reproducibility | CI Pipeline | Potential environment drift | Relying only on `requirements.txt` without hashes | Implement `pip-compile` / `requirements-lock.txt` | S |
| G-003 | Medium | Web Dep | `src/media_processing/video_processor/apps/web/` | Stale JS dependencies | Lack of `dependabot` | Configure `dependabot` for npm/pnpm | S |
| G-004 | Minor | Testing | `pytest.ini` | Missing `pytest-cov` | Dependency removed but config remained | Add `pytest-cov` back to dev requirements | S |

## Refactoring Plan
- **Short Term**: Resolve G-004 by ensuring development dependencies (like `pytest-cov`, `ruff`, `mypy`) are properly separated from production dependencies. Set up `dependabot` (G-003).
- **Medium Term**: Implement `pip-tools` to generate strict lockfiles (`requirements.txt` -> `requirements.lock`), resolving G-002.
- **Long Term**: Address G-001 by implementing a workspace-based dependency manager (e.g., `uv workspaces` or `poetry`) to allow tools to have isolated, minimal dependencies rather than the monolithic approach.
