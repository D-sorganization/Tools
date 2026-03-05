# Assessment O: Maintainability

## Executive Summary
This assessment evaluates the long-term viability of the codebase, focusing on technical debt, duplicate code (DRY violations), and the complexity of individual modules (God classes).
Maintainability is currently the highest non-security risk in the repository. The project has accumulated severe technical debt, quantified by 761 `TODO` markers, 289 `FIXME` markers, and over 449 duplicate code blocks (DRY violations) specifically centered around `_bootstrap.py` logic. Furthermore, UI code is highly procedurally coupled, with 24 distinct "God Functions" (methods exceeding 50 lines) responsible for generating PyQt interfaces.

## Scorecard
- **Grade: 5.0/10**

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| O-001 | Critical | DRY Violation | `_bootstrap.py` and across all tools | 449 instances of identical path logic | Copy-pasting boilerplate | Create a centralized `src/shared/bootstrap.py` | M |
| O-002 | Major | Tech Debt | Global Codebase | 761 TODOs / 289 FIXMEs | Leaving inline notes instead of tickets | Triage comments into GitHub Issues | H |
| O-003 | Major | Orthogonality | `src/function_generator/.../main_window.py` | `_init_ui` is 65+ lines long | Procedural UI construction | Extract to declarative UI classes | L |
| O-004 | Medium | Dead Code | `Launcher.py`, `tools_launcher.py` | Confusion over entry points | Incomplete deprecation | Delete legacy launchers in favor of `UnifiedToolsLauncher.py` | S |

## Refactoring Plan
- **Short Term**: Resolve O-004 immediately. The presence of `Launcher.py` and `tools_launcher.py` directly competes with the `UnifiedToolsLauncher.py`, creating fragmented mental models for maintainers. Delete the legacy files.
- **Medium Term**: Resolve O-001. The fact that `_bootstrap.py` logic is copied into nearly every script means that changing the import resolution strategy requires touching hundreds of files. Centralize this immediately.
- **Long Term**: Conduct a dedicated "Tech Debt Sprint" to process the 1,000+ inline `TODO/FIXME` markers. Group them by module, create GitHub issues for valid feature requests, and simply delete markers that represent stale or abandoned ideas (O-002).
