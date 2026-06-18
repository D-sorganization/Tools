# Assessment: Pragmatic Programmer Review

## Craftsmanship Scorecard

| Principle     | Score (0-10) | Notes |
| ------------- | ------------ | ----- |
| DRY           | 4            | Heavy duplication across UI files and launchers. |
| Orthogonality | 3            | 35 "God functions" (e.g., `_build_ui`) highly couple UI to logic. |
| Reversibility | 2            | 11 Hardcoded API keys in tests make rotation impossible. |
| Documentation | 6            | OK overall, but critical tool documentation is missing. |
| **Overall**   | **3.75**     | Severe craftsmanship issues requiring immediate attention. |

## Key Findings

### 1. DRY Violations
The automated scan highlighted significant duplication in UI setup methods and across the dual launcher system (`UnifiedToolsLauncher.py` vs. `tools_launcher.py`). The legacy launcher contains identical business logic to the modern one.

### 2. Orthogonality & Coupling
There are 35 distinct functions flagged as "God Functions" (exceeding 50 lines), almost entirely consisting of `_setup_ui` or `_build_ui` in PyQt6 files (e.g., `lower_body_model/launch_pyqt6.py`, `psa_package/psa_gui.py`). This tightly couples layout, styling, and widget instantiation, making the code incredibly hard to test or modify.

### 3. "Broken Windows" Theory
- **Hardcoded Secrets:** 11 hardcoded API keys exist in the `tests/shared/python/ai` directory. This is a massive broken window that invites further security decay.
- **Task Markers:** The codebase is littered with 123 `TRACKED_TASK`, `TODO`, `HACK`, and `XXX` comments.
- **Exception Handling:** Broad `except Exception:` clauses are used to swallow errors in UI code.

## Recommendations

1. **Purge Secrets**: Use environment variables (`os.getenv`) and `.env` files for the 11 hardcoded API keys immediately.
2. **Refactor God Functions**: Break down the 35 `_build_ui` and `_setup_ui` functions using the Builder pattern or by subclassing smaller widget components.
3. **Deprecate Legacy**: Delete `tools_launcher.py` to resolve the most glaring DRY violation.

## Conclusion

While functional, the codebase suffers from severe "prototype rot." The immediate focus must be on security (Reversibility/Secrets) and decoupling the UI (Orthogonality) to prevent the codebase from becoming unmaintainable.
