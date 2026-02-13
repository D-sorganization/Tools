# Code Quality Assessment — Tools Repository

**Assessment Date:** 2026-02-10
**Assessor:** Antigravity (Automated + Manual Review)
**Repository:** Tools (ud-tools)
**Commit Hash:** main @ 2026-02-10

---

## Executive Summary

| Overall Grade | Score (0-10) | Trend |
| ------------- | ------------ | ----- |
| **Overall**   | 4.8          | ➡️    |

**Key Findings:** The Tools repository contains 763 Python files and serves as the organization's central utility collection. Critical issues include **71 sys.path hacks**, **188 print statements in non-test code**, and several monolithic files exceeding 1000+ lines (Data_Processor_r0.py at 8,994 lines is the worst offender). Test coverage is weak at 40 test files for 763 source files. The repository has made progress on DRY consolidation since Jan 2026 but still has major structural debt.

---

## 1. DRY — Don't Repeat Yourself

**Score:** 4.0 / 10.0

| Metric                                                     | Count      | Severity |
| ---------------------------------------------------------- | ---------- | -------- |
| Duplicated `gui_registration.py` pattern                   | ~20+ tools | 🔴       |
| Duplicated `launch_pyqt6.py` / `launch_web.py` boilerplate | ~20+ tools | 🔴       |
| Duplicated `__init__.py` patterns                          | extensive  | 🟡       |
| Cross-module duplication in data processing                | 3+ modules | 🔴       |

**Findings:**

- Every tool in `src/` has near-identical `gui_registration.py`, `launch_pyqt6.py`, and `launch_web.py` files — massive duplication.
- `Data_Processor_r0.py` (8,994 lines) and `Data_Processor_Integrated.py` (2,708 lines) contain heavily overlapping logic.
- The shared utilities in `src/shared/python/` are good but underutilized.

**Remediation:**

- [ ] Create a base `ToolLauncher` class to eliminate per-tool launch boilerplate
- [ ] Merge or deprecate `Data_Processor_r0.py` vs `Data_Processor_Integrated.py`
- [ ] Template-ize `gui_registration.py` into a shared registry factory

---

## 2. Design by Contract (DbC)

**Score:** 3.5 / 10.0

| Metric                               | Count   | Severity |
| ------------------------------------ | ------- | -------- |
| Functions with precondition checks   | ~15%    | 🔴       |
| Functions with postcondition asserts | ~2%     | 🔴       |
| Uses of `assert` for invariants      | minimal | 🟡       |
| Input validation at API boundaries   | ~30%    | 🟡       |

**Findings:**

- Most calculator/tool functions accept inputs without validation.
- The `Data_Processor` module performs some input validation but inconsistently.
- No systematic use of DbC patterns across the codebase.

**Remediation:**

- [ ] Add input validation to all public API functions in `src/shared/python/`
- [ ] Add precondition checks to calculator entry points
- [ ] Document contracts in docstrings

---

## 3. Test-Driven Development (TDD)

**Score:** 3.0 / 10.0

| Metric                   | Value           | Severity |
| ------------------------ | --------------- | -------- |
| Test coverage %          | ~5% (estimated) | 🔴       |
| Test-to-code ratio       | 40:763 (1:19)   | 🔴       |
| Tests for edge cases     | minimal         | 🔴       |
| Mocking/stubbing quality | basic           | 🟡       |
| Tests run in CI          | ✅              | 🟢       |

**Findings:**

- Only 40 test files for 763 source files — severe coverage gap.
- Most tests appear to be GUI integration tests rather than unit tests.
- Core calculation modules lack dedicated test suites.

**Remediation:**

- [ ] Create unit tests for all `src/shared/python/` modules (priority)
- [ ] Add tests for each calculator's core logic
- [ ] Target 60% coverage on shared utilities

---

## 4. Orthogonality

**Score:** 4.5 / 10.0

| Metric                                           | Count               | Severity |
| ------------------------------------------------ | ------------------- | -------- |
| Tightly coupled modules                          | 5+                  | 🟡       |
| Circular imports                                 | unknown — potential | 🟡       |
| God classes (>500 lines)                         | 8+ files            | 🔴       |
| Cross-cutting concerns mixed with business logic | significant         | 🟡       |

**Findings:**

- UI logic is mixed with calculation logic in several tools (e.g., `electrode_advisor/ui/pyqt6/main_window.py` at 4,386 lines does both UI and calculations).
- The `model_generation` module couples tightly to specific tool implementations.
- Shared utilities are well-separated, which is a strength.

**Remediation:**

- [ ] Extract calculation logic from UI files into separate service modules
- [ ] Create clear interfaces between UI and business logic layers

---

## 5. Monolithic Files

**Score:** 2.0 / 10.0

| File                                        | Lines | Functions | Recommendation                  |
| ------------------------------------------- | ----- | --------- | ------------------------------- |
| `Data_Processor_r0.py`                      | 8,994 | many      | Split into 10+ modules          |
| `electrode_advisor/ui/pyqt6/main_window.py` | 4,386 | many      | Split UI from logic             |
| `folder_tools/Folders_Tool_r0.py`           | 3,291 | many      | Split into modules              |
| `Data_Processor_Integrated.py`              | 2,708 | many      | Merge or deprecate              |
| `folder_packer_pro.py`                      | 1,911 | many      | Split by concern                |
| `signal_toolkit/widget.py`                  | 1,794 | many      | Split UI from signal processing |
| `frankenstein_editor.py`                    | 1,399 | many      | Split editor components         |
| `neural_network.py`                         | 1,348 | many      | Split layers/training/inference |

**Threshold:** Files >400 lines are flagged. Files >800 lines are critical.

**Remediation:**

- [ ] Priority 1: Split `Data_Processor_r0.py` (8,994 lines — CRITICAL)
- [ ] Priority 2: Split `electrode_advisor main_window` (4,386 lines)
- [ ] Priority 3: Split `Folders_Tool_r0.py` (3,291 lines)

---

## 6. Reversibility

**Score:** 4.0 / 10.0

| Metric                      | Status            | Severity |
| --------------------------- | ----------------- | -------- |
| Hard-coded file paths       | 71 sys.path hacks | 🔴       |
| Hard-coded DB/API endpoints | minimal           | 🟢       |
| Framework lock-in (PyQt6)   | moderate          | 🟡       |
| Configuration externalized  | partial           | 🟡       |
| Dependency injection used   | minimal           | 🟡       |

**Findings:**

- 71 files contain `sys.path` manipulation — the single biggest reversibility issue.
- PyQt6 dependency is reasonable but should be abstracted behind interfaces.

**Remediation:**

- [ ] Eliminate all 71 `sys.path` hacks via proper package installation
- [ ] Externalize configuration to `.env` or config files

---

## 7. Reusability

**Score:** 5.5 / 10.0

| Metric                                | Count              | Severity |
| ------------------------------------- | ------------------ | -------- |
| Utility functions usable cross-repo   | good (src/shared/) | 🟢       |
| Functions with hard-coded assumptions | moderate           | 🟡       |
| Generic vs. project-specific ratio    | ~40% generic       | 🟡       |
| Shared library usage (ud-tools)       | established        | 🟢       |

**Findings:**

- The `src/shared/python/` directory is a strength — well-organized shared utilities.
- Individual tools embed assumptions that limit reusability.

**Remediation:**

- [ ] Extract more generic utilities from tool-specific code to shared/

---

## 8. Parity / Maintenance

**Score:** 5.0 / 10.0

| Metric                        | Status                       | Severity |
| ----------------------------- | ---------------------------- | -------- |
| AGENTS.md up to date          | ❌ (missing design criteria) | 🟡       |
| CI/CD pipeline passing        | ✅                           | 🟢       |
| Dependencies pinned & current | partial                      | 🟡       |
| Stale branches                | needs audit                  | 🟡       |
| Open issues triaged           | needs audit                  | 🟡       |
| README accurate               | ✅                           | 🟢       |

---

## 9. Changeability

**Score:** 4.5 / 10.0

| Metric                          | Status              | Severity |
| ------------------------------- | ------------------- | -------- |
| Single Responsibility adherence | weak in large files | 🔴       |
| Change impact isolation         | moderate            | 🟡       |
| Feature toggle capability       | none                | 🟡       |
| Config-driven behavior          | partial             | 🟡       |

---

## 10. Function Length & Signature Quality

**Score:** 4.0 / 10.0

| Metric                          | Count                   | Threshold | Severity |
| ------------------------------- | ----------------------- | --------- | -------- |
| Functions >50 lines             | significant (est. 100+) | 0         | 🔴       |
| Functions >30 lines             | significant             | ≤5%       | 🔴       |
| Functions with >4 parameters    | moderate                | 0         | 🟡       |
| Average function length (lines) | ~35 (estimated)         | ≤20       | 🟡       |

---

## 11. Law of Demeter

**Score:** 5.5 / 10.0

| Metric                                 | Count    | Severity |
| -------------------------------------- | -------- | -------- |
| Chained attribute access (>2 dots)     | moderate | 🟡       |
| Functions reaching into nested objects | some     | 🟡       |
| Wrapper/delegate methods missing       | some     | 🟡       |

---

## 12. God Functions

**Score:** 3.0 / 10.0

| Function                     | File               | Lines       | Responsibilities               | Severity |
| ---------------------------- | ------------------ | ----------- | ------------------------------ | -------- |
| Many in Data_Processor_r0.py | data_processor/    | 50-200+     | data loading + processing + UI | 🔴       |
| Main window setup            | electrode_advisor/ | 4,386 total | UI + calculation + validation  | 🔴       |
| Folder tool main             | folder_tools/      | 3,291 total | scanning + processing + UI     | 🔴       |

---

## 13. Deprecated / Outdated Code

**Score:** 3.5 / 10.0

| Metric                         | Count       | Severity |
| ------------------------------ | ----------- | -------- |
| `# TODO` / `# FIXME` markers   | 3 files     | 🟢       |
| `NotImplementedError` stubs    | needs audit | 🟡       |
| Dead code (unreachable/unused) | significant | 🔴       |
| Deprecated library usage       | needs audit | 🟡       |
| Legacy compatibility shims     | present     | 🟡       |
| `sys.path` hacks               | 71 files    | 🔴       |

---

## 14. Function Name Quality

**Score:** 6.0 / 10.0

| Metric                                  | Count                        | Severity |
| --------------------------------------- | ---------------------------- | -------- |
| Single-letter variable names (non-loop) | moderate                     | 🟡       |
| Ambiguous function names                | some                         | 🟡       |
| Inconsistent naming convention          | some PascalCase in filenames | 🟡       |
| Abbreviation overuse                    | minimal                      | 🟢       |

**Findings:**

- Files like `Data_Processor_r0.py` and `Folders_Tool_r0.py` use PascalCase filenames inconsistent with snake_case standard.

---

## 15. No Magic Numbers

**Score:** 6.0 / 10.0

| Metric                                | Count                               | Severity |
| ------------------------------------- | ----------------------------------- | -------- |
| Unexplained numeric literals in logic | moderate                            | 🟡       |
| Constants extracted to module-level   | good in data_processor/constants.py | 🟢       |

**Findings:**

- `constants.py` exists and is used well in data_processor.
- Other tools use inline magic numbers.

---

## 16. Project Structure & Organization

**Score:** 5.5 / 10.0

| Metric                            | Status          | Severity |
| --------------------------------- | --------------- | -------- |
| Standard `src/` layout            | ✅              | 🟢       |
| `tests/` directory present        | ✅              | 🟢       |
| `docs/` directory organized       | ✅              | 🟢       |
| Root clutter (non-standard files) | minimal         | 🟢       |
| Consistent module naming          | ❌ (mixed case) | 🟡       |

---

## 17. Cleanup of Outdated Documents & Code

**Score:** 4.5 / 10.0

| Metric                    | Count              | Severity |
| ------------------------- | ------------------ | -------- |
| Commented-out code blocks | moderate           | 🟡       |
| Obsolete scripts/tools    | `_r0` suffix files | 🟡       |

**Findings:**

- Files with `_r0` suffix suggest legacy versions that may be superseded.

---

## 18. Comment Quality

**Score:** 5.0 / 10.0

| Metric                                  | Count         | Severity |
| --------------------------------------- | ------------- | -------- |
| Functions without docstrings            | significant   | 🟡       |
| Missing "why" comments on complex logic | common        | 🟡       |
| print() used instead of logging         | 188 instances | 🔴       |

---

## 19. Calculation Optimization (Numerical Code)

**Score:** 5.0 / 10.0 _(applicable to data processing and scientific modeling modules)_

### 19a. Vectorization

- Data processing module makes some use of pandas vectorized ops but could improve.

### 19b–d. Other

- Neural network module could benefit from batch processing optimization.
- Signal toolkit has room for NumPy vectorization improvements.

---

## Summary Scorecard

| #       | Criterion                | Score      | Priority |
| ------- | ------------------------ | ---------- | -------- |
| 1       | DRY                      | 4.0/10     | 🔴       |
| 2       | Design by Contract       | 3.5/10     | 🔴       |
| 3       | TDD                      | 3.0/10     | 🔴       |
| 4       | Orthogonality            | 4.5/10     | 🟡       |
| 5       | Monolithic Files         | 2.0/10     | 🔴       |
| 6       | Reversibility            | 4.0/10     | 🔴       |
| 7       | Reusability              | 5.5/10     | 🟡       |
| 8       | Parity / Maintenance     | 5.0/10     | 🟡       |
| 9       | Changeability            | 4.5/10     | 🟡       |
| 10      | Function Length          | 4.0/10     | 🔴       |
| 11      | Law of Demeter           | 5.5/10     | 🟡       |
| 12      | God Functions            | 3.0/10     | 🔴       |
| 13      | Deprecated Code          | 3.5/10     | 🔴       |
| 14      | Name Quality             | 6.0/10     | 🟡       |
| 15      | Magic Numbers            | 6.0/10     | 🟡       |
| 16      | Project Structure        | 5.5/10     | 🟡       |
| 17      | Cleanup                  | 4.5/10     | 🟡       |
| 18      | Comment Quality          | 5.0/10     | 🟡       |
| 19      | Calculation Optimization | 5.0/10     | 🟡       |
| **AVG** | **Overall**              | **4.4/10** |          |

---

## Improvement Roadmap

### Phase 1 — Critical (This Sprint)

- [ ] Eliminate all 71 `sys.path` hacks via proper package installation
- [ ] Split `Data_Processor_r0.py` (8,994 lines) into focused modules
- [ ] Convert 188 `print()` calls to `logging`

### Phase 2 — High Priority (Next Sprint)

- [ ] DRY: Create base `ToolLauncher` to eliminate launch boilerplate duplication
- [ ] Split `electrode_advisor/main_window.py` (4,386 lines)
- [ ] Add unit tests for `src/shared/python/` (target 60% coverage)

### Phase 3 — Medium Priority (Backlog)

- [ ] Add DbC validation to all calculator entry points
- [ ] Rename PascalCase files to snake_case
- [ ] Add docstrings to all public functions

### Phase 4 — Polish (Future)

- [ ] Law of Demeter audit and fixes
- [ ] Magic number extraction
- [ ] Performance profiling of data processing pipeline

---

_Generated by the Organizational Code Quality Assessment Framework v2.0_
_Template: `Repository_Management/docs/templates/code_quality_assessment_template.md`_
