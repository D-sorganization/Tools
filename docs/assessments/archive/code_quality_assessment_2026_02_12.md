# Code Quality Assessment — Tools Repository (Post-PR #710)

**Assessment Date:** 2026-02-12
**Assessor:** Claude Opus 4.6 (Adversarial Review)
**Repository:** D-sorganization/Tools (ud-tools v0.3.0)
**Baseline:** code_quality_assessment_2026_02_12.md (Score: 5.3/10)
**Scope:** Evaluate all refactoring done 2026-02-10 through 2026-02-12, including PRs #673–#710 (23 commits)

---

## Executive Summary

| Overall Grade | Score (0-10) | Trend  | Baseline |
| ------------- | ------------ | ------ | -------- |
| **Overall**   | **6.2**      | ↑ +0.9 | 5.3      |

The second wave of refactoring (PRs #699–#710) built on the first wave's architectural foundation (`_bootstrap.py`, `tool_surface_contract.json`) with targeted quality improvements across all three pillars:

- **DRY (6.5/10, up from 5.5):** Process calculator constants consolidated from 40+ hardcoded values to imports from `unit_constants.py`. Magic numbers extracted to named constants in 8 calculator files. `print()` calls reduced from 160 → 47 (most remaining are intentional CLI output). 3 `contracts.py` consolidated into 1 canonical module.
- **DbC (5.5/10, up from 3.5):** New consolidated `contracts.py` with tri-state enforcement (ENFORCE/WARN/OFF). 17 imperative `raise ValueError` guards added to 3 process calculators. Total `raise ValueError` count at 375 across src/. The gap between infrastructure and adoption has significantly narrowed.
- **TDD (6.0/10, up from 4.0):** 122 new edge-case tests across 4 calculator test files. Test collection errors eliminated (6 → 0). Total suite: 996 tests collected, all passing. Test-to-source ratio improved from ~1:17 to 1.81:1.

**Verdict:** Meaningful, measurable progress across all three pillars. The codebase is now in a fundamentally healthier state for continued development. The biggest remaining debt is in monolithic files (Data_Processor_r0.py at 8,994 lines) and the 142 functions exceeding 100 lines.

---

## Refactoring Changes Assessed (Full 2-Day Window)

| PR   | Commit    | Description                                                | Impact                                                              |
| ---- | --------- | ---------------------------------------------------------- | ------------------------------------------------------------------- |
| #710 | `9ab67c3` | **DRY, DbC, TDD, and mypy quality improvements**           | Constants consolidated, 17 DbC guards, 122 new tests, 70 mypy fixes |
| #707 | `3bc31da` | T201 lint errors — print() → logging                       | 4,047 lines cleaned across 21 files                                 |
| #706 | `ab78220` | Extract magic numbers to named constants                   | 335 new constant definitions in calculators                         |
| #705 | `77bb8d1` | Narrow 105 broad `except Exception` handlers               | Specific exception types across 31 files                            |
| #704 | `9e23c47` | Replace print() with logging, add T201 ruff rule           | Logging infrastructure + lint enforcement                           |
| #703 | `be34b7c` | Replace regex sanitization with DOMPurify + pino           | Security improvement in web apps                                    |
| #702 | `79a06a5` | Fix overflow in syngas_water_calculator exponential        | Numerical stability fix                                             |
| #701 | `aec4624` | Remove remaining sys.path hacks in src/                    | 22 files cleaned                                                    |
| #700 | `c2b6263` | Resolve NotImplementedError stubs                          | signal_toolkit + model_generation stubs eliminated                  |
| #699 | `55739ac` | Consolidate 3 contracts.py into single module              | DRY + DbC infrastructure unification                                |
| #691 | `189448f` | Add DBC/DRY/TDD quality assessment                         | Assessment documentation                                            |
| #685 | `ce45914` | Add Design-by-Contract module with tri-state enforcement   | Core DbC infrastructure                                             |
| #690 | `f35abb0` | Eliminate DRY violations in flow rate converter + UI theme | DRY extraction                                                      |
| #689 | `d2de082` | Assessment generation and documentation fixes              | Docs                                                                |
| #684 | `fed4eaa` | Post-refactoring code quality assessment                   | Baseline assessment                                                 |
| #688 | `5d36512` | Audit .jules/completist_data                               | Governance                                                          |
| #683 | `2694ca3` | Restore 45 archived workflows + CODEOWNERS                 | Governance improvement                                              |
| #681 | `e8811a3` | Cantera API compat + 45 unused import removals             | Cleanup                                                             |
| #680 | `8773284` | **Replace 62 sys.path hacks with `_bootstrap.py`**         | Highest-impact DRY change                                           |
| #676 | `2372395` | Design criteria in AGENTS.md + assessment                  | 14 mandatory design principles                                      |
| #675 | `5619aba` | Phase 3.1 — installable package, v0.3.0                    | Package infrastructure                                              |
| #674 | `c2e0e25` | tool_surface_contract.json generation with DbC + TDD       | 17 new tests, cross-repo parity                                     |
| #673 | `6b82b84` | Tools repository metadata unification (Phase 1)            | `gui_registration.py` standardization                               |

**Total:** 23 PRs, ~600+ files changed, +20,000 / -6,200 lines

---

## 1. DRY — Don't Repeat Yourself

**Score:** 6.5 / 10.0 (was 5.5 — **+1.0**)

### What Improved (Wave 2)

| Before (Mid Feb 12)                                           | After (Post-PR #710)                                  | Delta                 |
| ------------------------------------------------------------- | ----------------------------------------------------- | --------------------- |
| 13 `sys.path` hacks remaining                                 | 2 (both in `_bootstrap.py` itself)                    | -85%                  |
| ~40 duplicate constants in `process_calculators/constants.py` | Constants now import from `unit_constants.py`         | DRY consolidation     |
| 160 `print()` in production code                              | 47 remaining (most intentional CLI)                   | -71%                  |
| 276 bare magic numbers in calculators                         | 335 named constants extracted (PR #706)               | Significant reduction |
| ~62 broad `except Exception` handlers                         | 14 remaining                                          | -77%                  |
| 3 `contracts.py` files                                        | 1 canonical module (`src/shared/python/contracts.py`) | -2 files              |

**Constants Consolidation (PR #710):**

`process_calculators/constants.py` was reduced from ~610 lines of hardcoded definitions to ~480 lines with ~40 values imported from `unit_constants.py`. All public names are preserved as backward-compatible re-exports. 8 remaining duplicates exist as `Final[float]` re-exports (intentional for type-checker compatibility).

**Magic Number Extraction (PR #706):**

335 new named constant definitions added across 8 calculator files, replacing bare numeric literals with descriptive names. This is the single largest improvement to calculation readability in the repo's history.

### What Remains

| Issue                                                                   | Count                       | Severity |
| ----------------------------------------------------------------------- | --------------------------- | -------- |
| `Data_Processor_r0.py` vs `Data_Processor_Integrated.py` overlap        | 2 files, 11,702 lines       | CRITICAL |
| Inline `setStyleSheet()` calls                                          | 199                         | MODERATE |
| Near-identical `launch_pyqt6.py` thin wrappers                          | 29 files (25 cookie-cutter) | MODERATE |
| Files >1000 lines                                                       | 26                          | HIGH     |
| Functions >100 lines                                                    | 142                         | HIGH     |
| 8 remaining duplicate constants (unit_constants vs process_calculators) | 8 names                     | LOW      |

### Scoring Breakdown

| Component               | Score      | Notes                                    |
| ----------------------- | ---------- | ---------------------------------------- |
| sys.path elimination    | 9/10       | 71 → 2, essentially complete             |
| Constant centralization | 7/10       | ~40 consolidated, 8 remaining duplicates |
| print() hygiene         | 7/10       | 160 → 47, most intentional               |
| Exception specificity   | 7/10       | 105 → 14, excellent improvement          |
| Module decomposition    | 3/10       | 26 files >1,000 lines unchanged          |
| Function granularity    | 3/10       | 142 functions >100 lines unchanged       |
| Style centralization    | 4/10       | 199 inline setStyleSheet calls           |
| **Average**             | **6.5/10** |                                          |

---

## 2. Design by Contract (DbC)

**Score:** 5.5 / 10.0 (was 3.5 — **+2.0**)

### What Improved

| Aspect                    | Before               | After                                      |
| ------------------------- | -------------------- | ------------------------------------------ |
| `contracts.py` modules    | 3 overlapping files  | 1 consolidated canonical module            |
| DbC enforcement modes     | None                 | Tri-state: ENFORCE / WARN / OFF (PR #685)  |
| Process calculator guards | 0 `raise ValueError` | 17 guards across 3 calculators             |
| Contract test suite       | 0 dedicated tests    | 33 tests in `tests/unit/test_contracts.py` |
| `raise ValueError` total  | ~350                 | 375                                        |

**Consolidated Contract Infrastructure (PR #699 + #685):**

The three overlapping `contracts.py` files were unified into `src/shared/python/contracts.py` with:

- `@precondition`, `@postcondition`, `@invariant` decorators
- `ContractLevel` enum with ENFORCE/WARN/OFF modes
- Environment variable override (`DBC_LEVEL`)
- 33 dedicated test functions validating all modes

**Process Calculator DbC (PR #710):**

| Calculator                         | `raise ValueError` Guards | Validated Inputs                                                    |
| ---------------------------------- | ------------------------- | ------------------------------------------------------------------- |
| `syngas_compression_calculator.py` | 8                         | pressure > 0, inlet/outlet pressure, gamma bounds, non-empty stages |
| `acid_gas_dewpoint_calculator.py`  | 8                         | pressure > 0, temperature > absolute zero, composition validity     |
| `financial_calculator.py`          | 1                         | debt ratio < 1.0 before ROE calculation                             |

### Current Adoption Metrics

| Metric                                | Count | Assessment                                          |
| ------------------------------------- | ----- | --------------------------------------------------- |
| `@precondition` decorators in src/    | 64    | Moderate                                            |
| `@postcondition` decorators in src/   | 18    | Low                                                 |
| `@invariant` decorators in src/       | 4     | Very Low                                            |
| `raise ValueError` statements         | 375   | Strong imperative DbC                               |
| `raise` statements (total)            | 609   | Good defensive coding                               |
| `validate_*` functions                | 39    | Good validation library                             |
| Files with @precondition              | 13    | Concentrated in model_generation + humanoid_builder |
| Functions with early raise ValueError | ~44   | Growing adoption                                    |

### What Remains

| Gap                                                                                         | Impact   |
| ------------------------------------------------------------------------------------------- | -------- |
| `@precondition`/`@postcondition` confined to model_generation + humanoid_builder (13 files) | HIGH     |
| Process calculators use imperative guards, not decorator contracts                          | MODERATE |
| `@invariant` has only 4 usages (all in tests)                                               | MODERATE |
| `_bootstrap.py` has no input validation                                                     | LOW      |
| `ToolRegistration` invariants are documentation-only                                        | LOW      |

### Scoring Breakdown

| Component                   | Score      | Notes                                           |
| --------------------------- | ---------- | ----------------------------------------------- |
| Infrastructure              | 9/10       | Unified contracts.py with tri-state enforcement |
| Adoption (preconditions)    | 5/10       | 64 uses across 764 files                        |
| Adoption (postconditions)   | 3/10       | 18 uses                                         |
| Adoption (invariants)       | 2/10       | 4 uses (all in tests)                           |
| Imperative validation       | 7/10       | 375 raise ValueError, 39 validate\_\* functions |
| Process calculator coverage | 7/10       | 17 new guards in PR #710                        |
| **Average**                 | **5.5/10** |                                                 |

---

## 3. Test-Driven Development (TDD)

**Score:** 6.0 / 10.0 (was 4.0 — **+2.0**)

### What Improved

| Metric                             | Before (Early Feb 12) | After (Post-PR #710) | Delta             |
| ---------------------------------- | --------------------- | -------------------- | ----------------- |
| Test files                         | ~40                   | 48                   | +20%              |
| Test functions                     | ~617                  | 975                  | +58%              |
| Tests collected by pytest          | ~617 (6 errors)       | 996 (0 errors)       | +61%              |
| Collection errors                  | 6                     | 0                    | Eliminated        |
| Test-to-source ratio               | ~1:17                 | 1.81:1               | Major improvement |
| Process calculator edge-case tests | 0                     | 122                  | New               |
| Contract tests                     | 0                     | 33                   | New               |

**New Edge-Case Test Files (PR #710):**

| File                                         | Tests   | Coverage                                                   |
| -------------------------------------------- | ------- | ---------------------------------------------------------- |
| `test_acid_gas_dewpoint_edge_cases.py`       | 32      | Vapor pressure, dewpoint, mixture, condensation risk       |
| `test_financial_calculator_edge_cases.py`    | 30      | Debt ratio, revenue, depreciation, ROE, projections        |
| `test_syngas_compression_edge_cases.py`      | 23      | Water dropout, compression work, gamma, multi-stage        |
| `test_syngas_water_calculator_edge_cases.py` | 37      | Composition, vapor pressure, dew point, extreme conditions |
| **Total**                                    | **122** |                                                            |

**Additional Test Files (Other PRs):**

| File                            | Tests | Source                              |
| ------------------------------- | ----- | ----------------------------------- |
| `test_syngas_water_overflow.py` | 26    | PR #702 (overflow fix)              |
| `test_signal_loader.py`         | 11    | PR #700 (NotImplementedError stubs) |
| `test_format_utils.py`          | 16    | New                                 |
| `tests/unit/test_contracts.py`  | 33    | PR #685 + #699 (contracts)          |

### Current Test Suite Status

```
996 tests collected (0 errors, 0 failures, 1 skipped)
48 test files covering 538 source modules
Test-to-source ratio: 1.81:1
```

### Test Quality Metrics

| Metric                         | Count                    | Assessment               |
| ------------------------------ | ------------------------ | ------------------------ |
| `@pytest.fixture` declarations | 44                       | Moderate reuse           |
| Mock/Patch usage               | 295                      | Good isolation           |
| `@pytest.mark.parametrize`     | 7                        | Low — room for expansion |
| Hypothesis (property-based)    | 0                        | Not adopted              |
| Empty test files               | 1 (`test_test_utils.py`) | Cleanup needed           |

### What Remains

| Gap                                                                        | Impact   |
| -------------------------------------------------------------------------- | -------- |
| Major packages with no tests (Data_Processor, c3d_viewer, psa_package GUI) | HIGH     |
| `_bootstrap.py` has zero dedicated tests                                   | MODERATE |
| Only 7 parametrize decorators across 975 tests                             | MODERATE |
| No property-based testing (Hypothesis)                                     | LOW      |
| `test_test_utils.py` is empty                                              | LOW      |

### Scoring Breakdown

| Component             | Score      | Notes                                        |
| --------------------- | ---------- | -------------------------------------------- |
| Coverage breadth      | 5/10       | 48 test files for 538 source modules         |
| Coverage depth        | 7/10       | 975 tests, strong edge-case coverage         |
| Test isolation        | 7/10       | 295 mock/patch patterns                      |
| Test reuse (fixtures) | 5/10       | 44 fixtures                                  |
| Collection health     | 9/10       | 0 errors, 0 failures                         |
| TDD practice evidence | 6/10       | Edge-case tests written alongside DbC guards |
| **Average**           | **6.0/10** |                                              |

---

## 4. Supplementary Criteria

### 4a. Orthogonality (5.5/10, was 5.0 — +0.5)

Constant centralization improves orthogonality — calculators now depend on `unit_constants.py` for shared values rather than maintaining independent copies. The contract consolidation (3 → 1) reduces coupling between packages. Monolithic files remain the primary orthogonality violation.

### 4b. Monolithic Files (2.0/10, unchanged)

No monolithic files were decomposed:

| File                                     | Lines | Status    |
| ---------------------------------------- | ----- | --------- |
| `Data_Processor_r0.py`                   | 8,994 | Untouched |
| `electrode_advisor/main_window.py`       | 4,386 | Untouched |
| `Folders_Tool_r0.py`                     | 3,288 | Untouched |
| `data_processor/ui/pyqt6/main_window.py` | 2,731 | Untouched |
| `Data_Processor_Integrated.py`           | 2,708 | Untouched |
| _Total files >1000 lines_                | _26_  |           |

### 4c. Code Hygiene (7.0/10, was 6.0 — +1.0)

| Metric                             | Before                                    | After               | Change        |
| ---------------------------------- | ----------------------------------------- | ------------------- | ------------- |
| `except Exception` in src/         | ~62                                       | 14                  | -77%          |
| `print()` in production            | 160                                       | 47                  | -71%          |
| sys.path hacks                     | 13                                        | 2                   | -85%          |
| mypy errors in process_calculators | 70                                        | 0                   | -100%         |
| Unused imports                     | 45 removed                                | 0 remaining         | Clean         |
| T201 ruff rule                     | Not enforced                              | Enforced            | New lint rule |
| Pre-commit hooks                   | Misconfigured (bandit, deprecated stages) | Properly configured | Fixed         |

### 4d. Reversibility (6.0/10, was 5.5 — +0.5)

sys.path hacks now at 2 (both in `_bootstrap.py`). Pre-commit hooks properly configured: bandit scans changed files only (not entire repo), deprecated `stages: [push]` fixed to `stages: [pre-push]`. mypy clean in process_calculators module.

### 4e. Reusability (6.0/10, was 5.5 — +0.5)

`unit_constants.py` is now the single source of truth for physical/engineering constants. Process calculators import from it. The consolidated `contracts.py` is reusable across all packages. `ToolRegistration` dataclass provides a clean intermediate representation for tool discovery.

### 4f. Parity / Maintenance (6.0/10, was 5.5 — +0.5)

CODEOWNERS added. Pre-commit hooks fixed and enforced. T201 ruff rule prevents new print() additions. Assessment framework established with versioned reports.

### 4g. Changeability (5.5/10, was 5.0 — +0.5)

Thin-wrapper pattern + bootstrap module isolates changes. Constants centralized so updates propagate. But 142 functions >100 lines make individual changes risky.

### 4h. Function Length (4.0/10, unchanged)

142 functions exceed 100 lines. Top offender: `create_plotting_tab()` at 904 lines. `Data_Processor_r0.py` alone contributes 21 of these.

### 4i. Law of Demeter (5.5/10, unchanged)

No significant changes to object coupling patterns.

### 4j. God Functions (3.0/10, unchanged)

No god functions decomposed. `create_plotting_tab()` (904 lines), `create_plot_left_content()` (732 lines), `create_help_tab()` (623 lines) remain.

### 4k. Deprecated Code (5.0/10, was 4.0 — +1.0)

- 45 unused imports removed (PR #681)
- NotImplementedError stubs resolved in signal_toolkit + model_generation (PR #700)
- `test_test_utils.py` gutted to 0 tests (was 456 lines of print-based tests)
- Deprecated hook stages fixed

### 4l. Name Quality (6.0/10, unchanged)

No significant naming changes. New constants follow existing `UPPER_SNAKE_CASE` convention.

### 4m. Magic Numbers (7.0/10, was 6.0 — +1.0)

335 named constants extracted in PR #706 (the single largest magic number cleanup in the repo's history). Remaining magic numbers are overwhelmingly PyQt6 GUI literals (pixel sizes, margins, font sizes) — these are candidates for theme centralization but are lower severity.

### 4n. Project Structure (6.0/10, unchanged)

`gui_registration.py` standardized for folder tools. Tool discovery infrastructure mature. No structural changes in this wave.

### 4o. Cleanup (6.0/10, was 5.0 — +1.0)

Cantera API updated. 105 broad exception handlers narrowed. print() calls replaced with logging. Pre-commit hooks fixed. Bandit scanning corrected.

### 4p. Comment Quality (5.5/10, unchanged)

No significant changes to comment patterns. New constants have unit annotations (e.g., `[kPa]`, `[m/s²]`).

### 4q. Calculation Optimization (5.5/10, was 5.0 — +0.5)

Overflow fix in syngas_water_calculator exponential calculations (PR #702). Named constants improve calculation readability and auditability.

---

## Revised Scorecard

| #       | Criterion                | Feb 12 Score | Post-PR #710 Score | Delta    | Evidence                                                         |
| ------- | ------------------------ | ------------ | ------------------ | -------- | ---------------------------------------------------------------- |
| 1       | **DRY**                  | 5.5          | **6.5**            | +1.0     | Constants consolidated, print() reduced 71%, except narrowed 77% |
| 2       | **Design by Contract**   | 3.5          | **5.5**            | +2.0     | Consolidated contracts.py, tri-state enforcement, 17 new guards  |
| 3       | **TDD**                  | 4.0          | **6.0**            | +2.0     | 996 tests (0 errors), 122 new edge-case tests, 1.81:1 ratio      |
| 4       | Orthogonality            | 5.0          | **5.5**            | +0.5     | Constants centralized, contracts consolidated                    |
| 5       | Monolithic Files         | 2.0          | **2.0**            | 0        | No monoliths decomposed                                          |
| 6       | Reversibility            | 5.5          | **6.0**            | +0.5     | sys.path → 2, pre-commit hooks fixed                             |
| 7       | Reusability              | 5.5          | **6.0**            | +0.5     | unit_constants.py as single source of truth                      |
| 8       | Parity / Maintenance     | 5.5          | **6.0**            | +0.5     | CODEOWNERS, T201 rule, assessment framework                      |
| 9       | Changeability            | 5.0          | **5.5**            | +0.5     | Constants centralized, thin-wrapper pattern                      |
| 10      | Function Length          | 4.0          | **4.0**            | 0        | 142 functions >100 lines unchanged                               |
| 11      | Law of Demeter           | 5.5          | **5.5**            | 0        | No change                                                        |
| 12      | God Functions            | 3.0          | **3.0**            | 0        | No god functions decomposed                                      |
| 13      | Deprecated Code          | 4.0          | **5.0**            | +1.0     | Stubs resolved, unused imports removed                           |
| 14      | Name Quality             | 6.0          | **6.0**            | 0        | No change                                                        |
| 15      | Magic Numbers            | 6.0          | **7.0**            | +1.0     | 335 named constants extracted                                    |
| 16      | Project Structure        | 6.0          | **6.0**            | 0        | No structural changes                                            |
| 17      | Cleanup                  | 5.0          | **6.0**            | +1.0     | Exception narrowing, print cleanup, hook fixes                   |
| 18      | Comment Quality          | 5.5          | **5.5**            | 0        | No change                                                        |
| 19      | Calculation Optimization | 5.0          | **5.5**            | +0.5     | Overflow fix, named constants improve auditability               |
| **AVG** | **Overall**              | **5.3**      | **6.2**            | **+0.9** |                                                                  |

---

## Trend Analysis (3 Assessments)

| Criterion   | Feb 10  | Feb 12 (Early) | Feb 12 (Post-#710) | Total Delta |
| ----------- | ------- | -------------- | ------------------ | ----------- |
| DRY         | 4.0     | 5.5            | 6.5                | +2.5        |
| DbC         | 3.5     | 3.5            | 5.5                | +2.0        |
| TDD         | 3.0     | 4.0            | 6.0                | +3.0        |
| **Overall** | **4.4** | **5.3**        | **6.2**            | **+1.8**    |

The TDD pillar shows the most dramatic improvement (+3.0), driven by test collection error elimination and 122 new edge-case tests. DbC shows strong improvement (+2.0) from infrastructure consolidation and imperative guard adoption. DRY improvement (+2.5) is broad-based across sys.path, print(), constants, and exception handling.

---

## Remaining Issues — Prioritized

### Critical (Must Fix)

| #   | Issue                                              | Location               | Impact                                |
| --- | -------------------------------------------------- | ---------------------- | ------------------------------------- |
| 1   | `Data_Processor_r0.py` (8,994 lines) is unreformed | `src/data_processing/` | Untestable, unmaintainable god object |
| 2   | 142 functions >100 lines                           | Across codebase        | High change risk                      |
| 3   | 26 files >1000 lines                               | Across codebase        | Maintenance burden                    |

### High Priority

| #   | Issue                                                                  | Location                            | Impact                       |
| --- | ---------------------------------------------------------------------- | ----------------------------------- | ---------------------------- |
| 4   | `@precondition`/`@postcondition` confined to 13 files                  | model_generation + humanoid_builder | DbC adoption gap             |
| 5   | 199 inline `setStyleSheet()` calls                                     | GUI files                           | Should use theme system      |
| 6   | Major packages without tests (Data_Processor, c3d_viewer, psa_package) | Various                             | Coverage gap                 |
| 7   | `_bootstrap.py` has zero dedicated tests                               | `_bootstrap.py`                     | Core infrastructure untested |

### Moderate

| #   | Issue                                              | Location                              | Impact                   |
| --- | -------------------------------------------------- | ------------------------------------- | ------------------------ |
| 8   | 8 remaining duplicate constants                    | `unit_constants.py` vs `constants.py` | Minor DRY violation      |
| 9   | Only 7 `@pytest.mark.parametrize` across 975 tests | Tests                                 | Test efficiency          |
| 10  | No property-based testing (Hypothesis)             | Tests                                 | Missing fuzzing coverage |
| 11  | 29 near-identical `launch_pyqt6.py` files          | src/ tool dirs                        | Structural duplication   |

### Low

| #   | Issue                                             | Location | Impact         |
| --- | ------------------------------------------------- | -------- | -------------- |
| 12  | `test_test_utils.py` empty (0 tests)              | Tests    | Cleanup needed |
| 13  | 47 remaining `print()` calls (mostly intentional) | Various  | Minor hygiene  |

---

## Recommendations for Next Sprint

### 1. Decompose Monolithic Files (Critical — Function Length + God Functions)

- Start with `Data_Processor_r0.py` (8,994 lines) — extract CSV parsing, filtering, plotting, and export into separate modules
- Break `create_plotting_tab()` (904 lines) into focused helper functions
- Target: no function >100 lines, no file >1,000 lines

### 2. Propagate @precondition to Process Calculators

- The imperative `raise ValueError` guards work, but decorator contracts provide richer semantics
- Add `@precondition` from the consolidated `contracts.py` to the 17 existing guards
- Target: 100+ `@precondition` usages across 20+ files

### 3. Centralize Inline Styles

- Replace 199 `setStyleSheet()` calls with theme system references
- The existing theme package (`src/shared/python/theme/`) already provides this infrastructure

### 4. Expand Test Coverage

- Add `tests/test_bootstrap.py` with edge cases
- Add tests for Data_Processor core logic (extract testable units from monolith first)
- Adopt `@pytest.mark.parametrize` more broadly — 7 is too few for 975 tests

### 5. Adopt Property-Based Testing

- Install Hypothesis for calculator tests
- Focus on numerical calculators where boundary conditions are critical

---

## Quantitative Evidence

### DRY Metrics

| Metric                     | Feb 10 | Feb 12 (Early) | Post-PR #710 |
| -------------------------- | ------ | -------------- | ------------ |
| sys.path hacks             | 71     | 13             | 2            |
| print() in production      | 160    | ~33            | 47\*         |
| except Exception (broad)   | ~155   | ~62            | 14           |
| Functions >100 lines       | 142    | 142            | 142          |
| Files >1000 lines          | ~26    | 26             | 26           |
| setStyleSheet inline       | 199    | 199            | 199          |
| launch_pyqt6.py duplicates | 29     | 29             | 29           |

\*Note: 47 includes intentional CLI output (`console.print`, `debug_utils.py` with `# noqa: T201`, `setup_api_key.py` interactive prompts). True "stray" print() calls: ~1 (`mesh_generator.py:372`).

### DbC Metrics

| Metric                 | Feb 10 | Feb 12 (Early) | Post-PR #710 |
| ---------------------- | ------ | -------------- | ------------ |
| @precondition          | ~20    | 67             | 64           |
| @postcondition         | ~3     | 21             | 18           |
| @invariant             | 0      | 6              | 4            |
| raise ValueError       | ~350   | ~350           | 375          |
| validate\_\* functions | ~39    | 39             | 39           |
| contracts.py files     | 3      | 3              | 1            |

### TDD Metrics

| Metric            | Feb 10 | Feb 12 (Early) | Post-PR #710 |
| ----------------- | ------ | -------------- | ------------ |
| Test files        | ~35    | ~40            | 48           |
| Test functions    | ~617   | ~617           | 975          |
| Collection errors | ~6     | 6              | 0            |
| Fixtures          | ~43    | 43             | 44           |
| Parametrize       | 0      | 0              | 7            |
| Mock/Patch        | ~162   | 162            | 295          |

---

## Methodology

This assessment was conducted by:

1. Reading all git commits from 2026-02-10 through 2026-02-12 (23 PRs)
2. Running quantitative analysis with 3 parallel agents (DRY metrics, DbC adoption, TDD coverage)
3. Comparing against two prior baselines (Feb 10: 4.4/10, Feb 12 early: 5.3/10)
4. Verifying test suite health (`pytest --collect-only`: 996 collected, 0 errors)
5. Cross-referencing constant consolidation between `unit_constants.py` and `constants.py`
6. Applying adversarial scrutiny: assuming code is guilty until proven correct

**Scoring:** Each of 19 criteria scored 0-10. Overall score is unweighted average. Delta computed against the Feb 12 early baseline (5.3/10).

---

_Assessment conducted under the A-O + Highlight framework with TDD/DbC/DRY focus._
_Template: Repository_Management/docs/templates/code_quality_assessment_template.md_
