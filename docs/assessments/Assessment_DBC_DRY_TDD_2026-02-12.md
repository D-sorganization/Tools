# Tools Repository — DBC / DRY / TDD Quality Assessment

**Assessment Date:** 2026-02-12
**Repository:** Tools (AffineDrift)
**Total Source Files:** 764 Python files (src/)
**Overall Grade:** 5.8/10

---

## Executive Summary

The Tools repository has **strong contract infrastructure** (three separate `contracts.py` implementations) but **poor production adoption**. DRY violations are significant with 142 functions exceeding 100 lines, massive monolith files (Data_Processor at 8,994 lines), and 276 magic number instances. Test coverage is moderate (41 test files, 845 tests for 538 source modules) but leaves major surface area uncovered.

---

## 1. Design by Contract (DbC) — Grade: 6.0/10

### 1.1 Infrastructure (9/10)

Three complete contract libraries exist:

- `src/shared/python/contracts.py` — Unified contract module (newly merged)
- `src/shared/python/model_generation/core/contracts.py` — Full decorator suite
- `src/shared/python/humanoid_character_builder/contracts.py` — Domain-specific contracts

### 1.2 Adoption Metrics

| Metric                              | Count | Assessment              |
| ----------------------------------- | ----- | ----------------------- |
| `@precondition` decorators in src/  | 67    | Moderate                |
| `@postcondition` decorators in src/ | 21    | Low                     |
| `@invariant` decorators in src/     | 6     | Very Low                |
| `raise` statements                  | 608   | Good defensive coding   |
| `assert` statements (non-test)      | 28    | Appropriately sparse    |
| `validate_*` functions              | 39    | Good validation library |

### 1.3 Critical Gaps

**High-Risk APIs Without Contracts:**

- `Data_Processor_r0.py` (8,994 lines) — No preconditions on data processing pipeline
- `base_builder.py` — Public API methods lack input validation decorators
- `types.py` — `from_dict()` methods accept arbitrary dictionaries without schema validation
- `conversion/service.py` — Unit conversion accepts unchecked inputs

**Contract Redundancy Issue:** Three separate `contracts.py` files implement overlapping functionality. This is itself a DRY violation of the contract infrastructure.

### 1.4 Scoring Breakdown

| Component                 | Score      | Notes                         |
| ------------------------- | ---------- | ----------------------------- |
| Infrastructure            | 9/10       | Three complete libraries      |
| Adoption (preconditions)  | 5/10       | 67 uses across 764 files      |
| Adoption (postconditions) | 3/10       | Only 21 uses                  |
| Adoption (invariants)     | 2/10       | Only 6 uses                   |
| Validation utilities      | 7/10       | 39 validate\_ functions       |
| Consistency               | 4/10       | Multiple overlapping patterns |
| **Average**               | **6.0/10** |                               |

---

## 2. Don't Repeat Yourself (DRY) — Grade: 4.5/10

### 2.1 Monolith Files (Critical)

| File                           | Lines | Issue                                       |
| ------------------------------ | ----- | ------------------------------------------- |
| `Data_Processor_r0.py`         | 8,994 | God object — UI, logic, plotting all in one |
| `main_window.py` (electrode)   | 4,386 | Monolithic window class                     |
| `Folders_Tool_r0.py`           | 3,288 | Single-file application                     |
| `main_window.py` (data proc)   | 2,731 | Duplicate of similar pattern                |
| `Data_Processor_Integrated.py` | 2,708 | Parallel implementation                     |
| `folder_packer_pro.py`         | 1,911 | Single-file tool                            |

### 2.2 Function-Level Duplication

| Metric                               | Count                             | Target               |
| ------------------------------------ | --------------------------------- | -------------------- |
| Functions >100 lines                 | 142                               | < 30                 |
| Largest function                     | 904 lines (`create_plotting_tab`) | < 100                |
| `sys.path` manipulations             | 28                                | 0                    |
| `print()` in production code         | 160                               | 0 (use logging)      |
| Inline `setStyleSheet()` calls       | 199                               | 0 (use theme system) |
| Duplicate JSON/YAML loading patterns | 70                                | Centralize           |

### 2.3 Top 5 DRY Hot Spots

1. **`Data_Processor_r0.py`** — 904-line function; entire app in one file
2. **`setStyleSheet` duplication** — 199 inline style definitions instead of using theme package
3. **`print()` statements** — 160 instances instead of `logging` module
4. **`sys.path` hacks** — 28 path manipulations instead of proper package installation
5. **Magic numbers** — 276 bare numeric literals scattered through calculations

### 2.4 Scoring Breakdown

| Component                  | Score      | Notes                       |
| -------------------------- | ---------- | --------------------------- |
| Module decomposition       | 3/10       | 6 files >1,000 lines        |
| Function granularity       | 3/10       | 142 functions >100 lines    |
| Style/theme centralization | 4/10       | 199 inline styles           |
| Logging consistency        | 4/10       | 160 print() calls           |
| Magic number hygiene       | 5/10       | 276 bare literals           |
| Path management            | 5/10       | 28 sys.path hacks           |
| Infrastructure reuse       | 7/10       | Good shared/ packages exist |
| **Average**                | **4.5/10** |                             |

---

## 3. Test-Driven Development (TDD) — Grade: 5.0/10

### 3.1 Coverage Metrics

| Metric                    | Count                          | Assessment         |
| ------------------------- | ------------------------------ | ------------------ |
| Test files                | 41                             | Low density        |
| Test functions            | 845                            | Moderate           |
| Source modules (non-init) | 538                            | Large surface area |
| Test-to-source ratio      | 0.076 files, 1.57 tests/module | Below target (3:1) |
| Mock/Patch usage          | 162                            | Moderate isolation |
| Fixture usage             | 43                             | Low reuse          |

### 3.2 Untested Components

Major packages with no corresponding tests:

- `acid_gas_dewpoint/` — Calculator with no test coverage
- `baghouse_calculator/` — Calculator with no test coverage
- `c3d_viewer/` — Viewer with no test coverage
- `document_processing/pdf_renamer/` — Tool with no test coverage
- `electrode_advisor/` — Large application with minimal testing
- `Data_Processor_r0.py` — 8,994-line file with insufficient tests

### 3.3 Test Quality Issues

1. **No parametrized tests visible** — Suggests repetitive test patterns
2. **Low fixture reuse** (43 fixtures for 845 tests) — Test setup is likely duplicated
3. **Some test directories are empty** — `tests/` subdirectories exist but contain no tests
4. **No property-based testing** — No Hypothesis usage detected
5. **No test coverage CI enforcement** — No minimum coverage gates

### 3.4 Scoring Breakdown

| Component             | Score      | Notes                             |
| --------------------- | ---------- | --------------------------------- |
| Coverage breadth      | 4/10       | Major modules untested            |
| Coverage depth        | 5/10       | 845 tests, but shallow            |
| Test isolation        | 6/10       | 162 mock patterns                 |
| Test reuse (fixtures) | 4/10       | Only 43 fixtures                  |
| CI enforcement        | 6/10       | Tests run but no coverage gate    |
| TDD practice evidence | 5/10       | Tests exist but not comprehensive |
| **Average**           | **5.0/10** |                                   |

---

## 4. Remediation Priority

### Phase 1: Quick Wins (1-2 days each)

| #   | Action                                     | Impact          | Effort |
| --- | ------------------------------------------ | --------------- | ------ |
| 1   | Replace 160 `print()` calls with `logging` | DRY +1, Quality | Low    |
| 2   | Extract constants for 276 magic numbers    | DRY +1, DbC     | Medium |
| 3   | Consolidate 3 contracts.py into 1          | DRY +1, DbC     | Low    |
| 4   | Remove 28 `sys.path` hacks                 | DRY +0.5        | Low    |

### Phase 2: Structural (3-5 days each)

| #   | Action                                                   | Impact | Effort |
| --- | -------------------------------------------------------- | ------ | ------ |
| 5   | Decompose `Data_Processor_r0.py` (8,994→<500 lines each) | DRY +2 | High   |
| 6   | Add `@precondition` to 20 critical APIs                  | DbC +2 | Medium |
| 7   | Add tests for acid_gas, baghouse, c3d, pdf_renamer       | TDD +2 | Medium |
| 8   | Centralize 199 inline stylesheet definitions             | DRY +1 | Medium |

### Phase 3: Strategic (1-2 weeks)

| #   | Action                                                | Impact   | Effort |
| --- | ----------------------------------------------------- | -------- | ------ |
| 9   | Introduce pytest-cov with 70% minimum gate            | TDD +1.5 | Low    |
| 10  | Add property-based tests (Hypothesis) for calculators | TDD +1   | Medium |
| 11  | Add `@invariant` to 5+ core classes                   | DbC +1   | Medium |

---

## 5. Target Grades After Remediation

| Dimension   | Current | Phase 1 | Phase 2 | Phase 3 |
| ----------- | ------- | ------- | ------- | ------- |
| DbC         | 6.0     | 6.5     | 8.0     | 9.0     |
| DRY         | 4.5     | 6.0     | 8.0     | 8.5     |
| TDD         | 5.0     | 5.5     | 7.5     | 8.5     |
| **Overall** | **5.8** | **6.3** | **8.0** | **8.7** |

---

_Assessment conducted 2026-02-12 using AST analysis, grep heuristics, and manual code review._
