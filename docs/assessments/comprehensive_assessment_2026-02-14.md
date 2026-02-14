# Tools — Comprehensive Quality Assessment (2026-02-14)

## Executive Summary

Tools (ud-tools) is the organization's shared engineering toolkit with 774 Python source files and 72 test files. It provides URDF generation, signal processing, process calculators, data processing, file management, media processing, and a comprehensive launcher system. The codebase is **sprawling and diverse** with excellent shared libraries but significant God Modules in GUI/tool applications and a disproportionately low test-to-source ratio.

**Overall Score: 6.5/10**

---

## A-O Framework Assessment

| ID    | Category                      | Score | Key Findings                                                                                                                                                                      |
| ----- | ----------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Architecture & Implementation | 6.5   | Good shared library pattern (`src/shared/python/`). But tool-specific code is deeply nested (`src/electrode_advisor/python/...`). Package discovery is complex.                   |
| **B** | Code Quality & Hygiene        | 6.0   | 19 print() calls (some with noqa). 11 TODO/FIXME markers. Missing type hints in baghouse/c3d test files.                                                                          |
| **C** | Documentation & Comments      | 6.5   | TOOLS_INDEX.md, QUICKSTART.md. But many tool directories lack README files. Docstrings vary widely.                                                                               |
| **D** | User Experience               | 7.0   | Unified launcher system (`launch_tools_main.py`). Tool registration via `tools.json`.                                                                                             |
| **E** | Performance & Scalability     | 6.5   | Some tools process files sequentially. Signal toolkit could benefit from batch processing.                                                                                        |
| **F** | Installation & Deployment     | 6.5   | Complex `pyproject.toml` with 7 optional dependency groups. Package discovery searches 3 directories.                                                                             |
| **G** | Testing & Validation          | 4.5   | **Critical**: Only 72 test files for 774 source files (9.3% ratio). Existing issue #756 targets 25% coverage.                                                                     |
| **H** | Error Handling & Debugging    | 6.0   | Print statements in mesh generator, debug utils, and scientific auditor. `debug_utils.py` uses print by design (with noqa).                                                       |
| **I** | Security & Input Validation   | 6.5   | `scientific_auditor.py` does security scanning. But some file operations lack path sanitization.                                                                                  |
| **J** | Extensibility                 | 7.0   | Tool registration system via JSON manifest. Shared Python packages available fleet-wide.                                                                                          |
| **K** | Reproducibility               | 6.5   | `requirements-lock.txt` exists. But stale artifacts in root (`=`, `folder_fix_pro.log`, etc.).                                                                                    |
| **L** | Maintainability               | 4.5   | **Critical**: `electrode_advisor main_window.py` (4,386 lines), `Folders_Tool_r0.py` (3,288 lines), `data_processor main_window.py` (2,734 lines). These are extreme God Modules. |
| **M** | Education                     | 5.5   | Basic tool documentation. No tutorials or getting-started guides per tool.                                                                                                        |
| **N** | Visualization                 | 6.5   | Signal toolkit has plotting. Solar system model has 3D visualization.                                                                                                             |
| **O** | CI/CD                         | 7.5   | `ci-standard.yml`, `topology-governance.yml`, `docs-governance.yml`. Good automation.                                                                                             |

**A-O Average: 6.27/10**

---

## Pragmatic Programmer Assessment

### 1. Don't Repeat Yourself (DRY) — 4.5/10

**Issues Identified:**

- **PP-DRY-001**: Cross-tool boilerplate — every tool directory has duplicate `launch_pyqt6.py` files with nearly identical content. Issue #735 exists.
- **PP-DRY-002**: Tool skeleton (setup, configuration, logging initialization) is repeated across 30+ tools. Issue #741, #726 exist.
- **PP-DRY-003**:`data_processor` has duplicated code between `Data_Processor_Integrated.py` (2,717 lines) and `ui/pyqt6/main_window.py` (2,734 lines).
- **PP-DRY-004**: Model generation has `editor/frankenstein_editor.py` duplicated between `src/tools/` (1,399 lines) and `src/shared/python/` (1,399 lines).
- **PP-DRY-005**: Multiple folder tools (`folder_tools/`) share file-traversal, size calculation, and reporting logic.

### 2. Orthogonality & Decoupling — 4.0/10

**Issues Identified:**

- **PP-ORTH-001**: `electrode_advisor main_window.py` (4,386 lines) — **extreme** God Module mixing UI layout, calculation engine, visualization, file I/O, and theming. Issue #736 exists.
- **PP-ORTH-002**: `Folders_Tool_r0.py` (3,288 lines) — mixes file system operations, GUI, reporting, and analysis.
- **PP-ORTH-003**: `data_processor main_window.py` (2,734 lines) — mixes data loading, transformation, visualization, and export.
- **PP-ORTH-004**: `neural_network.py` (1,353 lines) — mixes model architecture, training, evaluation, and persistence.
- **PP-ORTH-005**: `signal_toolkit/widget.py` (1,794 lines) — combines signal processing UI, computation, and export.
- **PP-ORTH-006**: `pressure_drop_interface.py` (1,294 lines) — mixes calculation with GUI rendering.

### 3. Reversibility & Flexibility — 6.0/10

- Tool registration via JSON is a good reversibility pattern.
- **CQ-REV-001**: Many tools have hardcoded file paths and output directories.
- **CQ-REV-002**: No abstraction for data backend in data processor — tightly coupled to in-memory pandas.

### 4. Code Quality & Craftsmanship — 6.0/10

- Modern Python used in shared libraries.
- Legacy tool code uses older patterns.
- 11 TODO/FIXME markers indicate unfinished work.

### 5. Error Handling & Robustness — 5.5/10

- **PP-ERR-001**: 19 print() calls in production code (beyond debug utils).
- **PP-ERR-002**: Mesh generator uses broad `print(f"Warning: ...")` instead of `logging.warning()`.
- **PP-ERR-003**: Several tool widgets swallow exceptions with bare `except` blocks (need grep verification).

### 6. Testing & Validation — 4.5/10

- **PP-TEST-001**: 72 test files for 774 source files — **9.3% ratio** is critically low.
- **PP-TEST-002**: Most large GUI modules have zero or minimal test coverage.
- **PP-TEST-003**: No property-based tests detected.
- **PP-TEST-004**: Missing test infrastructure for tools using PyQt6 GUIs.

### 7. Documentation & Communication — 6.0/10

- TOOLS_INDEX.md provides a catalog.
- Most individual tools lack detailed documentation.
- **PP-DOC-001**: Need per-tool README files.

### 8. Automation & Tooling — 7.5/10

- Unified launcher system.
- CI/CD comprehensive.
- Tool manifest auto-checking.

**Pragmatic Programmer Average: 5.50/10**

---

## Code Quality Deep-Dive

### Design by Contract (DbC) — 5.0/10

- **CQ-DBC-001**: Shared libraries (`model_generation/`, `signal_toolkit/`) have some contract patterns but adoption is inconsistent. Issue #721 exists.
- **CQ-DBC-002**: Calculator interfaces lack standardized pre/postconditions. Issue #729 exists.
- **CQ-DBC-003**: No input validation contracts for file management tools — risky for path traversal.

### Test-Driven Development (TDD) — 4.5/10

- **CQ-TDD-001**: Critical coverage gap — 9.3% test ratio.
- **CQ-TDD-002**: 11 TODO/FIXME markers.
- **CQ-TDD-003**: No mutation testing.
- **CQ-TDD-004**: Issue #756 targets 25% but current state is far from that.

### DRY Compliance — 4.5/10

_(See PP-DRY-001 through PP-DRY-005 above)_

### Orthogonality — 4.0/10

_(See PP-ORTH-001 through PP-ORTH-006 above)_

### Reversibility — 6.0/10

_(See CQ-REV-001, CQ-REV-002 above)_

---

## Issue Summary

| ID          | Category       | Severity | Description                                              | Existing Issue |
| ----------- | -------------- | -------- | -------------------------------------------------------- | -------------- |
| PP-DRY-001  | DRY            | Critical | 30+ duplicate `launch_pyqt6.py` files                    | #735           |
| PP-DRY-002  | DRY            | Critical | Tool skeleton boilerplate duplication                    | #741, #726     |
| PP-DRY-003  | DRY            | Major    | Data processor code duplication                          | —              |
| PP-DRY-004  | DRY            | Major    | Frankenstein editor duplication across packages          | —              |
| PP-DRY-005  | DRY            | Minor    | Folder tools file-traversal duplication                  | —              |
| PP-ORTH-001 | Orthogonality  | Critical | `electrode_advisor main_window.py` 4,386-line God Module | #736, #740     |
| PP-ORTH-002 | Orthogonality  | Critical | `Folders_Tool_r0.py` 3,288-line God Module               | #740           |
| PP-ORTH-003 | Orthogonality  | Critical | `data_processor main_window.py` 2,734-line God Module    | #740           |
| PP-ORTH-004 | Orthogonality  | Major    | `neural_network.py` 1,353-line mixed module              | —              |
| PP-ORTH-005 | Orthogonality  | Major    | `signal_toolkit/widget.py` 1,794-line mixed module       | —              |
| PP-ORTH-006 | Orthogonality  | Major    | `pressure_drop_interface.py` 1,294-line mixed module     | —              |
| PP-ERR-001  | Error Handling | Major    | 19 print() calls in production code                      | —              |
| PP-ERR-002  | Error Handling | Minor    | Generic warning prints in mesh generator                 | —              |
| PP-TEST-001 | Testing        | Critical | 9.3% test ratio — critically low                         | #756           |
| PP-TEST-002 | Testing        | Major    | GUI modules lack test coverage                           | —              |
| PP-TEST-003 | Testing        | Major    | No property-based tests                                  | —              |
| CQ-DBC-001  | DbC            | Major    | Inconsistent contract adoption in shared libs            | #721, #729     |
| CQ-DBC-002  | DbC            | Major    | Calculator interfaces lack pre/postconditions            | #729           |
| CQ-DBC-003  | DbC            | Minor    | File management tools lack path validation               | —              |
| CQ-REV-001  | Reversibility  | Minor    | Hardcoded file paths in tools                            | —              |
| CQ-REV-002  | Reversibility  | Minor    | Data processor tightly coupled to pandas                 | —              |
| CQ-TDD-001  | Testing        | Major    | 11 TODO/FIXME markers                                    | —              |

**Total Issues: 22 (6 Critical, 10 Major, 6 Minor)**
