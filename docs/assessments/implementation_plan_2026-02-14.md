# Tools — Implementation Plan (2026-02-14)

> **Reference Assessment**: `docs/assessments/comprehensive_assessment_2026-02-14.md` > **Principles**: TDD, DbC, DRY, Orthogonality, Reversibility, Decoupled Code

---

## Phase 1: Error Handling, Type Safety & Cleanup (Quick Wins)

**Target Issues**: PP-ERR-001, PP-ERR-002, CQ-TDD-002

### 1.1 Replace print() with logging in production code

**Files to modify (19 occurrences):**

- `src/shared/python/humanoid_character_builder/generators/mesh_generator.py` line 372 — Replace `print(f"Warning: ...")` with `logging.warning(...)`
- `src/tools/scientific_auditor.py` lines 72, 75 — Replace print with logging (remove noqa)
- `src/python/src/utils/debug_utils.py` — These are intentional debug prints with noqa; leave as-is but document the exception

### 1.2 Add return type hints to test classes

**Files to modify:** All baghouse/c3d test files where `def` lines lack `->`.

### 1.3 Resolve TODO/FIXME markers

Review and resolve the 11 TODO/FIXME markers across source code.

### 1.4 Clean up stale root files

**Delete:**

- `=` (stale pip artifact)
- `folder_fix_pro.log`, `folder_packer_pro.log`, `folder_processor.log` (stale logs)
- `tools_launcher.log` (stale log)
- `ruff_output.json`, `ruff_verify.json` (stale CI output)
- `test_output.txt` (stale test output)
- `checks.json`, `issues.json`, `open_issues.json`, `tools_prs.json` (stale metadata)

---

## Phase 2: DRY Consolidation (Critical)

**Target Issues**: PP-DRY-001, PP-DRY-002, PP-DRY-003, PP-DRY-004, relates to #735, #741, #726

### 2.1 Create shared launcher factory (relates to #735)

**Create:** `src/shared/python/upstream_drift_tools/launcher_factory.py`

Extract common launcher patterns from the 30+ duplicate `launch_pyqt6.py` files:

```python
from pathlib import Path
from typing import Protocol
import logging
import sys

logger = logging.getLogger(__name__)

class LaunchableApp(Protocol):
    """Protocol for apps that can be launched via the factory."""

    @staticmethod
    def create_app() -> "QApplication": ...

    @staticmethod
    def create_main_window() -> "QMainWindow": ...

def launch_pyqt6_app(
    app_class: type,
    window_title: str,
    min_size: tuple[int, int] = (800, 600),
) -> int:
    """Unified launcher for PyQt6 applications.

    Preconditions:
        - app_class has create_app and create_main_window methods
    Postconditions:
        - Returns exit code (0 for success)
    """
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        window = app_class()
        window.setWindowTitle(window_title)
        window.setMinimumSize(*min_size)
        window.show()
        return app.exec()
    except ImportError as exc:
        logger.error("PyQt6 not installed: %s", exc)
        return 1
```

**TDD requirement:** Write `tests/test_launcher_factory.py` FIRST.

### 2.2 Create shared tool skeleton (relates to #741, #726)

**Create:** `src/shared/python/upstream_drift_tools/tool_skeleton/`

Package containing:

- `base_tool.py` — Abstract base class for all tools
- `config.py` — Standardized configuration loading
- `logging_setup.py` — Standardized logging initialization
- `cli.py` — Standardized CLI argument parsing

### 2.3 Deduplicate data processor code

Merge overlapping logic between:

- `data_processor/Data_Processor_Integrated.py` (2,717 lines)
- `data_processor/ui/pyqt6/main_window.py` (2,734 lines)

If `Data_Processor_Integrated.py` is legacy, archive it and keep only the PyQt6 version.

### 2.4 Resolve frankenstein_editor duplication

Determine canonical location for `frankenstein_editor.py`:

- If `src/shared/python/model_generation/editor/` is the canonical package, remove `src/tools/model_generation/editor/`
- Update imports to use the shared package location

---

## Phase 3: Orthogonality & Decomposition (Critical)

**Target Issues**: PP-ORTH-001 through PP-ORTH-006, relates to #736, #740

### 3.1 Decompose `electrode_advisor main_window.py` (4,386 lines → relates to #736)

Split into sub-package `src/electrode_advisor/python/electrode_advisor/ui/pyqt6/`:

- `main_window.py` — Thin shell, orchestrator (< 200 lines)
- `calculation_engine.py` — Pure calculation logic (no Qt imports, testable independently)
- `visualization.py` — Matplotlib/Qt visualization
- `input_panel.py` — Input form widgets
- `results_panel.py` — Results display
- `file_operations.py` — Import/export logic
- `theme_manager.py` — Theming

**DbC requirement:** `calculation_engine.py` must enforce:

- Precondition: All required inputs validated before calculation
- Postcondition: Results contain all expected fields
- Invariant: Calculation is pure — no side effects

### 3.2 Decompose `Folders_Tool_r0.py` (3,288 lines)

Split into:

- `folder_tools/folder_tool/core/scanner.py` — File system scanning
- `folder_tools/folder_tool/core/analyzer.py` — Size/structure analysis
- `folder_tools/folder_tool/core/reporter.py` — Report generation
- `folder_tools/folder_tool/ui/main_window.py` — PyQt6 UI (< 300 lines)

### 3.3 Decompose `data_processor main_window.py` (2,734 lines)

Split into:

- `data_processor/core/data_loader.py` — Data loading from various formats
- `data_processor/core/transformer.py` — Data transformation operations
- `data_processor/core/exporter.py` — Export to various formats
- `data_processor/ui/main_window.py` — Thin PyQt6 shell
- `data_processor/ui/visualization.py` — Plot widgets

### 3.4 Decompose `signal_toolkit/widget.py` (1,794 lines)

Split into:

- `signal_toolkit/core/processor.py` — Pure signal processing (no Qt)
- `signal_toolkit/core/analyzer.py` — Analysis algorithms
- `signal_toolkit/ui/widget.py` — PyQt6 widget shell
- `signal_toolkit/ui/plots.py` — Plot rendering

### 3.5 Decompose `neural_network.py` (1,353 lines)

Split into:

- `data_processor/core/neural_network/architecture.py` — Model definitions
- `data_processor/core/neural_network/trainer.py` — Training loop
- `data_processor/core/neural_network/evaluator.py` — Evaluation metrics
- `data_processor/core/neural_network/persistence.py` — Model save/load

---

## Phase 4: DbC & Contract Enforcement

**Target Issues**: CQ-DBC-001, CQ-DBC-002, CQ-DBC-003, relates to #721, #729

### 4.1 Standardize DbC in shared libraries

Add precondition/postcondition assertions to all public functions in:

- `src/shared/python/model_generation/`
- `src/shared/python/signal_toolkit/`
- `src/shared/python/upstream_drift_tools/`

### 4.2 Add path validation contracts to file management tools

**Files to modify:**

- All tools in `src/tools/folder_tools/`
- `src/file_management/`

```python
def process_directory(path: Path) -> ScanResult:
    """Process a directory for analysis.

    Preconditions:
        - path.exists() and path.is_dir()
        - path is not a symlink pointing outside workspace
    """
    assert path.exists(), f"Directory does not exist: {path}"
    assert path.is_dir(), f"Not a directory: {path}"
    assert not (path.is_symlink() and not path.resolve().is_relative_to(WORKSPACE_ROOT))
```

### 4.3 Add architecture fitness tests (relates to #727)

**Create:** `tests/architecture/test_layer_boundaries.py`

Verify:

- `src/shared/` does not import from tool-specific code
- Tool `core/` modules do not import from `ui/` modules
- Calculation engines are pure (no Qt imports)

---

## Phase 5: Testing Uplift (Critical)

**Target Issues**: PP-TEST-001, PP-TEST-002, PP-TEST-003, CQ-TDD-001, relates to #756

### 5.1 Add property-based tests

**Create:** `tests/test_properties.py` using Hypothesis for:

- Signal processing functions (linearity, time-invariance)
- Model generation (URDF validity)
- Calculator functions (physical consistency)

### 5.2 Add tests for decomposed modules

After Phase 3 decompositions, create test files for each new pure-domain module:

- `tests/test_electrode_calculation_engine.py`
- `tests/test_folder_scanner.py`
- `tests/test_data_loader.py`
- `tests/test_signal_processor.py`
- `tests/test_neural_network_architecture.py`

### 5.3 Increase coverage to 25% target (relates to #756)

Priority test targets:

1. Shared library public APIs
2. Calculator core logic (separated in Phase 3)
3. Launcher factory (Phase 2.1)
4. Tool skeleton (Phase 2.2)

---

## Cross-Repository Dependencies

- **UpstreamDrift**: UpstreamDrift imports `ud-tools` via vendor submodule. All shared library changes (especially `launcher_factory.py`, `tool_skeleton/`) must maintain backward compatibility. Run UpstreamDrift's test suite after Tools changes.
- **Gasification_Model**: Also imports `ud-tools`. Same backward compatibility requirement. The `BaseCalculatorWidget` pattern in Gasification_Model should be consistent with `tool_skeleton/base_tool.py`.
- **AffineDrift**: May reference assessment utilities. Changes to shared assessment patterns should be coordinated.

---

## Success Criteria

After all phases:

- [ ] 0 print() calls in production code (excluding noqa-annotated debug utils)
- [ ] All public functions have return type hints
- [ ] 0 TODO/FIXME markers
- [ ] Stale root files cleaned up
- [ ] No file exceeds 500 lines (down from 4,386)
- [ ] Shared launcher factory replaces 30+ duplicate files
- [ ] Shared tool skeleton available for new tools
- [ ] Pure domain logic separated from Qt UI code
- [ ] Architecture fitness tests pass in CI
- [ ] Property-based tests for shared libraries
- [ ] Test coverage ≥ 25% (up from 9.3%)
- [ ] CI/CD passes on all changes
