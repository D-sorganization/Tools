# Comprehensive Decoupling Plan

**Date**: 2026-02-09
**Status**: PROPOSED
**Scope**: Internal architecture of the `Tools` repository

---

## Executive Summary

This plan identifies seven major coupling problems in the codebase and proposes concrete,
incremental fixes for each. The overarching theme is: **separate what things _are_ from
how they _find each other_**. The most pervasive issue is that modules discover their
dependencies through `sys.path` manipulation at import time, creating fragile, invisible
coupling that breaks when files move or new tools are added.

---

## Table of Contents

1. [Problem 1: sys.path Spaghetti](#1-syspath-spaghetti)
2. [Problem 2: Duplicate "Repo Root" Discovery](#2-duplicate-repo-root-discovery)
3. [Problem 3: Two Utility Namespaces](#3-two-utility-namespaces)
4. [Problem 4: Launcher Proliferation](#4-launcher-proliferation)
5. [Problem 5: Hardcoded Inline Stylesheets in GUI Components](#5-hardcoded-inline-stylesheets-in-gui-components)
6. [Problem 6: Global Singleton State Manager](#6-global-singleton-state-manager)
7. [Problem 7: Calculator GUI ↔ Core Entanglement](#7-calculator-gui--core-entanglement)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Migration Strategy](#migration-strategy)
10. [Risk Assessment](#risk-assessment)

---

## 1. sys.path Spaghetti

### Problem

Over **60 instances** of `sys.path.insert(0, ...)` are scattered across the codebase.
Every launcher script (`launch_pyqt6.py`), every test `conftest.py`, and several library
modules manually prepend paths before they can import anything. This is the single largest
source of coupling in the repository because:

- Every module implicitly depends on the _physical layout_ of the repository.
- Import behavior changes depending on which script is the entry point.
- Adding a new tool requires copying and adapting the path manipulation boilerplate.
- Tests can import a _different version_ of a module than production code if the path
  ordering is wrong.

**Evidence** (representative samples):

| File                                                                  | Path(s) Inserted                                                 |
| --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `src/pressure_drop_calculator/launch_pyqt6.py:10-11`                  | `TOOLS_ROOT / "src"`, `TOOLS_ROOT / "src" / "shared" / "python"` |
| `src/syngas_compression/launch_pyqt6.py:15-16`                        | Same two paths                                                   |
| `src/signal_processing_studio/launch_pyqt6.py:20-23`                  | Four paths                                                       |
| `src/glass_bath_fea/ui/pyqt6/main_window.py:45`                       | `TOOLS_ROOT / "src"`                                             |
| `src/tools/config_loader.py:17-19`                                    | `parent.parent / "python" / "src"`                               |
| `src/shared/python/upstream_drift_tools/utils/state_manager.py:24-26` | `parents[4] / "python" / "src"`                                  |
| `src/python/src/utils/path_setup.py:66,118`                           | Dynamic paths via `setup_python_path()`                          |

### Proposed Fix

**Make `upstream_drift_tools` a proper installed package** using the existing
`pyproject.toml`, and use `pip install -e .` during development. This eliminates all
`sys.path` manipulation.

#### Steps

1. **Consolidate package roots** in `pyproject.toml` so that `setuptools` can find all
   source packages from a single `where` directive:

   ```toml
   [tool.setuptools.packages.find]
   where = ["src/shared/python", "src/python/src", "src"]
   ```

   This is already partially configured but not consistently used.

2. **Add a `conftest.py` at the repo root** (or update the existing `pytest.ini`) to
   set `pythonpath` correctly for tests. The current `pyproject.toml` already has a
   `pythonpath` setting but it's incomplete.

3. **Remove all `sys.path.insert` / `sys.path.append` calls** from:
   - All `launch_pyqt6.py` scripts (replace with package-relative imports)
   - All `conftest.py` files in sub-tools
   - Library code (`config_loader.py`, `state_manager.py`)

4. **Replace `ensure_utils_in_path()` calls** (currently in 10+ files) with direct
   imports that work via the installed package.

5. **Update CI** to run `pip install -e ".[dev]"` before tests.

#### Priority: **CRITICAL** — This is the prerequisite for all other decoupling work.

---

## 2. Duplicate "Repo Root" Discovery

### Problem

There are **three separate implementations** of "find the repository root":

| Implementation    | Location                                                           | Strategy                                                                                    |
| ----------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `get_repo_root()` | `src/tools/launch_utils.py:12-26`                                  | Walk up 5 levels looking for `tools.json` or `.git`                                         |
| `get_repo_root()` | `src/python/src/utils/path_setup.py:16-54`                         | Walk up 20 levels looking for `.git`, `pyproject.toml`, `requirements.txt`, or `tools.json` |
| Inline logic      | `src/shared/python/upstream_drift_tools/utils/state_manager.py:24` | `Path(__file__).resolve().parents[4]`                                                       |

Each implementation has different depth limits, different marker files, and different
fallback behavior. The one in `launch_utils.py` caps at 5 levels; `path_setup.py` caps at
20 and also searches for `requirements.txt` (which could match a sub-project). The one in
`state_manager.py` hardcodes an ancestor index.

### Proposed Fix

1. **Keep exactly one `get_repo_root()` function** in `src/python/src/utils/path_setup.py`
   (the most complete implementation).
2. **Make `launch_utils.get_repo_root()` a re-export**:
   ```python
   from utils.path_setup import get_repo_root
   ```
3. **Remove the inline `parents[4]` hack** in `state_manager.py` — once the package
   is properly installed, this becomes unnecessary.
4. **Standardize the marker file list**: `.git` and `pyproject.toml` only. Drop
   `requirements.txt` and `tools.json` to avoid false positives in sub-projects.

#### Priority: **HIGH** — Low effort, eliminates a common source of subtle bugs.

---

## 3. Two Utility Namespaces

### Problem

There are **two distinct utility packages** that serve overlapping purposes:

| Package                      | Location                                        | Contains                                                                                                                                                                                                                                                                                                           |
| ---------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `utils`                      | `src/python/src/utils/`                         | `file_utils`, `path_setup`, `path_helpers`, `logging_utils`, `error_handling`, `debug_utils`, `csv_utils`, `validation`, `subprocess_utils`, `config_loader`, `compatibility`, `env_utils`, `os_utils`, `test_utils`, `integration_test_helpers`, `dependency_checker`, `quality_checker`, `plotting` (20 modules) |
| `upstream_drift_tools.utils` | `src/shared/python/upstream_drift_tools/utils/` | `logging`, `state_manager`, `unit_constants` (3 modules)                                                                                                                                                                                                                                                           |

On top of that, the `src/tools/` package has its _own_ utilities:

| Module                    | Location                            | Overlaps with              |
| ------------------------- | ----------------------------------- | -------------------------- |
| `config_loader.py`        | `src/tools/config_loader.py`        | `utils.config_loader`      |
| `dependency_utils.py`     | `src/tools/dependency_utils.py`     | `utils.dependency_checker` |
| `launch_utils.py`         | `src/tools/launch_utils.py`         | Nothing (unique)           |
| `quality_utils.py`        | `src/tools/quality_utils.py`        | `utils.quality_checker`    |
| `icon_utils.py`           | `src/tools/icon_utils.py`           | Nothing (unique)           |
| `ui_utils.py`             | `src/tools/ui_utils.py`             | Nothing (unique)           |
| `matlab_quality_utils.py` | `src/tools/matlab_quality_utils.py` | Nothing (unique)           |

The `utils` package uses a **god `__init__.py`** that eagerly imports and re-exports
~80 symbols from four sub-modules. Any consumer that does `from utils import get_logger`
triggers loading of `debug_utils` (30KB, heavy profiling machinery), `test_utils` (26KB,
test frameworks), and `error_handling`, even if none of those are needed.

### Proposed Fix

#### Phase 1: Stop the bleeding

1. **Thin out `utils/__init__.py`** — remove all eager imports. Each consumer should
   import from the specific sub-module:

   ```python
   # Before (triggers loading everything)
   from utils import get_logger

   # After (loads only logging_utils)
   from utils.logging_utils import get_logger
   ```

2. **Move `upstream_drift_tools.utils` modules into `utils`** where they belong:
   - `unit_constants.py` → `utils/unit_constants.py`
   - `state_manager.py` → `utils/state_manager.py` (remove the `safe_read_json` fallback;
     just import from `utils.file_utils`)
   - `logging.py` → merge into `utils/logging_utils.py`

#### Phase 2: Consolidate overlapping modules

| From (`src/tools/`)   | Into (`utils/`)                                                | Notes                                                                       |
| --------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `config_loader.py`    | Rename to `tools/config_loader.py` (keep — it's tool-specific) | Ensure it imports `safe_read_json` from `utils.file_utils` without fallback |
| `dependency_utils.py` | Merge into `utils/dependency_checker.py`                       | Unify the two dependency-checking APIs                                      |
| `quality_utils.py`    | Merge with `utils/quality_checker.py`                          | Combine the two quality-checking APIs                                       |

#### Priority: **HIGH** — Reduces confusion, prevents accidental import of heavy modules.

---

## 4. Launcher Proliferation

### Problem

There are **four separate launcher mechanisms**, plus a per-tool `gui_registration.py`
pattern that is partially adopted:

| Launcher                  | Tech        | Location | Status                  |
| ------------------------- | ----------- | -------- | ----------------------- |
| `Launcher.py`             | Tkinter     | Root     | Legacy / fallback       |
| `UnifiedToolsLauncher.py` | PyQt6       | Root     | Primary                 |
| `launch_tools_main.py`    | Tkinter/CTk | Root     | Data Processor specific |
| `run_tile_launcher.py`    | ?           | Root     | Tile-based              |

Each tool _also_ has its own `launch_pyqt6.py` that duplicates:

- `sys.path` setup (covered in Problem 1)
- Dependency checking (2-5 imports tested with try/except)
- `QApplication` creation and window setup
- Theme application via `setup_themed_app()`

The `gui_registration.py` pattern (using `shared.python.gui_launcher`) exists in some
tools but not others — it was introduced as a plugin-like system but adoption is
inconsistent.

### Proposed Fix

1. **Complete the `gui_registration.py` pattern** across all tools. Each tool registers
   itself with a `LaunchConfig` that specifies:
   - Module and class to import
   - Required dependencies
   - Window title and minimum size
   - Settings app name for theming

2. **Create a single `launch.py` entry point** at the repo root that:
   - Auto-discovers registered tools via `gui_registration.py` modules
   - Creates the `QApplication` once
   - Handles dependency checking centrally
   - Applies theming centrally
   - Can launch any tool by name: `python launch.py --tool "Pressure Drop Calculator"`

3. **Keep individual `launch_pyqt6.py` scripts** but make them thin wrappers:

   ```python
   from tools.launcher import launch_tool_by_name
   if __name__ == "__main__":
       launch_tool_by_name("Pressure Drop Calculator")
   ```

4. **Deprecate `Launcher.py`** (Tkinter) and `launch_tools_main.py` — they are
   maintenance burden with low value now that PyQt6 is the standard.

#### Priority: **MEDIUM** — High payoff but large surface area; do after Problems 1-3.

---

## 5. Hardcoded Inline Stylesheets in GUI Components

### Problem

`ToolCard` (`src/tools/gui/components/tool_card.py`) has **120+ lines of inline CSS
strings** hardcoded into the Python file. These styles don't participate in the theme
system (`shared.python.theme`) that every other component uses. When the user switches
themes, ToolCard stays blue-and-white.

Similar hardcoded styles exist in other components but ToolCard is the worst offender.

### Proposed Fix

1. **Register ToolCard styles with the theme system**. The `shared.python.theme`
   module already has a `stylesheets.py` that generates per-widget CSS. Add ToolCard
   entries there.

2. **Replace inline `setStyleSheet()` calls** with object names and let the theme engine
   handle colors:

   ```python
   # Before
   btn.setStyleSheet("background-color: #2196F3; ...")

   # After
   btn.setObjectName("launchButton")
   # Theme stylesheet handles: #launchButton { background-color: ... }
   ```

3. **Remove the HAS_PYQT6 guard** in `tool_card.py` — the file is only used from PyQt6
   contexts, so the guard is dead code.

#### Priority: **LOW** — Cosmetic, but addresses a violation of the established pattern.

---

## 6. Global Singleton State Manager

### Problem

`state_manager.py` (line 547) creates a **module-level singleton**:

```python
state_manager = StateManager()  # Creates dirs on import!
```

This means:

- Importing the module creates filesystem directories (`saved_states/`, etc.)
- Every module that imports `state_manager` shares the same global instance
- The `StateManager` constructor has side effects (directory creation, JSON file reading)
- Testing requires either filesystem cleanup or monkeypatching

Additionally, `StateManager` has a **try/except/try/except/inline-fallback** import chain
for `safe_read_json` and `safe_write_json` that is 30 lines of import boilerplate
(lines 20-57).

### Proposed Fix

1. **Lazy initialization**: Replace the module-level singleton with a factory function:

   ```python
   _state_manager: StateManager | None = None

   def get_state_manager(base_directory: str = "saved_states") -> StateManager:
       global _state_manager
       if _state_manager is None:
           _state_manager = StateManager(base_directory)
       return _state_manager
   ```

2. **Remove the import fallback chain** — once Problem 1 is fixed, `utils.file_utils`
   will always be importable.

3. **Inject dependencies** where `StateManager` is used in calculator/GUI code instead
   of importing the global directly.

#### Priority: **MEDIUM** — Prerequisite for reliable testing of stateful components.

---

## 7. Calculator GUI ↔ Core Entanglement

### Problem

Several calculator tools have their core logic tightly coupled to their GUI, making it
impossible to use the calculation engine without PyQt6. The coupling appears in two forms:

**Form A: Import-path coupling**. The tool's `__init__.py` re-exports from the shared
library using absolute paths:

```python
# src/pressure_drop_calculator/__init__.py
from shared.python.upstream_drift_tools.process_calculators.pressure_drop_calculator import (
    PressureDropCalculationEngine, ...
)
```

This is _good_ for decoupling (core is in shared library), but the import path is brittle
— it includes `shared.python.` as a package prefix rather than using the installed
package name `upstream_drift_tools`.

**Form B: GUI code in calculator packages**. Some calculators mix calculation logic and
UI code in the same directory tree (e.g., `pressure_drop_calculator/python/.../ui/pyqt6/`
lives alongside the core models). While the directory separation exists, the `__init__.py`
re-export blurs the line.

**Form C: Direct cross-reference via `from shared.python.`**. There are **100+ imports**
using `from shared.python.upstream_drift_tools...` or `from shared.python.theme...`.
These work only because `src/` and `src/shared/python/` are on `sys.path`. They break
if the physical path changes.

### Proposed Fix

1. **Standardize import paths** to use the installed package name:

   ```python
   # Before
   from shared.python.upstream_drift_tools.process_calculators.pressure_drop_calculator import ...

   # After
   from upstream_drift_tools.process_calculators.pressure_drop_calculator import ...
   ```

   This works once the package is properly installed (Problem 1).

2. **Standardize theme imports**:

   ```python
   # Before
   from shared.python.theme import setup_themed_app

   # After (register theme as a sub-package of upstream_drift_tools)
   from upstream_drift_tools.theme import setup_themed_app
   ```

3. **Each calculator tool should have exactly two importable units**:
   - **Core**: `upstream_drift_tools.process_calculators.<name>` — no UI dependencies
   - **GUI**: `src/<name>/python/<name>/ui/` — depends on core + PyQt6

4. **Enforce the boundary** by adding a `mypy` or import-linter rule that prevents
   `upstream_drift_tools` packages from importing PyQt6.

#### Priority: **HIGH** — This is the main enabler for cross-repository reuse (the stated

goal of the `upstream_drift_tools` library).

---

## Implementation Roadmap

### Wave 1: Foundation (Prerequisite for all else)

| #   | Task                                                   | Effort | Depends On |
| --- | ------------------------------------------------------ | ------ | ---------- |
| 1.1 | Fix `pyproject.toml` package discovery                 | S      | —          |
| 1.2 | Update CI to `pip install -e ".[dev]"`                 | S      | 1.1        |
| 1.3 | Add repo-root `conftest.py` with proper `pythonpath`   | S      | 1.1        |
| 1.4 | Remove all `sys.path.insert` from library code         | M      | 1.1, 1.2   |
| 1.5 | Remove all `sys.path.insert` from launcher scripts     | M      | 1.1, 1.2   |
| 1.6 | Remove all `sys.path.insert` from test files           | M      | 1.1, 1.3   |
| 1.7 | Delete `ensure_utils_in_path()` and all calls          | S      | 1.4        |
| 2.1 | Consolidate `get_repo_root()` to single implementation | S      | 1.4        |

### Wave 2: Library Cleanup

| #   | Task                                                | Effort | Depends On |
| --- | --------------------------------------------------- | ------ | ---------- |
| 3.1 | Thin out `utils/__init__.py` (remove eager imports) | M      | Wave 1     |
| 3.2 | Migrate `upstream_drift_tools/utils/` into `utils/` | M      | 3.1        |
| 3.3 | Merge overlapping quality/dependency checkers       | S      | 3.1        |
| 6.1 | Make StateManager lazily initialized                | S      | 3.2        |
| 6.2 | Remove import fallback chains from StateManager     | S      | Wave 1     |

### Wave 3: Import Standardization

| #   | Task                                                                                  | Effort | Depends On |
| --- | ------------------------------------------------------------------------------------- | ------ | ---------- |
| 7.1 | Convert `from shared.python.upstream_drift_tools...` → `from upstream_drift_tools...` | L      | Wave 1     |
| 7.2 | Convert `from shared.python.theme...` → package-relative or installed                 | M      | Wave 1     |
| 7.3 | Convert `from shared.python.signal_toolkit...` → installed                            | M      | Wave 1     |
| 7.4 | Convert `from shared.python.plot_engine...` → installed                               | M      | Wave 1     |
| 7.5 | Add import-linter rule: no PyQt6 in `upstream_drift_tools`                            | S      | 7.1        |

### Wave 4: Launcher Consolidation

| #   | Task                                               | Effort | Depends On |
| --- | -------------------------------------------------- | ------ | ---------- |
| 4.1 | Complete `gui_registration.py` in all tools        | M      | Wave 3     |
| 4.2 | Create unified `launch.py` entry point             | M      | 4.1        |
| 4.3 | Slim down individual `launch_pyqt6.py` scripts     | M      | 4.2        |
| 4.4 | Deprecate `Launcher.py` and `launch_tools_main.py` | S      | 4.2        |
| 5.1 | Move ToolCard styles into theme system             | S      | Wave 3     |

**Effort Key**: S = Small (< 1 hour), M = Medium (1-4 hours), L = Large (4+ hours)

---

## Migration Strategy

### Guiding Principles

1. **Never break the launcher**. At every intermediate step, `UnifiedToolsLauncher.py`
   must still work.

2. **One wave at a time**. Don't mix foundation work with import changes. Each wave
   should be a merge-able PR.

3. **Test-driven migration**. Before removing a `sys.path.insert`, verify that the
   import works through the installed package path. Run the full test suite after each
   batch of changes.

4. **Keep backward-compatible shims temporarily**. When moving a function (e.g.,
   `get_repo_root` from `launch_utils` to `utils.path_setup`), keep a re-export in the
   old location until all consumers are updated. Remove shims in a cleanup PR.

### Testing After Each Wave

```bash
# Wave 1: Verify package installation works
pip install -e ".[dev]"
python -c "from utils.file_utils import safe_read_json; print('OK')"
python -c "from upstream_drift_tools import __version__; print('OK')"
pytest tests/ -x

# Wave 2: Verify utility consolidation
python -c "from utils.logging_utils import get_logger; print('OK')"
python -c "from utils.state_manager import StateManager; print('OK')"
pytest tests/ -x

# Wave 3: Verify import standardization
python -c "from upstream_drift_tools.process_calculators.pressure_drop_calculator import PressureDropCalculationEngine; print('OK')"
pytest tests/ -x

# Wave 4: Verify launcher works
python UnifiedToolsLauncher.py --dry-run  # (if supported)
python launch.py --list-tools
```

---

## Risk Assessment

| Risk                                                    | Likelihood | Impact | Mitigation                                                                  |
| ------------------------------------------------------- | ---------- | ------ | --------------------------------------------------------------------------- |
| Breaking imports during Wave 1                          | High       | High   | Git tag before starting; keep old paths in `pythonpath` temporarily         |
| CI fails after `sys.path` removal                       | Medium     | Medium | Run full test suite locally before pushing; update CI first                 |
| Consumers in other repos (`Gasification_Model`) break   | Medium     | High   | Communicate timeline; provide migration guide for `pip install -e ../Tools` |
| `pyproject.toml` package discovery misses a sub-package | Medium     | Low    | Add explicit `include` patterns for all known packages                      |
| Merge conflicts if multiple PRs in flight               | Low        | Medium | Execute waves sequentially; one PR per wave                                 |

---

## Metrics for Success

After completing all four waves, the following should be true:

1. **Zero `sys.path.insert` calls** in the codebase (excluding test scaffolding if needed).
2. **One `get_repo_root()` implementation** with all other locations re-exporting.
3. **`utils/__init__.py` is empty** (or contains only `__version__`).
4. **All `from shared.python.` imports replaced** with installed package imports.
5. **Every calculator's core can be imported without PyQt6**:
   ```python
   # This must work in a venv without PyQt6 installed:
   from upstream_drift_tools.process_calculators.pressure_drop_calculator import PressureDropCalculationEngine
   ```
6. **Single launcher entry point** that can launch any registered tool.
7. **Full test suite passes** with only `pip install -e ".[dev]"` — no manual path setup.

---

## Relationship to Existing Plans

This plan **complements** the existing `refactoring_plan_final.md` (Architecture Plan) and
`refactoring_plan_tools.md` (Component Migration Plan). Those plans focus on _what_ to
share across repositories and _where_ to put shared components. This plan focuses on _how_
the internal wiring should work to make that sharing reliable.

Specifically:

- **refactoring_plan_final.md Phase 1** ("Shared Library Foundation") is partially
  complete — the `upstream_drift_tools` package exists. This plan's Wave 1 finishes
  the job by making it properly installable.
- **refactoring_plan_tools.md Phase 2** ("Data Processor Extraction") will be easier
  after this plan's Wave 3 standardizes import paths.
- Both plans mention "Zero UI Dependencies" in core logic — this plan's Problem 7 and
  the import-linter rule in task 7.5 enforce that boundary mechanically.
