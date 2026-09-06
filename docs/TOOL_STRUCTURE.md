# TOOL_STRUCTURE.md — Canonical Tool Organization

## Overview

This document defines the canonical structure for tools in the Tools monorepo. It serves as the single source of truth for Phase 2 refactoring work (issue #2405).

## Current State (As-Is, Pre-Refactor)

### Tool Locations

Tools are scattered across three primary locations:

1. **src/<tool_name>/** — Most tools (19 tools)

   - Direct children of src/ directory
   - Examples: c3d_viewer, financial_calculator, rotation_converter, urdf_builder_gui

2. **src/<category>/<tool_name>/** — Category-grouped tools (2 tools)

   - data_processing/data_processor (registered in manifest)
   - document_processing/pdf_renamer (registered in manifest)

3. **Unregistered Tools** — No manifest entry (1 tool)
   - src/lower_body_model/ (has launch_pyqt6.py and gui_registration.py but NOT in tool_manifest.yaml)

### Registration and Plugin System

**Manifest:** `src/shared/python/gui_launcher/tool_manifest.yaml`

- Single source of truth for tool registration
- Contains 20 tools (missing: lower_body_model)
- Schema: tool_name, name, description, category, icon, pyqt6, web, engine configs
- Centralized GUI metadata (resolves issue #1863 duplication)

**Deprecated Files (Still Present):**

- gui_registration.py — 21 copies (one per tool) — contains duplicate manifest data
- launch_pyqt6.py — 21 copies (one per tool) — entry point to PyQt6 UI

### Namespace Collisions

**Critical Duplicates (21 of each):**

- launch_pyqt6.py — appears once per tool
- gui_registration.py — appears once per tool

**Secondary Duplicates (3-5 of each):**

- core.py — signal_toolkit, upstream_drift_tools (2 instances), rotation_converter, pdf_renamer
- models.py — tile_launcher, chat, notes
- main_window.py — 19 instances across tools (deeply nested, low collision risk)

**No collision risk:** main_window.py because each is nested in tool-specific paths (e.g., c3d_viewer/python/c3d_viewer/ui/pyqt6/main_window.py)

### Shared Libraries and Infrastructure

**Location:** src/shared/python/ (16 items)

**Categories:**

- **Platform Infrastructure:** gui_launcher
- **Backend Services:** calc_backend, humanoid_character_builder, model_generation, upstream_drift_tools
- **Shared Libraries:** signal_toolkit, plot_engine, plot_theme, rotation_transforms, programmatic_pid, theme
- **Applications:** chat, notes
- **Tooling:** scripting, tests, data_processing (partial)

These are NOT tools themselves but shared code consumed by multiple tools.

### Platform Utilities

**Location:** src/tools/ (not a tool, but tool utilities)

**Contents:**

- launch_utils.py — PyQt6 launch infrastructure
- gui/ — GUI components and windows
- icon_utils.py, ui_utils.py — UI utilities
- config_loader.py, dependency_utils.py — configuration
- matlab_quality_utils.py, quality_utils.py — code quality
- mypy_autofix_agent.py — type checking automation
- scientific_auditor.py — scientific code validation

## Canonical Structure (Target, Phase 2+)

### 1. Tool Directory Layout

```
src/<category>/<tool_name>/
├── __init__.py                      # Entry point, __all__ exports
├── launch_pyqt6.py                  # DEPRECATED — delegates to gui_launcher
├── gui_registration.py              # DEPRECATED — remove in Phase 3
├── README.md                        # Tool documentation
├── python/                          # Implementation (optional)
│   └── <tool_name>/
│       ├── __init__.py
│       ├── core.py                  # Core logic
│       ├── models.py                # Data models
│       ├── api.py                   # If has web API
│       └── ui/                      # UI code (optional)
│           ├── pyqt6/
│           │   ├── __init__.py
│           │   └── main_window.py
│           └── web/
│               ├── __init__.py
│               └── app.py
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_core.py
│   └── test_gui.py
├── web/                             # Web UI (optional)
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   └── ...
│   └── dist/
└── data/                            # Static assets (optional)
    └── examples.yaml
```

### 2. Rationale for Structure

**Why this structure?**

1. \***\*init**.py (Entry Point)\*\*

   - Declares tool name, version, description
   - Exports public API
   - Avoids name collisions by being tool-specific
   - Example: `src/financial_calculator/__init__.py` vs. `src/rotation_converter/__init__.py`

2. **launch_pyqt6.py & gui_registration.py (Deprecated)**

   - Currently: separate copy per tool
   - Purpose: override point for tool-specific launch logic
   - Future: delegate to gui_launcher, read from manifest
   - Timeline: keep in Phase 2, deprecate in Phase 3, remove in Phase 4

3. **python/<tool_name>/ (Implementation)**

   - Nested package prevents import collisions
   - All tool-specific modules under this tree
   - core.py, models.py are now tool-scoped: `<tool_name>.core`, not global `core`
   - Enables clean separation: `from financial_calculator.core import FinancialModel`

4. **web/ (Optional Web UI)**

   - Isolated Node.js/React/Vue project
   - Separate tooling (npm, webpack, etc.)
   - Does not interfere with Python package structure

5. **tests/ (Test Suite)**
   - Colocated with implementation
   - pytest discovers from any directory
   - Avoids circular imports (tests not in package)

### 3. Specific Variants

**Variant A: Simple Tool (cli-only or utility)**

```
src/flow_rate_converter/
├── __init__.py
├── python/
│   └── flow_rate_converter/
│       ├── __init__.py
│       ├── core.py
│       └── models.py
└── tests/
    └── test_core.py
```

**Variant B: Tool with PyQt6 UI**

```
src/financial_calculator/
├── __init__.py
├── python/
│   └── financial_calculator/
│       ├── __init__.py
│       ├── core.py
│       ├── models.py
│       └── ui/
│           └── pyqt6/
│               ├── __init__.py
│               └── main_window.py
├── web/
│   ├── package.json
│   └── src/
│       └── App.jsx
└── tests/
    ├── test_core.py
    └── test_gui.py
```

**Variant C: Tool with Backend Engine**

```
src/ode_solver/
├── __init__.py
├── python/
│   └── ode_solver/
│       ├── __init__.py
│       ├── engine.py           # Main computation engine
│       └── ui/
│           └── pyqt6/
│               └── main_window.py
└── tests/
    └── test_engine.py
```

### 4. Shared Libraries (src/shared/python/)

**Rules:**

- NOT tools themselves
- Imported BY multiple tools
- Examples: signal_toolkit, plot_engine, upstream_drift_tools

**Structure:**

```
src/shared/python/<library_name>/
├── __init__.py
├── <module1>.py
├── <module2>.py
└── tests/
```

**Do NOT use `python/` subdirectory** for shared libraries (only for tools).
Reason: shared libraries are utilities meant to be imported, not standalone tools.

### 5. Platform Services (src/shared/python/)

**Examples:** gui_launcher, calc_backend, humanoid_character_builder

These follow tool structure within shared/python:

- Have **init**.py, tests/, potentially web/
- Centralized because they're shared across multiple tools
- NOT moved to src/ root because they're not UI tools

## Import Paths (Post-Refactor)

### Tool-Internal Imports

```python
# GOOD (after refactor to python/ subdirectory)
from financial_calculator.core import FinancialModel
from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorMainWindow

# BAD (global collision risk, old structure)
from core import FinancialModel  # which tool's core?
```

### Cross-Tool Shared Libraries

```python
# GOOD
from signal_toolkit import FFT, Spectrogram
from plot_engine import plot_time_series
from upstream_drift_tools.calculators import FluidCalculator

# BAD (circular dependency risk)
from financial_calculator.core import something_unrelated_to_tools
```

## Categories (Recommended)

Reorganize tools by category in src/ (Phase 3+):

```
src/
├── analysis/          # Multi-parameter analysis, etc.
├── biomechanics/      # C3D viewer, motion capture
├── calculators/       # Financial, pressure drop, steam, inertia
├── data_processing/   # PDF renamer, data processor
├── engineering/       # P&ID generator, URDF builder, Vessel drafter
├── optimization/      # Optimizer, ODE solver, function generator
├── robotics/          # Humanoid builder, rotation converter
├── signal_processing/ # Signal studio, flow rate converter
├── simulation/        # Pendulum simulator, RRT path planner
├── tools/             # Folder tools, low-level utilities
└── shared/
    └── python/        # Platform libraries and services
```

**Decision Point:** This requires coordination with downstream repos. Defer to Phase 3.

## Manifest (tool_manifest.yaml)

**Current Status:** Implemented, 20/21 tools registered

**Missing:** lower_body_model

**Format (per-tool):**

```yaml
- tool_name: financial_calculator # unique ID (snake_case)
  name: Financial Calculator # display name
  description: > # help text
    Comprehensive financial modeling...
  category: Process Simulation
  icon: calculator
  pyqt6: # if has PyQt6 UI
    module: financial_calculator.ui.pyqt6.main_window
    class: FinancialCalculatorMainWindow
    dependencies:
      - PyQt6
      - numpy
    settings_app: FinancialCalculator
  web: # if has web UI
    port: 5173
    auto_open_browser: false
  engine: # if has backend engine
    module: upstream_drift_tools.process_calculators.financial_calculator
    class: FinancialModelCalculator
```

**Responsibilities:**

- gui_launcher reads this to populate launcher menu
- Dynamic discovery of tools
- Removes need for tool-specific gui_registration.py
- Single source of truth (resolves issue #1863)

## Refactoring Timeline

### Phase 2.1 (Audit & Documentation) — ✓ THIS PHASE

- Scan and document current structure
- Create canonical structure definition (this doc)
- Identify collisions and issues
- Create refactoring plan

### Phase 2.2 (Register Missing Tools)

- Add lower_body_model to tool_manifest.yaml
- Verify manifest completeness
- Update gui_launcher to handle all registered tools

### Phase 2.3 (Deprecate Legacy Files)

- Mark gui_registration.py as deprecated in all tools
- Mark launch_pyqt6.py as deprecated in all tools
- Update documentation to use manifest instead

### Phase 3 (Optional: Reorganize by Category)

- Create src/analysis/, src/calculators/, etc. directories
- Move tools to category directories
- Coordinate with downstream repos (UpstreamDrift, Gasification_Model)
- Update import paths in all files

### Phase 4 (Remove Deprecated Files)

- Delete all gui_registration.py files
- Delete all launch_pyqt6.py files
- Update tool loading to use manifest exclusively

## Impact Analysis

### Breaking Changes: None (Phase 2)

- Only documentation and deprecation
- No changes to public APIs
- No changes to import paths
- No changes to tool behavior

### No Immediate Refactoring

- Tools stay in current locations
- Nested python/ subdirectories already in use
- Manifest already centralized

### Future Impact (Phase 3+)

- Moving tools by category requires downstream coordination
- Import paths may change: `from src.analysis.multi_param_analysis` etc.
- These changes trigger breaking change notifications to UpstreamDrift and Gasification_Model

## Validation Checklist

For each tool, verify:

- [ ] Tool is in tool_manifest.yaml
- [ ] Tool has **init**.py at root level
- [ ] Tool has launch_pyqt6.py (for now; deprecated in Phase 3)
- [ ] Tool has gui_registration.py (for now; deprecated in Phase 3)
- [ ] Tool has python/<tool_name>/ subdirectory (where implementation lives)
- [ ] Tool's main_window.py (if GUI) matches manifest module path
- [ ] Tool's README.md documents tool purpose and usage
- [ ] Tool's tests/ directory exists and has test coverage

## See Also

- Issue #2405 — Inconsistent module structure and namespace collisions
- Issue #1863 — Duplicate GUI_INFO dict in 20 files (resolved by manifest)
- CLAUDE.md — Project-wide coding standards and governance
- tool_manifest.yaml — Complete tool registry
