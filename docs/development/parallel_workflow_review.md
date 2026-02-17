# Review: Parallel PyQt6/React Workflow & Fleet-Wide Migration Plan

**Date**: 2026-02-08
**Session Reference**: `session_011EG3f1Sc6ud7yBXuk6aCYx`
**Status**: REVIEW & PLAN

---

## 1. Current State Assessment

### 1.1 Shared Infrastructure (What's Been Built)

The repository has successfully established three cross-platform shared systems:

| Layer              | Python (PyQt6)                            | TypeScript (React)             | Source of Truth                            |
| ------------------ | ----------------------------------------- | ------------------------------ | ------------------------------------------ |
| **Theme System**   | `src/shared/python/theme/`                | `src/shared/typescript/theme/` | `src/shared/theme-definitions/themes.json` |
| **Plot Engine**    | `src/shared/python/plot_engine/`          | (via PlotlyConverter JSON)     | `plot_engine/specs.py` (Pydantic)          |
| **Core Libraries** | `src/shared/python/upstream_drift_tools/` | N/A                            | Python package                             |

**Architecture pattern** (already proven):

```
themes.json (canonical)
    ├── Python: ThemeManager → QSS stylesheets → PyQt6 windows
    └── TypeScript: themeDefinitions.ts → CSS variables → React components

PlotSpec (Pydantic contract)
    ├── MatplotlibRenderer → PyQt6 PlotWidget (FigureCanvas)
    └── PlotlyConverter → JSON → React <Plot /> component
```

This is a solid "write specs once, render anywhere" pattern. The `PlotSpec` Pydantic models serve as the cross-platform contract, and each renderer knows how to consume them for its respective GUI framework.

### 1.2 Tools Platform Coverage

**Dual-platform (PyQt6 + React web)** — 14 tools:
| Tool | Tauri Desktop | Notes |
|------|--------------|-------|
| data_processing | Yes | Reference implementation, full Tauri 2 |
| function_generator | Yes | Full Tauri 2 |
| acid_gas_dewpoint | No | Web only (Vite) |
| baghouse_calculator | No | Web only |
| electrode_advisor | No | Web only |
| financial_calculator | No | Web only |
| flare_calculator | No | Web only |
| pressure_drop_calculator | No | Web only |
| scrubber_calculator | No | Web only |
| steam_engine_calculator | No | Web only |
| syngas_compression | No | Web only |
| trc_vessel_designer | No | Web only |
| wgs_reactor | No | Web only |
| media_processing | No | Web only |

**PyQt6 only** — 15 tools:
| Tool | Has Core/UI Separation | Migration Complexity |
|------|----------------------|---------------------|
| c3d_viewer | Partial | Medium |
| document_processing | No | High |
| flow_rate_converter | Partial | Low |
| glass_bath_fea | **Yes** (core/, ui/) | Low |
| humanoid_builder_gui | Partial | High |
| inertia_calculator | No | Low |
| multi_param_analysis | No | Medium |
| ode_solver | Yes (python/) | Low |
| optimizer_gui | No | Medium |
| psa_package | Partial | Medium |
| scientific_modeling | Mixed | High (3D) |
| signal_processing_studio | Yes (python/) | Medium |
| syngas_water_calculator | No | Low |
| thermal_profile_predictor | No | Medium |
| urdf_builder_gui | Partial | High (3D) |

### 1.3 Structural Patterns Observed

Two directory layout patterns exist in the codebase:

**Pattern A: Canonical dual-platform** (used by newer tools)

```
src/tool_name/
  gui_registration.py      # Metadata for launcher
  launch_pyqt6.py          # PyQt6 entry point
  launch_web.py            # Web dev server entry point
  python/
    tool_name/
      __init__.py
      ui/pyqt6/main_window.py
  web/
    package.json
    src/
      App.tsx
      components/ToolComponent.tsx
      main.tsx
  tests/
```

**Pattern B: Core-separated** (used by glass_bath_fea, some shared modules)

```
src/tool_name/
  core/                    # UI-agnostic logic
    config.py
    calculations.py
  ui/pyqt6/main_window.py # UI layer
  tests/
```

**Pattern B is better for migration** because the core logic is already decoupled. Pattern A tools often have logic mixed into `main_window.py`.

---

## 2. What's Working Well

1. **`themes.json` as single source of truth**: The JSON → Python/TypeScript dual-consumer pattern is clean. 13 themes, consistent key schema.

2. **PlotSpec Pydantic contracts**: The `specs.py` hierarchy (`PlotSpec`, `SurfacePlotSpec`, `ContourPlotSpec`, `HeatmapSpec`, `HistogramSpec`, `FilterComparisonSpec`) covers the visualization needs. The dual renderer approach (`MatplotlibRenderer` / `PlotlyConverter`) is DRY.

3. **PyQt6 PlotWidget**: Drop-in `QWidget` with toolbar, export, and theme integration already built.

4. **TypeScript theme store**: Zustand-based with localStorage persistence and optional API sync. This is production-grade.

5. **CI pipeline**: `ci-standard.yml` (ruff + black + mypy + tests across 3.10/3.11/3.12) and `tauri-build.yml` for desktop apps.

---

## 3. Gaps and Issues

### 3.1 No Shared React Component Library

The Python side has `PlotWidget`, `ThemeManager`, `ThemedWindowMixin` as reusable components. The React side has no equivalent component library — each web tool rebuilds its own `<App>`, layout scaffolding, and theme wiring. There should be a `src/shared/typescript/components/` with:

- `<ThemedApp>` wrapper (applies theme, provides context)
- `<PlotlyPlot>` wrapper (consumes PlotSpec JSON, auto-themes)
- `<CalculatorLayout>` (common input panel + results panel pattern)
- Form primitives styled with theme CSS variables

### 3.2 No Shared Python Calculation Extraction for Dual-Platform Tools

Most of the 14 dual-platform tools have their calculation logic embedded in `main_window.py` (PyQt6) and **duplicated** in the React `.tsx` component or not implemented at all. The `upstream_drift_tools` package was started (calculators, thermo) but most tools haven't extracted their core logic there.

### 3.3 Inconsistent Directory Structure

Some tools use `python/tool_name/ui/pyqt6/main_window.py` (Pattern A), while others use `core/` + `ui/pyqt6/` (Pattern B), and some have logic directly at the top level. This makes automation harder.

### 3.4 No API Layer for Web Tools

The web tools are pure client-side React (Vite). For tools with complex calculations (FEA, signal processing), there's no shared FastAPI/backend pattern to leverage Python computation from the React frontend. The architecture plan mentions this (`Phase 3.3: The Headless API Wrapper`) but it hasn't been built.

### 3.5 PlotlyConverter Not Used by Web Tools

The `PlotlyConverter` exists in `shared/python/plot_engine/` but the web tools don't consume it. The React components build Plotly traces directly in TypeScript. The converter should be used via API or code generation to maintain DRY.

---

## 4. Migration Plan: Extending Parallel Workflow to All Tools

### Phase 1: Standardize the Canonical Layout (Foundation)

**Goal**: Every tool follows the same directory pattern.

```
src/tool_name/
  gui_registration.py          # Metadata (name, category, entry points)
  launch_pyqt6.py              # PyQt6 standalone launcher
  launch_web.py                # Web dev server launcher (when web/ exists)
  core/                        # UI-AGNOSTIC LOGIC (new for most tools)
    __init__.py
    calculator.py              # Pure computation functions
    models.py                  # Pydantic models for inputs/outputs
    validators.py              # Input validation
  python/
    tool_name/
      ui/pyqt6/main_window.py  # PyQt6 GUI (imports from core/)
  web/                          # React GUI (when applicable)
    package.json
    src/
      App.tsx
      components/
  tests/
    test_core.py               # Tests for core/ (no GUI deps)
    test_gui.py                # Tests for GUI (optional)
```

**Key rule**: `core/` has **zero** GUI imports (`PyQt6`, `tkinter`, `React`). It depends only on numpy/scipy/pandas/pydantic.

#### Priority order for core extraction:

**Tier 1 — Low effort, high value** (already partially separated or simple calculators):

1. `flow_rate_converter` — pure math, trivial extraction
2. `syngas_water_calculator` — pure calculation
3. `inertia_calculator` — pure math
4. `ode_solver` — solver logic extractable
5. `glass_bath_fea` — already has `core/` directory

**Tier 2 — Medium effort** (logic mixed with UI but identifiable): 6. `multi_param_analysis` — plotting + analysis logic 7. `psa_package` — calculation engine 8. `thermal_profile_predictor` — prediction models 9. `signal_processing_studio` — signal toolkit already in shared

**Tier 3 — High effort** (deeply coupled or 3D): 10. `c3d_viewer` — 3D rendering intertwined 11. `document_processing` — file I/O heavy 12. `humanoid_builder_gui` — mesh generation (partially in shared already) 13. `optimizer_gui` — optimization loops 14. `scientific_modeling` — multiple sub-projects, 3D 15. `urdf_builder_gui` — 3D/URDF generation (partially in shared)

### Phase 2: Shared React Component Library

**Goal**: Create `src/shared/typescript/components/` to eliminate React boilerplate duplication.

```
src/shared/typescript/
  theme/                    # (existing)
  components/               # NEW
    ThemedApp.tsx            # App wrapper: ThemeProvider + layout chrome
    PlotlyPlot.tsx           # Wraps react-plotly.js, consumes PlotSpec JSON
    CalculatorLayout.tsx     # Two-panel: inputs (left) + results/plots (right)
    InputField.tsx           # Themed number/text input
    ResultsTable.tsx         # Themed results display
    ExportButton.tsx         # CSV/PNG export
  hooks/                    # NEW
    usePlotSpec.ts           # Fetch/manage PlotSpec from API
    useCalculation.ts        # Generic calc runner hook
```

Each new web tool becomes ~50 lines of glue code:

```tsx
import { ThemedApp, CalculatorLayout, PlotlyPlot } from "@shared/components";

function FlareCalculator() {
  const [results, calculate] = useCalculation("/api/flare/calculate");
  return (
    <CalculatorLayout
      inputs={<FlareInputs onSubmit={calculate} />}
      results={<PlotlyPlot spec={results.plotSpec} />}
    />
  );
}
```

### Phase 3: Optional API Layer (for compute-heavy tools)

For tools where Python computation is complex (FEA, signal processing, optimization), add a lightweight FastAPI wrapper:

```
src/tool_name/
  api/                    # NEW - optional
    __init__.py
    routes.py             # FastAPI router importing from core/
  core/                   # Already extracted
    calculator.py
```

The web tool calls the Python API instead of reimplementing calculations in TypeScript. This is essential for:

- `glass_bath_fea` (FEA solver)
- `signal_processing_studio` (scipy signal processing)
- `optimizer_gui` (scipy.optimize)
- `scientific_modeling` (physics simulations)
- `multi_param_analysis` (statistical analysis)

Simple calculators (flow rate, inertia, syngas) can stay client-side in TypeScript.

### Phase 4: Tauri Desktop Wrappers

Once a tool has a web frontend, adding Tauri is mechanical:

1. Add `web/src-tauri/` with `tauri.conf.json` and `main.rs`
2. Register in `tauri-build.yml` matrix
3. Optional: Tauri commands for file system access

**Priority for Tauri**: Only tools that benefit from desktop (file I/O, offline use). The two current Tauri apps (data_processor, function_generator) are the right starting point.

---

## 5. Implementation Sequence

```
Phase 1A: Core extraction for Tier 1 tools (5 tools)
    ↓
Phase 1B: Core extraction for Tier 2 tools (4 tools)
    ↓
Phase 2: Shared React component library
    ↓
Phase 1C: Add web/ to Tier 1 tools using shared components
    ↓
Phase 3: API layer for compute-heavy Tier 2/3 tools
    ↓
Phase 1D: Core extraction + web for Tier 3 tools
    ↓
Phase 4: Tauri wrappers for selected tools
```

### Estimated Scope

| Phase | Tools Affected | New Files             | Key Deliverable                        |
| ----- | -------------- | --------------------- | -------------------------------------- |
| 1A    | 5              | ~15 core modules      | `core/` packages for simple tools      |
| 1B    | 4              | ~12 core modules      | `core/` packages for medium tools      |
| 2     | All web tools  | ~8 shared components  | `shared/typescript/components/`        |
| 1C    | 5              | ~25 (web scaffolding) | React UIs for Tier 1 tools             |
| 3     | 5              | ~10 API routers       | FastAPI wrappers for heavy computation |
| 1D    | 5              | ~20 core + web        | Full dual-platform for remaining tools |
| 4     | Select 3-5     | ~15 (Tauri configs)   | Desktop builds                         |

---

## 6. Conventions to Enforce

1. **Every `core/` module must have Pydantic input/output models** — these serve as the API contract between Python and TypeScript.
2. **PlotSpec for all visualizations** — no direct matplotlib or Plotly calls from tool code. Tools produce `PlotSpec`, renderers consume it.
3. **`gui_registration.py` is mandatory** — this is how the launcher discovers tools.
4. **Tests for `core/` are mandatory** — TDD per AGENTS.md. GUI tests are optional.
5. **`upstream_drift_tools` for truly shared calculations** — tool-specific logic stays in `tool_name/core/`, only generic calculations (unit conversion, steam tables, etc.) go to the shared package.

---

## 7. CI/CD Updates Needed

1. **Add TypeScript linting to `ci-standard.yml`** for web tools (eslint + tsc type-check)
2. **Expand `tauri-build.yml` matrix** as new Tauri apps are added
3. **Add a shared component build check** — ensure `src/shared/typescript/` compiles
4. **Consider a `web-check.yml`** workflow for npm builds across all web tools

---

## 8. Risks and Mitigations

| Risk                                                      | Mitigation                                                            |
| --------------------------------------------------------- | --------------------------------------------------------------------- |
| Core extraction breaks existing PyQt6 GUIs                | Extract then re-import; run existing tests after each extraction      |
| React component library becomes too opinionated           | Keep primitives generic, allow per-tool customization                 |
| API layer adds deployment complexity                      | Make it optional; simple calculators stay client-side                 |
| 3D tools (URDF, C3D, solar system) don't translate to web | Use Three.js/WebGL for web 3D; accept some divergence                 |
| Shared TypeScript package import paths                    | Use TypeScript path aliases (`@shared/*`) in each web tool's tsconfig |
