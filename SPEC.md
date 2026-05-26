# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-05-23

  This is the canonical specification template for all repositories in the
  D-sorganization fleet. Every repo MUST have a SPEC.md at its root.

  INSTRUCTIONS:
  1. Copy this template to the root of your repository as SPEC.md
  2. Fill in every section — leave nothing as "[TODO]"
  3. Keep this document updated with every PR that changes functionality
  4. CI will block merges if SPEC.md is stale (source changed but spec didn't)

  AUDIENCE: This document is designed for both human developers AND AI agents.
  Write clearly, use concrete examples, and avoid ambiguity.
-->

## 1. Identity

| Field                   | Value                                      |
| ----------------------- | ------------------------------------------ |
| **Repository Name**     | `Tools`                                    |
| **GitHub URL**          | `https://github.com/D-sorganization/Tools` |
| **Owner**               | D-sorganization                            |
| **Primary Language(s)** | Python 3.11+, Rust, JavaScript, TypeScript |
| **License**             | MIT                                        |
| **Current Version**     | N/A                                        |
| **Spec Version**        | 1.1.204                                    |
| **Last Spec Update**    | 2026-05-24                                 |

## 2. Purpose & Mission

Comprehensive monorepo housing 45+ utility tools for data processing, scientific computing, process engineering, and automation. This is the central tooling hub for the D-sorganization fleet, providing modular engineering calculation tools with PyQt6 GUIs, FastAPI web services, Rust numerical kernels, and a unified launcher with plugin architecture for extensibility.

## 3. Goals & Non-Goals

### Goals

- Deliver 45+ modular engineering calculation tools with consistent interfaces
- Provide PyQt6 GUI launcher (UnifiedToolsLauncher) for tool discovery and execution
- Implement plugin discovery and loading system for extensibility
- Build Rust numerical kernels for performance-critical operations
- Offer FastAPI web interfaces for programmatic and integration access
- Provide MATLAB scientific code integration and wrappers
- Maintain fleet theme system for consistent UI across all tools
- Support multiple Python versions (3.11, 3.12) with comprehensive test matrix

### Non-Goals

- Not application-specific business logic (each application repo owns its logic)
- Not a framework (Tools is a collection, not an opinionated framework)

## 4. Architecture Overview

### System Context

Tools is the central utility hub for the D-sorganization fleet. Other repos depend on Tools for:

- Scientific and numerical computations (via Rust kernels and numpy/scipy)
- Data processing pipelines (pandas, specialized modules)
- Document and media processing (PDF, audio, video tools)
- Web service capabilities (FastAPI)
- GUI building blocks (PyQt6 theme system, shared widgets)
- Shared chat contracts, including optional terminal-agent shell/provider
  descriptors for project-scoped agent sessions
- Shared AI chat memory persists auditable `user_memory.json` prompt context,
  resolves project-root `AGENTS.md` instructions, and can extract explicit
  user preferences from archived conversations without treating archives as
  model training data
- Shared chat services expose a launcher-facing `condense_to_memory` API that
  writes explicit user memory candidates through the shared memory manager and
  reports processed, missing, and inserted conversation counts
- Shared chat Qt dock imports expose subprocess-backed PyQt6 runtime
  diagnostics so hosts and tests can report broken Qt DLL/runtime installs
  without crashing the importing process
- Project-scoped terminal-agent runtime coordination for shared chat provider
  processes
- Shared chat WebSocket terminal-session actions for start/input/resize/events
  and stop lifecycle control
- Shared chat dock terminal mode with shell/provider selectors and terminal
  session input routing
- Shared chat dock close and terminal stop controls, with terminal
  shell/provider dropdowns populated from the shared provider registry
- Shared chat dock terminal lifecycle controls disable duplicate starts,
  enable Stop only for active sessions, and lock shell/provider choices while
  a terminal session is pending or active
- Shared chat dock shutdown treats intentional widget close as terminal for
  WebSocket reconnects so launcher-hosted Sidekick chat surfaces do not revive
  after close while unexpected disconnects still retry
- P1AM SCADA firmware control-loop contracts fail closed on corrupt SCADA or
  flash routing, non-finite process values, invalid PID timing, and non-finite
  analog outputs instead of invoking runtime `assert()` abort paths on the PLC
- Sidekick tabs declare versioned per-tab settings schemas and persist
  materialized settings by stable tab or duplicate instance id behind the
  selected-tab settings action
- Shared chat history rows use wrapped readable item widgets with transparent
  icon-only archive, restore, and delete controls available without right-click
- Shared chat dock close control lives in the persistent status header instead
  of the terminal provider control row
- Shared unified tools sidebar widgets provide optional dockable/tear-off host
  integration for project file browsing, workspace variables, chat, terminal,
  calculator, unit conversion, and notes tabs
- Sidekick runtime tabs embed real utility surfaces for chat status, workspace
  Python execution, symbolic calculator evaluation, and project-persistent
  notes instead of placeholder panels
- Sidekick sidebar configuration extends the shared sidebar with persisted
  left/right docking, minimized state, tab order, hidden tabs, popped-out tab
  tracking, duplicate tab instances, and host-provided tab definitions
- Sidekick tab bars expose per-tab context menus for left/right dock moves,
  pop-out, duplicate, close, and sidebar minimization actions without relying
  on a separate toolbar
- Sidekick tab display names can be customized per stable tab id, persisted in
  sidebar state, reset to defaults, and resolved consistently for docked tab
  labels and pop-out window titles
- Sidekick design tokens provide reusable QSS and CSS-variable mappings, with
  stable Qt object names/selectors for downstream host styling
- Provider-contract CI includes non-GUI coverage for the deprecated
  `upstream_drift_tools` compatibility shim so legacy imports keep resolving
  to canonical Sidekick APIs during the migration window
- Shared Qt theme stylesheets use relative control typography and minimum tab
  widths so application-level zoom scales shared sidebar and launcher text more
  consistently
- Sidekick web/Tauri styling aliases expose the same `--sidekick-*` token names
  as the PyQt sidebar contract, mapped onto the shared `--theme-*` variables
- Shared TypeScript theme helpers generate and apply dynamic `--sidekick-*`
  variables from the same canonical theme definitions used by React/Tauri hosts
- Sidekick host factory/install helpers accept shared theme names and resolve
  them through the canonical design-token bridge when explicit token overrides
  are not supplied
- Sidekick widgets can reapply canonical shared themes or explicit design-token
  sets at runtime without reconstructing the dock/sidebar instance
- Sidekick terminal tabs inherit resolved Sidekick design tokens by default and
  support validated terminal-scoped custom foreground, background, cursor,
  selection, and ANSI palette colors without changing the global sidebar theme
- Sidekick calculator startup imports are validated, persisted in sidebar
  state, optional-dependency safe for NumPy/SciPy defaults, and surfaced as
  structured UI diagnostics when a configured dependency is unavailable
- Sidekick state profiles persist named sidebar snapshots below a host-provided
  storage root, validate path-safe profile names, reject malformed loads without
  mutating the active sidebar, and require an explicit confirmation token before
  clearing profile data
- Sidekick default tabs now ship shared help metadata that stays import-safe in
  headless contexts, exposes a Help action from the shared tab context menu,
  and standardizes hover hints for compact runtime controls
- Sidekick calculator tabs expose a bounded workspace command line for
  explicit local/global variable assignment, inspection, deletion, clear,
  and load/save workflows without falling back to arbitrary terminal execution
- Sidekick calculator workspaces keep local variables isolated from shared
  global state by default, support explicit local-to-global promotion, and
  persist local/global JSON workspace files through shared scoped helpers
- Sidekick can lazily expose the existing Data Processor UI as an optional
  first-class tab, degrade to a clear placeholder when its heavier runtime
  dependencies are missing, and export validated selected results into the
  shared workspace registry
- Sidekick notes use shared markdown-backed note cards with path-safe IDs,
  validated per-card colors, persisted board background settings, reversible
  recycle-bin deletion, and legacy `project.notes.txt` migration
- Sidekick can lazily expose the Function Generator as an optional tab,
  launch the PyQt6 generator through an import-safe wrapper, and provide
  compact help/design-token metadata for downstream sidebar hosts
- Sidekick sidebar instances expose `open_tab(tab_id)` for downstream launcher
  menu routing, including compatibility for the `os_terminal` launcher id and
  hidden-tab materialization before focus
- Data-driven shared chat terminal-provider descriptors for Claude Code, Codex,
  Cline CLI, Gemini CLI, Antigravity CLI (`agy`), and GitHub CLI, including
  probe command metadata with diagnostic redaction helpers
- Shared source-tree logging and environment helpers keep AI adapter and chat
  service imports self-contained for downstream consumers that install or
  vendor only the shared Tools modules
- Plugin system for extending functionality

No repo is required to use Tools, but it provides optional high-value integrations.

### Module Map

```
Tools/
├── src/
│   ├── python/                     # Core infrastructure and shared utilities
│   │   ├── plugin_system/          # Plugin discovery and loading
│   │   ├── shared_utilities/       # Common functions, decorators, helpers
│   │   └── infrastructure/         # Base classes, interfaces
│   ├── tools/                      # Tool implementations
│   │   ├── calculator/             # Engineering calculators
│   │   ├── converter/              # Unit and format converters
│   │   └── [40+ tool directories]
│   ├── data_processing/            # Data processing pipelines
│   │   ├── pipelines/
│   │   ├── transformers/
│   │   └── validators/
│   ├── document_processing/        # Document utilities
│   │   ├── pdf_tools/
│   │   ├── text_extractors/
│   │   └── formatters/
│   ├── media_processing/           # Audio/video tools
│   │   ├── audio/
│   │   └── video/
│   ├── scientific_modeling/        # Modeling and simulation
│   │   ├── thermal/
│   │   ├── mechanical/
│   │   └── chemical/
│   ├── web_applications/           # Web dashboards and APIs
│   │   ├── api/                    # FastAPI services
│   │   ├── dashboards/             # Web UIs
│   │   └── integrations/
│   └── verification/               # Testing utilities
├── rust_core/                      # Rust numerical kernels
│   ├── math-primitives/            # Fundamental math operations
│   └── tools-core/                 # Core tool runtime
├── matlab/                         # Scientific MATLAB code
├── UnifiedToolsLauncher.py         # Primary GUI launcher entry point
├── tests/                          # 197 test files
│   ├── unit/
│   ├── integration/
│   ├── acceptance/
│   └── conftest.py
├── .github/workflows/              # 50 CI/CD workflows
└── SPEC.md                         # This file

```

### Key Components

| Component                | Location                                                                               | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UnifiedToolsLauncher     | `UnifiedToolsLauncher.py`                                                              | PyQt6 GUI for tool discovery and execution                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Shared Chat              | `src/shared/python/chat/`                                                              | Shared chat contracts, dock widgets, terminal-agent runtime boundaries, UI-agnostic default provider descriptors for Claude Code, Codex, Cline CLI, Gemini CLI, and GitHub CLI, and prompt-time AI memory context backed by explicit archived-chat preference extraction                                                                                                                                                                                                             |
| Unified Tools Sidebar    | `src/shared/python/upstream_drift_tools/ui/tools_sidebar/`                             | Optional Qt dock widget contract for downstream host applications, including project-scoped file browsing, workspace registry/state persistence, reusable Sidekick design tokens, stable stylesheet selectors, embedded Sidekick runtime widgets for chat status, workspace Python execution, symbolic calculator evaluation, project notes, unit conversion, and an optional lazy Data Processor tab with workspace export, plus configurable tabbed utility surfaces for host apps |
| GUI Launcher Web Helpers | `src/shared/python/gui_launcher/launcher_web.py`                                       | Focused React/Vite launcher process helpers shared by direct web launch scripts and the unified GUI launcher                                                                                                                                                                                                                                                                                                                                                                         |
| Plugin System            | `src/python/plugin_system/`                                                            | Discover, load, and manage plugins                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Shared Utilities         | `src/python/shared_utilities/`                                                         | Common functions, decorators, error handling                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Signal Toolkit           | `src/shared/python/signal_toolkit/`                                                    | Shared signal-processing primitives, including adaptive filters implemented in `adaptive_filter.py`, waveform generators that reject underspecified sample arrays and non-positive frequencies, and re-exports through the package and legacy `filters` module                                                                                                                                                                                                                       |
| Pressure Drop Calculator | `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/` | Facade-driven gas pressure-drop workflows with extracted API, validation, reference, results, and engine-domain helper modules                                                                                                                                                                                                                                                                                                                                                       |
| Model Generation API     | `src/shared/python/model_generation/api/`                                              | Route facade with framework-specific Flask and FastAPI adapters behind a compatibility shim, plus repository download helpers that require HTTPS downloads and validate archive and mesh paths to prevent traversal                                                                                                                                                                                                                                                                  |
| Engineering Tools        | `src/tools/`                                                                           | 45+ specialized calculation and processing tools                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Data Processing          | `src/data_processing/`                                                                 | Pipelines, transformers, validators, and facade-based data-processor core modules for exporter, ANOVA, vectorized filter workflows, Butterworth filters with explicit or time-derived sample rates, checked normalize/standardize transforms, operator-whitelisted row filtering, and pickle-safe file I/O defaults                                                                                                                                                                  |
| Document Processing      | `src/document_processing/`                                                             | PDF extraction, text processing                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Media Processing         | `src/media_processing/`                                                                | Audio and video utilities                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Scientific Modeling      | `src/scientific_modeling/`                                                             | Thermal, mechanical, chemical simulations                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Web Services             | `src/web_applications/api/`                                                            | FastAPI endpoints and integrations                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Unit Converter WSGI      | `src/web_applications/unit_converter/`                                                 | Flask web application with a production WSGI entry point; debug mode is development-only and gated by `FLASK_DEBUG`                                                                                                                                                                                                                                                                                                                                                                  |
| Rust Kernels             | `rust_core/`                                                                           | High-performance mathematical operations, including standard atmosphere calculations that require finite, non-negative altitudes and a canonical full-precision universal gas constant                                                                                                                                                                                                                                                                                               |
| MATLAB Integration       | `matlab/`                                                                              | Wrapped MATLAB scientific code                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Fleet Theme System       | `src/shared/python/theme/`                                                             | Shared PyQt6 theme infrastructure for fleet UI parity, including built-in/custom themes, generated QSS, theme-aware mixins, icon colorization, Matplotlib color synchronization, responsive text-aware sizing helpers, application-level zoom controls, and compatibility exports for downstream tools                                                                                                                                                                               |

### Production-Readiness Hardening

- Generated data-processing batch scripts serialize input glob patterns and
  output directories with Python literal-safe formatting, write CSV outputs via
  temporary files followed by atomic replace, aggregate per-file failures, and
  exit non-zero when any file fails. Parallel generated scripts bound worker
  count by default and allow `DATA_PROCESSOR_BATCH_MAX_WORKERS` overrides.
- Shared pandas formula entry points validate expressions before calling
  `DataFrame.eval`. The allowlist accepts column names, numeric/boolean
  constants, arithmetic, boolean operators, and comparisons; it rejects function
  calls, attribute access, indexing, unknown names, overly long formulas, overly
  complex ASTs, and unbounded exponent expressions. `numexpr` remains an
  optional accelerator with the existing Python-engine fallback.
- Model-generation mesh inertia upload handlers reject empty payloads,
  unsupported mesh filename suffixes, and payloads above the configured 10 MiB
  limit before parser handoff. Temporary mesh files are deleted in cleanup
  paths, and malformed parser failures are normalized into API error responses.
- MakeHuman humanoid mesh generation uses safely serialized export paths inside
  generated scripts, validates modifier keys and finite numeric values before
  script creation, rejects non-directory output paths, and keeps
  `mesh_generator_makehuman.py` as a compatibility shim over the extracted
  `_makehuman_generator.py` implementation.

## 5. Desired Functionality

### Core Features

| #   | Feature                             | Status | Description                                                                                   |
| --- | ----------------------------------- | ------ | --------------------------------------------------------------------------------------------- |
| F1  | UnifiedToolsLauncher (PyQt6 GUI)    | ✅     | Main entry point with tool discovery, search, favorites, and launch                           |
| F2  | 45+ engineering calculation tools   | ✅     | Diverse tools for calculations, conversions, analysis                                         |
| F3  | Rust math primitives                | ✅     | Performance-critical numerical operations in Rust                                             |
| F4  | Shared upstream_drift_tools library | ✅     | Common utilities for drift detection and analysis                                             |
| F5  | Plugin discovery system             | ✅     | Auto-discover tools via plugin registry, support dynamic loading                              |
| F6  | FastAPI web interfaces              | 🔄     | RESTful API for programmatic access to tools                                                  |
| F7  | MATLAB scientific tools             | ✅     | Integration with MATLAB code and wrappers                                                     |
| F8  | Fleet theme system                  | ✅     | Consistent theming across all PyQt6 GUIs                                                      |
| F9  | Lower-body hip rotation target      | ✅     | Deterministic inclined-plane golf hip rotation profile with both-socket simulator application |
| F10 | Unified Tools Sidebar               | 🔄     | Optional dockable tabbed utility sidebar for downstream PyQt/PySide host applications         |

### API / Interface Contract

**GUI:**

- `UnifiedToolsLauncher()` — Main launcher application
- Tools accessed via discoverable plugin interface
- Search, filter, favorites, recent tools in UI

**CLI:**

- `python -m tools <tool_name> [args]` — Command-line invocation
- `python -m tools --list` — List available tools
- `python -m tools --help <tool_name>` — Tool-specific help
- Launcher and maintenance CLI entry points may write explicit stdout/stderr user messages via small helper functions when terminal output is part of the script contract; non-CLI runtime diagnostics should use structured logging.

**Library:**

```python
from tools import get_tool_loader
loader = get_tool_loader()
calculator = loader.load("Engineering_Calculator")
result = calculator.compute(params)
```

Shared Qt hosts can install the optional unified tools sidebar when the
`upstream_drift_tools.ui.tools_sidebar` package is available:

```python
from upstream_drift_tools.ui.tools_sidebar import install_tools_sidebar

status = install_tools_sidebar(main_window, project_root=project_root)
```

**Web API:**

- POST `/api/tools/<tool_name>/compute` — Execute tool
- GET `/api/tools/<tool_name>/schema` — Get input/output schema
- GET `/api/tools/list` — List all available tools
- GET `/api/health` — Health check

**Plugin Interface:**

```python
from tools.plugin_system import BaseTool

class MyTool(BaseTool):
    name = "My_Tool"
    description = "Does something useful"

    def compute(self, inputs: dict) -> dict:
        # Implementation
        pass

    def get_schema(self) -> dict:
        # Return JSON schema for inputs/outputs
        pass
```

## 6. Data & Configuration

### Input Data

| Input               | Format                | Source                          | Schema                                    |
| ------------------- | --------------------- | ------------------------------- | ----------------------------------------- |
| Tool parameters     | JSON/YAML/Python dict | User input, CLI args, API calls | Per-tool schema (JSON-schema)             |
| Configuration files | YAML                  | `config/`                       | Tool registry, theme config, plugin paths |
| Scientific data     | CSV/HDF5/NetCDF       | Files, databases                | Domain-specific formats                   |
| MATLAB models       | .m/.mat files         | `matlab/`                       | MATLAB simulation parameters              |

Pickle-backed DataFrame reads and writes are disabled by default in shared data-processing helpers because pickle loading can execute arbitrary code. CSV, Parquet, JSON, Excel, HDF5, Feather, NumPy, MATLAB, Arrow, and SQLite remain the preferred interchange formats; trusted legacy pickle files require an explicit `allow_pickle=True` override.

### Output Data

| Output              | Format        | Destination               | Description                                                                                       |
| ------------------- | ------------- | ------------------------- | ------------------------------------------------------------------------------------------------- |
| Calculation results | JSON/CSV/HDF5 | User's disk, API response | Tool output matching schema                                                                       |
| Cached results      | SQLite/HDF5   | `.cache/`                 | Memoized expensive calculations                                                                   |
| Logs                | JSON/text     | `logs/`                   | Tool execution logs with timings; root-level debug logs and trigger markers must not be committed |
| Reports             | HTML/PDF      | User's disk               | Generated analysis reports                                                                        |

### Configuration

**Environment Variables:**

All Tools environment variables use the `TOOLS_*` prefix. This is the canonical naming
convention enforced for all new variables. See `docs/SECRETS_MANAGEMENT.md` for guidance
on using these variables safely without hardcoding values.

_System configuration:_

- `TOOLS_PLUGIN_PATH` — Colon-separated paths to plugin directories
- `TOOLS_THEME` — Default theme name (light/dark/custom)
- `TOOLS_CACHE_DIR` — Cache directory for results
- `TOOLS_LOG_LEVEL` — Logging verbosity (DEBUG/INFO/WARNING)
- `TOOLS_RUST_WORKERS` — Number of Rust kernel worker threads

_Optional service credentials (all optional; tools degrade gracefully when absent):_

- `TOOLS_GEMINI_API_KEY` — Gemini API key for AI-powered PDF renaming
  (also accepts legacy `GEMINI_API_KEY` / `GOOGLE_API_KEY` for backward compatibility)
- `TOOLS_GITHUB_TOKEN` — GitHub personal access token for model-generation downloads;
  increases rate limit from 60 to 5000 requests/hour. See QUICKSTART for details.
- `TOOLS_MATLAB_PATH` — Full path to the `matlab` executable. If unset, the launcher
  searches `PATH` via `shutil.which("matlab")` then falls back gracefully.

_Naming convention enforcement:_ New optional-service variables **must** use the
`TOOLS_` prefix. Legacy bare names (e.g. `GEMINI_API_KEY`) are accepted only for
backward compatibility and will not be added for new services.

**Config Files:**

- `config/tools_registry.yml` — Available tools and metadata
- `config/theme_config.yml` — Theme settings and customization
- `config/plugin_config.yml` — Plugin discovery paths and settings
- `config/web_api_config.yml` — FastAPI server configuration

### Repository Hygiene

- Generated logs, trigger files, and empty marker files belong in `logs/`, `output/`, or temporary work directories and must not be tracked at the repository root.
- Root-level artifacts such as `.ci_trigger.py`, `MUJOCO_LOG.TXT`, `error_log.txt`, `wave_log.txt`, and marker files ending in `Last` are treated as disposable debug output.

## 7. Testing Specification

### Testing Strategy

Test pyramid with unit tests at the base, integration tests for tool interactions, acceptance tests for end-to-end workflows. Markers organize tests by category: unit, integration, acceptance, contract, and slow. GUI and Rust components tested separately.
Sidekick optional-dependency tests must simulate importable and missing
optional packages without requiring those packages in the base test
environment. Sidekick Qt dock chrome tests run serially on Windows xdist
workers and set Qt offscreen mode before importing PyQt6 to avoid GUI worker
crashes.

### Test Organization

| Category    | Location             | Framework | Markers                    |
| ----------- | -------------------- | --------- | -------------------------- |
| Unit        | `tests/unit/`        | pytest    | `@pytest.mark.unit`        |
| Integration | `tests/integration/` | pytest    | `@pytest.mark.integration` |
| Acceptance  | `tests/acceptance/`  | pytest    | `@pytest.mark.acceptance`  |
| Contract    | `tests/contract/`    | pytest    | `@pytest.mark.contract`    |
| GUI         | `tests/gui/`         | pytest-qt | `@pytest.mark.gui`         |
| DWSIM       | `tests/dwsim/`       | pytest    | `@pytest.mark.dwsim`       |
| Slow        | `tests/slow/`        | pytest    | `@pytest.mark.slow`        |

`pytest.ini` registers every marker required by `CLAUDE.md`, including benchmark,
scientific, headless-safe, OpenGL, and parity markers. Pytest runs with strict
marker validation and strict xfail handling so stale marker names or unexpected
passes fail early in CI.

### Coverage Requirements

| Scope         | Minimum                              | Current         | Enforced By                             |
| ------------- | ------------------------------------ | --------------- | --------------------------------------- |
| Overall       | 60% target, current-baseline ratchet | 24.48% baseline | CI (`scripts/check_coverage_policy.py`) |
| Core tools    | 75%                                  | ~81%            | CI                                      |
| Plugin system | 80%                                  | ~85%            | CI                                      |

### Required Test Scenarios

- [ ] Tool instantiation returns valid object with correct schema
- [ ] UnifiedToolsLauncher starts and displays available tools
- [ ] Plugin discovery finds all registered tools
- [ ] Calculation produces deterministic results for same inputs
- [x] Pressure-drop interface regression tests cover facade exports, helper-driven validation, and calculator/model interoperability
- [ ] Web API endpoint validates input and returns JSON response
- [ ] Rust kernel outperforms pure Python equivalent by 10x+
- [ ] Theme system applies consistently across all GUI tools
- [ ] Data processing pipeline handles malformed input gracefully

## 8. Quality Standards

### Code Quality Tools

| Tool      | Version | Purpose                           | Blocking? |
| --------- | ------- | --------------------------------- | --------- |
| ruff      | latest  | Linting + formatting              | Yes       |
| mypy      | latest  | Type checking                     | Yes       |
| pytest    | latest  | Testing framework                 | Yes       |
| bandit    | latest  | Security scanning                 | Yes       |
| pip-audit | latest  | Dependency vulnerability scanning | Yes       |

### Design Principles

- **TDD**: Yes — tests written before/with implementation for core tools
- **Design by Contract (DbC)**: Yes — schema validation, precondition/postcondition checks
- **DRY**: Yes — shared_utilities module reduces duplication across tools
- **Orthogonality**: Yes — tools are independent, composable, minimal coupling

### CI/CD Pipeline

| Workflow                | Trigger        | Purpose                                | Blocking? |
| ----------------------- | -------------- | -------------------------------------- | --------- |
| `ci-standard.yml`       | Push/PR        | Unit tests, linting, type checking     | Yes       |
| `test-matrix.yml`       | Push/PR        | Test on Python 3.10/3.11/3.12          | Yes       |
| `integration-tests.yml` | Push/PR        | Integration and contract tests         | Yes       |
| `gui-tests.yml`         | Push/PR        | GUI rendering and interaction tests    | Yes       |
| `rust-build.yml`        | Push/PR        | Rust kernel compilation and benches    | Yes       |
| `dwsim-tests.yml`       | Manual trigger | DWSIM simulation tests (long-running)  | No        |
| `security-scan.yml`     | Daily          | bandit + pip-audit                     | Yes       |
| `performance-bench.yml` | Weekly         | Benchmark Rust kernels vs alternatives | No        |
| `build-release.yml`     | Tag push       | Build wheels, binaries, docs           | Yes       |

## 9. Dependencies

### Runtime Dependencies

| Package    | Version | Purpose                      |
| ---------- | ------- | ---------------------------- |
| numpy      | latest  | Numerical computing          |
| scipy      | latest  | Scientific functions         |
| pandas     | latest  | Data frames and manipulation |
| matplotlib | latest  | Plotting and visualization   |
| sympy      | latest  | Symbolic mathematics         |
| pydantic   | latest  | Data validation              |
| PyYAML     | latest  | YAML parsing                 |
| defusedxml | latest  | Safe XML parsing             |
| PyQt6      | latest  | GUI toolkit                  |

### Development Dependencies

| Package          | Version | Purpose                  |
| ---------------- | ------- | ------------------------ |
| pytest           | latest  | Testing framework        |
| pytest-cov       | latest  | Coverage reporting       |
| pytest-xdist     | latest  | Parallel test execution  |
| pytest-timeout   | latest  | Test timeout enforcement |
| pytest-benchmark | latest  | Performance benchmarking |
| pytest-qt        | latest  | PyQt6 testing utilities  |
| mypy             | latest  | Type checking            |
| ruff             | latest  | Linting and formatting   |
| bandit           | latest  | Security scanning        |
| pip-audit        | latest  | Dependency audit         |

### Optional Dependency Groups

| Group    | Packages                 | Purpose                             |
| -------- | ------------------------ | ----------------------------------- |
| urdf     | urdfpy, trimesh          | Robot URDF parsing and manipulation |
| signal   | scipy.signal extensions  | Signal processing tools             |
| process  | thermodynamics libs      | Process engineering calculations    |
| robotics | PyBullet, ikpy           | Robotics simulation and kinematics  |
| gui      | PyQt6, plotly            | GUI and interactive visualization   |
| theme    | custom theme libs        | Advanced theming and styling        |
| pid      | control, slycot          | PID controller design               |
| cad      | CadQuery, Fusion 360 API | CAD generation and integration      |
| dwsim    | DWSIM COM integration    | Process simulation (Windows only)   |

### Fleet Dependencies

| Repo                  | Relationship            | Description                                |
| --------------------- | ----------------------- | ------------------------------------------ |
| Repository_Management | Depends on              | Consumes templates, workflows, skills      |
| Tools_Private         | Depends by / Depends on | Shares test patterns, assessment framework |
| [Other fleet repos]   | Depends by              | Optional integration with Tools utilities  |

## 10. Deployment & Operations

### How to Run

```bash
# Prerequisites
- Python 3.10+
- Rust toolchain (for kernel compilation)
- pip, poetry, or uv
- Qt6 runtime libraries (for GUI)
- Git

# Installation (local development)
git clone https://github.com/D-sorganization/Tools.git
cd Tools
pip install -e ".[dev,gui]"

# Installation (with optional groups)
pip install -e ".[dev,gui,process,robotics,dwsim]"

# Running the GUI launcher
python UnifiedToolsLauncher.py

# Running via CLI
python -m tools --help
python -m tools Engineering_Calculator --params '{"param1": 10, "param2": 20}'

# Running via library
from tools import get_tool_loader
loader = get_tool_loader()
tool = loader.load("My_Tool")
result = tool.compute({"input": "value"})

# Running web API
python -m tools.web_applications.api --host 0.0.0.0 --port 8000

# Running tests
pytest tests/ -v
pytest tests/ -m "not slow" --maxfail=3
pytest tests/ -m "unit or integration" --cov=src --cov-fail-under=60
pytest tests/ -m "gui" --qt-no-opengl

# Building Rust kernels
cd rust_core/math-primitives && cargo build --release
cd rust_core/tools-core && cargo build --release

# Building distribution
python -m build
python -m twine upload dist/* --skip-existing
```

### Build Artifacts

| Artifact            | Format       | Destination         |
| ------------------- | ------------ | ------------------- |
| Python wheel        | `.whl`       | PyPI / `dist/`      |
| Source distribution | `.tar.gz`    | PyPI / `dist/`      |
| Rust binaries       | `.so`/`.pyd` | Embedded in wheel   |
| Documentation       | HTML         | `docs/_build/html/` |
| Test reports        | HTML/JSON    | `reports/`          |

## 11. Roadmap & Open Issues

### Current Phase

Active development with stable core, continuous tool expansion, and web API in progress.

### Planned Work

| Priority | Item                                 | Issue/PR | Target Date |
| -------- | ------------------------------------ | -------- | ----------- |
| P0       | Complete FastAPI web interfaces (F6) | TBD      | Q2 2026     |
| P1       | Add 10 more scientific tools         | TBD      | Q2 2026     |
| P1       | Optimize Rust kernels for multi-core | TBD      | Q3 2026     |
| P2       | Plugin marketplace / registry        | TBD      | Q3 2026     |
| P2       | Cloud deployment templates           | TBD      | Q4 2026     |

### Known Limitations

- DWSIM integration Windows-only (COM interface limitation)
- Some MATLAB tools require MATLAB runtime installed
- Large datasets may cause GUI slowdowns without optimization
- Plugin system doesn't yet support hot-reloading
- Web API authentication/authorization not yet implemented

## 12. Change Log

| Date       | Version | Changes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-05-23 | 1.1.200 | Added `sidekick.bootstrap` import to the deprecated `upstream_drift_tools` compatibility shim to preserve legacy import paths.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-05-22 | 1.1.199 | Fixed mypy TYPE_CHECKING import guards in sidekick process calculators (syngas_compression_calculator, acid_gas_dewpoint_calculator, pressure_drop_interface, syngas_compression_engine) and calculator_state_mixin to use `if TYPE_CHECKING:` conditional imports for optional PyQt6/matplotlib dependencies, eliminating incompatible-assignment and no-redef errors across Qt-installed and Qt-absent environments.                                                                                                                                              |
| 2026-05-22 | 1.1.198 | Tightened local hook behavior for consolidated task branches so pre-push fleet guardrails inspect the unpushed commit range before falling back to the full repository, and changed the Bandit pre-push hook to scan the Python files selected by pre-commit instead of re-scanning existing repository-wide baseline debt.                                                                                                                                                                                                                                         |
| 2026-05-21 | 1.1.195 | Resolved shared AI/chat unit-test failures by tightening Rust adapter optional-backend behavior, removing obsolete phase-one integration coverage, and updating Ollama, Rust adapter, and AI memory manager tests to use deterministic mocks for terminal-provider and event-loop contracts.                                                                                                                                                                                                                                                                        |
| 2026-05-20 | 1.1.192 | Fixed shared Sidekick chat dock shutdown so an intentional widget close suppresses the WebSocket reconnect timer while unexpected disconnects retain the existing retry path; added focused regression coverage for both lifecycle branches.                                                                                                                                                                                                                                                                                                                        |
| 2026-05-20 | 1.1.191 | Hardened Sidekick test-health coverage so the Jupyter tab availability positive path simulates an importable optional `nbformat` module without requiring the package in the base environment, while the missing-dependency negative path remains covered. Marked the Sidekick dock close-affordance Qt tests as serial/offscreen and skipped them inside Windows xdist workers so the serial lane keeps coverage without crashing parallel workers.                                                                                                                |
| 2026-05-20 | 1.1.190 | Added shared Sidekick/chat launcher integration contracts: `ChatServiceBase.condense_to_memory()` now persists explicit memory candidates through the shared memory manager, `UnifiedToolsSidebar.open_tab()` focuses visible and hidden tabs with `os_terminal` compatibility, ChatDockWidget exposes readiness diagnostics, and Qt chat imports gained subprocess-backed PyQt6 runtime diagnostics with focused regression coverage.                                                                                                                              |
| 2026-05-18 | 1.1.185 | Added `htmlFor` and `id` mapping to range inputs in `SwingComparison.tsx` (`src/media_processing/video_processor/apps/web`) to improve screen reader accessibility.                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-05-18 | 1.1.184 | Optimized Nelder-Mead optimization loop in pendulum simulator by replacing map and slice with pre-allocated arrays and standard for loops to minimize GC pauses.                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-05-17 | 1.1.183 | Pre-allocated the `results` array in the `solveODESystem` hot RK4 integration loop (`src/ode_solver/web/src/lib/odeSolver.ts`) to eliminate continuous memory reallocation overhead and garbage collection pauses during large numerical simulations.                                                                                                                                                                                                                                                                                                               |
| 2026-05-15 | 1.1.181 | Split AI settings local-provider configuration widgets so Ollama keeps its host/model discovery controls, Cline shows its own endpoint test UI, BitNet shows an installation-root hint tied to the main model selector, and CLI-backed providers no longer render misleading Ollama-specific fields; added focused PyQt6 regression coverage for the provider-specific widget contracts.                                                                                                                                                                            |
| 2026-05-15 | 1.1.179 | Added a markdown-backed shared notes card store with stable path-safe IDs, metadata round trips, validated note and board colors, reversible markdown-card recycling/restoration, legacy `project.notes.txt` migration, import-safe backend coverage, and a lightweight Sidekick Notes color-control contract that reuses the shared store.                                                                                                                                                                                                                         |
| 2026-05-15 | 1.1.178 | Added an optional Sidekick Function Generator tab with import-safe PyQt6 launcher integration, shared default-tab/help metadata, design-token aliases, and focused sidebar regression coverage.                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-05-15 | 1.1.176 | Added Sidekick calculator workspace management with isolated calculator-local variables, explicit local-to-global promotion, scoped local/global JSON workspace persistence helpers, focused regression coverage for merge, replace, malformed-file rollback, and duplicate-facade separation behavior, stabilized Sidekick data explorer dtype summaries across pandas string dtype changes, and kept calculator-tab expression evaluation inside the shared safe math evaluator so headless imports do not require Flask or tool-specific calculator packages.    |
| 2026-05-14 | 1.1.175 | Added a lazy optional Sidekick Data Processor tab that stays hidden by default, reports missing UI/runtime dependencies without crashing Sidekick, and exports validated selected Data Processor results into the shared workspace registry with focused import/runtime regression coverage.                                                                                                                                                                                                                                                                        |
| 2026-05-14 | 1.1.174 | Added a Sidekick Data Explorer tab with project-scoped file validation, bounded CSV/TSV/JSON/Parquet/Excel preview service limits, schema/null-count sample summaries, preview-to-workspace export, and a structured Data Processor handoff request contract plus focused backend/UI regression coverage.                                                                                                                                                                                                                                                           |
| 2026-05-14 | 1.1.173 | Added a bounded Sidekick workspace command line to the calculator tab for explicit local/global variable assignment, inspection, deletion, clear, and load/save operations, reusing the shared command-history and workspace persistence contracts while keeping workspace mutations separate from arbitrary terminal execution.                                                                                                                                                                                                                                    |
| 2026-05-14 | 1.1.172 | Added a pure-Python Sidekick help registry for default tabs and shared context-menu actions, wired default-tab help metadata into the shared sidebar, exposed a Help action in the tab context menu, added hover hints to compact terminal/notes controls, documented custom-tab help requirements in the sidebar README, and expanded the shared UI regression suite to enforce the new help contract.                                                                                                                                                             |
| 2026-05-14 | 1.1.171 | Added Sidekick named state profile storage helpers with path-safe save/load contracts, atomic malformed-profile rejection, explicit clear-data warning confirmation, sidebar wrapper methods, README guidance, and focused regression coverage.                                                                                                                                                                                                                                                                                                                     |
| 2026-05-14 | 1.1.170 | Added validated Sidekick calculator startup import preferences with default optional NumPy/SciPy aliases, JSON sidebar-state persistence, transaction-safe import execution, missing-dependency diagnostics in the calculator tab, and focused backend/UI regression coverage.                                                                                                                                                                                                                                                                                      |
| 2026-05-14 | 1.1.169 | Added calculator-local Sidekick workspace save/load wiring with an explicit scoped persistence controller, JSON path validation, atomic save, merge-versus-confirmed-replace load behavior, malformed-file rollback, and UI button coverage that keeps calculator workspace persistence separate from the global sidebar workspace registry.                                                                                                                                                                                                                        |
| 2026-05-14 | 1.1.168 | Added a Sidekick file explorer navigation controller with normalized current path state, back/forward/up history, injectable common-location discovery, project-boundary containment, and predictable disabled-state flags, then wired the project explorer widget to expose a compact navigation bar and common-locations sidebar.                                                                                                                                                                                                                                 |
| 2026-05-14 | 1.1.165 | Optimized the ODE solver RK4 integration loop by moving state and derivative buffers from keyed objects to indexed arrays, extracted the solver and presets into a pure module, and added Vitest coverage for analytical decay, coupled oscillator order, and solver preconditions.                                                                                                                                                                                                                                                                                 |
| 2026-05-14 | 1.1.164 | Improved calculator bounds/value input accessibility by labeling the grouped lower-bound, upper-bound, and evaluation-point controls with a shared group name plus explicit accessible names for each field.                                                                                                                                                                                                                                                                                                                                                        |
| 2026-05-14 | 1.1.163 | Optimized the pressure-drop calculator gas-composition hot paths by replacing repeated object-entry/value reductions with single-pass keyed loops for mixture molecular weight, total composition, and normalized composition construction.                                                                                                                                                                                                                                                                                                                         |
| 2026-05-14 | 1.1.162 | Refactored Sidekick default tab construction into a focused helper module so `UnifiedToolsSidebar` stays below the changed-file LOC budget while preserving the runtime tab behavior introduced in 1.1.161.                                                                                                                                                                                                                                                                                                                                                         |
| 2026-05-14 | 1.1.161 | Replaced remaining Sidekick runtime placeholders with embedded utility widgets: chat status/optional PyQt chat dock loading, a workspace-aware Python terminal with optional numpy/pandas/scipy aliases, a TI-89 symbolic calculator tab that publishes results into workspace state, and project-persistent notes with explicit save and debounced autosave. Added widget contract coverage for the runtime tabs.                                                                                                                                                  |
| 2026-05-14 | 1.1.160 | Added runtime Sidekick theme reapplication APIs so existing PyQt sidebar instances can switch shared themes or explicit design-token sets without being reconstructed.                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-05-14 | 1.1.159 | Added shared-theme-name resolution to the Sidekick host factory/install helpers so PyQt hosts can opt into canonical theme definitions without hand-building design tokens.                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-05-14 | 1.1.156 | Added shared PyQt6 responsive sizing and application zoom utilities for issue #2647. The theme package now exposes text-aware minimum width helpers, readable form-layout configuration, scroll-area wrapping, a persisted application zoom event filter for Ctrl+wheel/Ctrl+plus/Ctrl+minus/Ctrl+0, and scaled UI tokens for downstream QSS/layout regeneration; package discovery now includes the `shared*` namespace so these fleet imports ship with `ud-tools`.                                                                                               |
| 2026-05-14 | 1.1.155 | Added the canonical Sidekick design-token bridge with pure-Python token exports, CSS-variable and QSS mapping helpers, stable Qt object names/selectors, default shared sidebar styling, and focused tests for token contract and backend import safety.                                                                                                                                                                                                                                                                                                            |
| 2026-05-13 | 1.1.154 | Expanded the shared sidebar into the Sidekick toolkit with configurable tab definitions, persisted left/right dock placement, minimized state, tab ordering, hidden tabs, popped-out tab tracking, redock and duplicate-tab APIs, and tests for flexible host workflows while preserving the existing `install_tools_sidebar` contract.                                                                                                                                                                                                                             |
| 2026-05-13 | 1.1.153 | Added the shared `upstream_drift_tools.ui.tools_sidebar` package with a Qt-binding-compatible dockable sidebar, project file explorer, workspace registry/state persistence, public `create_tools_sidebar` and `install_tools_sidebar` APIs, and focused backend/import/widget contract tests for downstream host integration.                                                                                                                                                                                                                                      |
| 2026-05-13 | 1.1.152 | Improved chat layout by moving the shared dock Close button into the persistent status header, replacing clipped history-list text with wrapped row widgets, and adding transparent icon-only archive, restore, and delete actions directly on chat-history rows.                                                                                                                                                                                                                                                                                                   |
| 2026-05-13 | 1.1.151 | Hardened shared chat dock terminal lifecycle controls so Start is disabled while a terminal session is pending or active, Stop is enabled only for active sessions, and shell/provider selectors are locked while the selected terminal agent session is running.                                                                                                                                                                                                                                                                                                   |
| 2026-05-13 | 1.1.150 | Improved the shared chat dock terminal interface by populating shell/provider selectors from the terminal provider registry, adding an explicit terminal Stop action wired to the existing WebSocket stop protocol, and adding an in-dock Close button so embedded chat windows can be dismissed from inside the chat UI.                                                                                                                                                                                                                                           |
| 2026-05-13 | 1.1.149 | Added shared AI chat memory management with a Tools-scoped `user_memory.json` store, explicit archived-conversation preference extraction, project-root `AGENTS.md` prompt inclusion, bounded prompt-memory formatting across provider adapters, and focused regression coverage so archived chats inform future sessions without becoming opaque model training data.                                                                                                                                                                                              |
| 2026-05-13 | 1.1.148 | Added data-driven shared chat terminal-provider descriptors for Claude Code, Codex, Cline CLI, and Gemini CLI, plus default registry builders, install/auth probe command metadata, and command redaction helpers so downstream UIs can enumerate terminal agents without copying provider lists or logging secret-like command values.                                                                                                                                                                                                                             |
| 2026-05-13 | 1.1.144 | Added a native BitNet direct subprocess adapter for shared AI chat provider resolution, exposing local 1.58b models through the adapter factory and settings metadata without requiring an external FastAPI server.                                                                                                                                                                                                                                                                                                                                                 |
| 2026-05-13 | 1.1.143 | Synchronized Signal Toolkit Matplotlib canvas theming for issue #2582 by applying the active fleet plot theme after axes are created, keeping legacy `setup_dark_theme()` wired to the shared theme manager, and adding regression coverage for themed axes and spines.                                                                                                                                                                                                                                                                                             |
| 2026-05-13 | 1.1.142 | Registered the migrated Video Analyzer PyQt6 surface in the generator-backed tools catalog and surface contract so issue #2585 is visible through both the canonical GUI manifest and generated launcher outputs.                                                                                                                                                                                                                                                                                                                                                   |
| 2026-05-13 | 1.1.141 | Made the migrated Video Analyzer installable and launchable from Tools for issue #2585 by adding package discovery, a `video-analyzer` console script, optional video runtime dependencies, installed-package import paths, and focused packaging/launcher regression tests.                                                                                                                                                                                                                                                                                        |
| 2026-05-13 | 1.1.140 | Tightened the shared chat package contract for issue #2592 by exporting the documented model/list/index facade symbols, adding a `chat` optional dependency group and compatibility matrix, fixing installed-package lazy Qt loading, validating model/index status payloads, and removing product-specific defaults from reusable AI assistant GUI metadata.                                                                                                                                                                                                       |
| 2026-05-12 | 1.1.135 | Added Rust `tools-core.signal` moving-average and exponential-smoothing kernels with PyO3 numpy vector-in/vector-out endpoints, filling the remaining smoothing-filter slice after the LMS/RLS migration.                                                                                                                                                                                                                                                                                                                                                           |
| 2026-05-12 | 1.1.134 | Promoted LMS/RLS adaptive filters to native Rust implementations via PyO3 bindings, eliminating Python-side vectorization overhead for high-frequency signal processing pipelines (PR #2575).                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-05-11 | 1.1.132 | Fixed `signal_toolkit.calculus` import: replaced bare `from src.shared.python.contracts import require` (broken because the repo root is not on `pytest`'s pythonpath) with the sibling-module try/except pattern used in `core.py`, and cast `Differentiator.differentiate`'s return to `np.asarray(dy)` to keep mypy `no-any-return` clean. Unblocks `tests (3.x)` matrix on `main`.                                                                                                                                                                              |
| 2026-05-11 | 1.1.131 | Added shared `codemap` package (`src/shared/python/codemap/`) — tree-sitter symbol index over SQLite FTS5 with a 6-function pydantic query API (`search_code`, `get_symbol`, `who_calls`, `imports_of`, `neighbors`, `repo_summary`), CLI (`codemap rebuild/search/who-calls/export/info`), `watchdog` daemon (`codemap-watch`), and FastMCP server (`codemap-mcp`) so external coding agents inherit the same data the in-app chat consumes. `.codemap/` is gitignored; embedding layer deferred to a follow-up.                                                   |
| 2026-05-11 | 1.1.130 | Hardened `signal_toolkit.calculus.Differentiator.differentiate` with an explicit `require(order >= 1, ...)` precondition so non-positive derivative orders raise `PreconditionError` instead of silently producing an empty derivative loop.                                                                                                                                                                                                                                                                                                                        |
| 2026-05-11 | 1.1.129 | Added dynamic focus shifting to inline form validation within the Calculator app. This prevents keyboard focus traps by focusing the first invalid input (`.focus()`) and marking it with `aria-invalid="true"`.                                                                                                                                                                                                                                                                                                                                                    |
| 2026-05-07 | 1.1.128 | Pre-compiled ODE Solver derivative expressions outside the RK4 loop while preserving the existing non-finite fallback behavior, so singular or overflowing user formulas still collapse to `0` instead of poisoning the integration state with `NaN` or `Infinity`.                                                                                                                                                                                                                                                                                                 |
| 2026-05-05 | 1.1.125 | Optimized polynomial evaluation using Horners method in `pendulum-web` physics engines (`physics.ts`, `physics_triple.ts`, `physics_golfer.ts`).                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-05-04 | 1.1.124 | Documented production-readiness hardening for generated data-processing batch scripts, shared pandas formula allowlist validation, model-generation mesh upload size and filename checks with cleanup, and MakeHuman generated-script serialization plus the `mesh_generator_makehuman.py` compatibility shim.                                                                                                                                                                                                                                                      |
| 2026-04-26 | 1.1.111 | Improved accessibility for the calculator clear button's soft confirm state. Added `aria-live="polite"` to the parent row and dynamically toggled the `aria-label` between "Clear all fields" and "Confirm clear all fields" to keep screen reader users informed of the required secondary action.                                                                                                                                                                                                                                                                 |
| 2026-04-25 | 1.1.107 | Fixed StrEnum import compatibility for Python 3.10 by routing `steam_engine_calculator` and `video_processor` API modules through the existing `utils.compatibility` backport facade, eliminating import-time failures on the 3.10 CI interpreter.                                                                                                                                                                                                                                                                                                                  |
| 2026-04-25 | 1.1.106 | Added dynamic focus shifting to inline form validation within the Unit Converter app's Custom Units modal. This prevents keyboard focus traps by focusing the first invalid input (`.focus()`) and marking it with `aria-invalid="true"`.                                                                                                                                                                                                                                                                                                                           |
| 2026-04-23 | 1.1.103 | Tightened the shared `model_generation` unified-loader conversion contract so malformed MJCF/URDF XML parse failures are wrapped as `ConversionError`, converter-raised `ConversionError` instances propagate unchanged, and regression tests lock the typed error/logging behavior.                                                                                                                                                                                                                                                                                |
| 2026-04-23 | 1.1.101 | Hardened model-generation REST routing so unexpected route-handler programming errors propagate to the framework adapter instead of being flattened into JSON 500 responses by the route facade, with regression coverage for the propagation contract.                                                                                                                                                                                                                                                                                                             |
| 2026-04-23 | 1.1.100 | Extended the Python 3.10 UTC compatibility contract across document-processing, folder-packing, shared model-generation, upstream-drift UI/state, folder-tool analysis, and launcher timestamp paths by using `timezone.utc` instead of the Python 3.11-only `datetime.UTC` alias while preserving timezone-aware datetime behavior.                                                                                                                                                                                                                                |
| 2026-04-23 | 1.1.99  | Kept shared data-processing result timestamps timezone-aware while preserving Python 3.10 compatibility by using `timezone.utc` rather than the Python 3.11-only `datetime.UTC` alias, keeping the data-processing import contract green across the supported CI interpreter matrix.                                                                                                                                                                                                                                                                                |
| 2026-04-25 | 1.1.105 | Narrowed `ConsoleEnvironment.refresh_user_functions()` to re-raise `KeyboardInterrupt` and `SystemExit` while still logging expected user-code failures from the persisted scripting library, and added focused regression coverage for both reload paths.                                                                                                                                                                                                                                                                                                          |
| 2026-04-23 | 1.1.98  | Documented the rotation converter API exception-boundary tests that keep invalid quaternion parsing mapped to HTTP 422 while allowing unexpected reference-frame runtime failures to propagate for diagnostics instead of being silently swallowed.                                                                                                                                                                                                                                                                                                                 |
| 2026-04-23 | 1.1.97  | Security and robustness remediation pass from adversarial review: tightened exception boundaries and error propagation for shared rotation conversion, scripting runtime, and model-generation loaders; hardened data-processing and state-management paths against invalid inputs and silent failures; and aligned related test coverage for the updated failure-handling contracts.                                                                                                                                                                               |
| 2026-04-23 | 1.1.96  | Hardened ODE and signal generation preconditions so direct RK4 calls reject fewer than two output points, chirp generation rejects single-point time arrays, and sawtooth/triangle/square generation reject non-positive frequencies with clear `ValueError` messages instead of division-by-zero failures.                                                                                                                                                                                                                                                         |
| 2026-04-22 | 1.1.92  | Fixed Design by Contract runtime toggling so contract primitives, decorators, invariant checks, and validation helpers read the canonical contract state instead of stale module-level compatibility aliases; added regression coverage for alias/state divergence.                                                                                                                                                                                                                                                                                                 |
| 2026-04-22 | 1.1.91  | Security hardening (closes #2219): removed starred argument unpacking from the safe mathematical expression evaluator AST allowlist and added regression coverage so expressions such as `sum(*x)` are rejected before execution.                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-22 | 1.1.88  | Test-enforcement fix (closes #2211): restricted GH1732 logging-consistency excluded-directory matching to the top-level `src/<segment>` only, and added regression coverage proving nested path segments named like excluded dirs remain in sweep scope.                                                                                                                                                                                                                                                                                                            |
| 2026-04-22 | 1.1.87  | Documented the `signal_toolkit` package organization for adaptive filters: `AdaptiveFilter` now lives in `adaptive_filter.py` while remaining available from the package root and legacy `filters` module.                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-22 | 1.1.85  | Implementation (closes #2200): added a flat Asteroid Jumper controller snapshot DTO and routed the renderer through it to remove nested state traversal from the draw path.                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-22 | 1.1.84  | Documentation (closes #2200): reviewed deep object traversal hotspots in launchers, Matplotlib/Qt UI code, assessment scripts, Rust ball-flight physics, and Asteroid Jumper controller code, documenting framework/path/import/value-object boundaries that do not require DTO or facade extraction.                                                                                                                                                                                                                                                               |
| 2026-04-22 | 1.1.83  | Optimized statistical calculation in data processor using Welford's algorithm to compute variance in a single pass.                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-19 | 1.1.82  | Removed QTimer.singleShot startup races and leaky lambda captures in shared chat dock and syngas compression calculator UI code by routing deferred initialization through named callbacks and stored helper methods (PR #2163).                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-19 | 1.1.81  | Aligned dependency metadata with the supported Python and toolchain baseline: Python package metadata now starts at Python 3.11, lint/type configuration shares that floor, Black was removed from the canonical format path, and the reproducible requirements lock includes the pytest timeout and benchmark plugins declared by the development manifests (PR #2161).                                                                                                                                                                                            |
| 2026-04-19 | 1.1.80  | Hardened model-generation archive extraction and URDF mesh resolution by normalizing archive member paths, rejecting traversal or absolute members before extraction, and preserving unsafe mesh references as text instead of resolving them to local files (PR #2157).                                                                                                                                                                                                                                                                                            |
| 2026-04-19 | 1.1.79  | Consolidated stale Tools PR fixes covering shared rotation primitives, data processor background worker error surfacing and UI offload, PDF renamer API-key/CORS hardening, narrower exception fallbacks, shared GUI boundary checks, and lower-body manifest registration; also tightened NumPy return typing for the rotation modern robotics helpers checked by quality-gate (PR #2149).                                                                                                                                                                         |
| 2026-04-19 | 1.1.78  | Optimized `TimeRangePanel.tsx` in `data-processor-web` by computing time-column ranges in a single pass and avoiding `Math.min`/`Math.max` spread calls that can overflow the call stack on large datasets (PR #2156).                                                                                                                                                                                                                                                                                                                                              |
| 2026-04-19 | 1.1.77  | Hardened model-generation library GitHub discovery and downloads by validating generated GitHub API URLs, rejecting non-HTTPS model source URLs, and skipping untrusted subdirectory URLs before network retrieval (PR #2146).                                                                                                                                                                                                                                                                                                                                      |
| 2026-04-21 | 1.1.76  | Added screen-reader-only context to dynamic video progress text and pose detection counters so numeric readouts expose their meaning to assistive technology; decorative pulsing dots are now hidden from screen readers (PR #2138).                                                                                                                                                                                                                                                                                                                                |
| 2026-04-21 | 1.1.75  | Optimized `calculateStatistics` in `useDataProcessor.ts` by extracting numbers into a dynamically resizing `Float64Array` during the first pass to eliminate a second pass over the original array of objects (PR #2137).                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-21 | 1.1.74  | Disabled pickle-backed reads, writes, and file-dialog discovery in shared data-processing helpers and upstream drift tooling to prevent arbitrary code execution through unsafe deserialization (PR #2139).                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-21 | 1.1.73  | Improved exception handling and signal re-raising in rotation converter UI threads, scripting environment, and model library imports by capturing background thread exceptions, adding structured logging, and re-raising with context (PR #2088).                                                                                                                                                                                                                                                                                                                  |
| 2026-04-21 | 1.1.72  | Enhanced data processor exception handling by wrapping background threading tasks with try-except blocks that log exceptions and propagate errors to the main thread instead of silently failing (PR #2084).                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-21 | 1.1.71  | Hardened data-processing file I/O by disabling pickle reads and writes by default, removing pickle extensions from GUI-supported file discovery paths, and requiring an explicit trusted-legacy override for pickle use.                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-21 | 1.1.70  | Test configuration hygiene: registered the complete CLAUDE.md marker set in `pytest.ini`, enabled strict xfail handling, and added a contract-test backbone for the ODE solver, pressure-drop calculator, and rotation-converter calc backend request/response models.                                                                                                                                                                                                                                                                                              |
| 2026-04-21 | 1.1.69  | Stopped the bot CI trigger workflow from using stale external credentials for repository checkout and PR/check API operations so bot-authored PRs use repo-scoped workflow credentials for required check discovery.                                                                                                                                                                                                                                                                                                                                                |
| 2026-04-21 | 1.1.68  | Restricted Data Processor web row-copy paths to own enumerable properties via a shared `Object.keys` helper and added regression coverage to prevent inherited prototype keys from being copied into processed rows.                                                                                                                                                                                                                                                                                                                                                |
| 2026-04-21 | 1.1.67  | Filter deleted test files out of the CI changed-test list so PRs that intentionally remove stale tests do not pass non-existent paths to pytest.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-21 | 1.1.66  | Hardened asteroid-jumper physics validation so non-finite timesteps and physics parameters are rejected with explicit `ValueError`s instead of propagating NaN or infinity through simulation state.                                                                                                                                                                                                                                                                                                                                                                |
| 2026-04-21 | 1.1.66  | Simplified root pytest addopts in `pyproject.toml` by removing benchmark and xdist-specific defaults so repository-level test runs do not require those plugins outside focused plugin test contexts.                                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-17 | 1.1.64  | Optimized `applyFilter` loop in `useDataProcessor.ts` by replacing the object spread operator with manual property copying to eliminate significant garbage collection overhead during large dataset processing.                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-17 | 1.1.63  | Hardened model-generation GitHub repository downloads by requiring HTTPS retrievals and validating mesh output paths so API-provided mesh names cannot escape the destination directory; kept the unit-converter development WSGI debugger disabled unless `FLASK_DEBUG=1` is explicitly set.                                                                                                                                                                                                                                                                       |
| 2026-04-17 | 1.1.62  | Enhanced video editor UX by replacing native alert dialogs with inline accessible errors and ensuring proper focus styles.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-17 | 1.1.61  | Replaced runtime `assert` validation in asteroid-jumper physics, rotation-converter UI helpers, and scripting console execution with explicit exceptions so invalid caller input remains guarded under optimized Python.                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-16 | 1.1.60  | Hardened launcher process handling by validating tool names, cleaning up spawned process groups, surfacing explicit model-conversion errors, and regression-testing temporary-file cleanup paths.                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-16 | 1.1.59  | Removed stale root-level debug artifacts (`.ci_trigger.py`, `MUJOCO_LOG.TXT`, `error_log.txt`, `wave_log.txt`, and the empty marker file ending in `Last`), added root-scoped ignore rules for those paths, and locked the hygiene policy with regression tests.                                                                                                                                                                                                                                                                                                    |
| 2026-04-16 | 1.1.58  | Hardened GitHub archive extraction in the model-generation repository helper by validating zip members before unpacking so repository downloads cannot escape the destination directory.                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-16 | 1.1.55  | Replaced object spread operator with manual property copy in `integrateSignals` and `differentiateSignals` loops in `useDataProcessor.ts`; wrapped UI components (`AdvancedPanel`, `ExportPanel`, `FilterPanel`, `ResamplePanel`) in `React.memo()` to prevent unnecessary re-renders.                                                                                                                                                                                                                                                                              |
| 2026-04-15 | 1.1.56  | Refreshed the data processor regression-preparation optimization spec after CI retriggers so the PR-level SPEC freshness gate sees a documentation update on the latest source-changing branch head.                                                                                                                                                                                                                                                                                                                                                                |
| 2026-04-16 | 1.1.57  | Improved the accessibility and semantics of the `AudioRecorder` component in the Video Processor app. Added `aria-label`s to recording control buttons, formatted recording duration for screen readers, hid purely visual elements from screen readers, and enhanced keyboard navigation by adding `focus-visible` styling to all buttons.                                                                                                                                                                                                                         |
| 2026-04-15 | 1.1.55  | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead.                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-16 | 1.1.53  | Added `aria-label` and `title` to the dynamically generated "Remove" button (`×`) in the unit converter Custom Units list for screen reader accessibility.                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-13 | 1.1.52  | Added visually hidden `sr-only` span before the raw timer text in `AudioRecorder.tsx` to provide screen reader context and added `aria-hidden` to purely decorative pulsing red dot.                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-04-13 | 1.1.51  | Added `tools.shared.python.model_generation.editor` compatibility namespace so downstream repos can import the text editor via `tools.shared.python` without duplicating the module; added `-p no:xvfb` to pytest addopts so the test suite runs on headless self-hosted runners that lack Xvfb; applied ruff formatting fixes across GUI stylesheets and multiline string literals.                                                                                                                                                                                |
| 2026-04-12 | 1.1.51  | Replace remaining `print()` calls with `logging` across `src/` modules and disable xvfb pytest plugin to fix CI timeout on headless runners.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-13 | 1.1.48  | Wrapped the `SignalList` and `StatisticsPanel` components in `React.memo()` to prevent expensive re-render cascades in the data processor web application during UI tab navigation.                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-12 | 1.1.47  | Added the shared `tools.mypy_autofix_agent` module and `mypy-autofix` console entry point so downstream fleet repositories can call one maintained mypy autofix implementation instead of carrying duplicated script copies; kept `tools.setup_logging` lazy so CLI startup does not import optional heavy dependencies.                                                                                                                                                                                                                                            |
| 2026-04-11 | 1.1.46  | Lower-body builder DRY refactor: extracted `_build_leg_xml(side, ...)` and `_build_leg_actuators_xml(side)` helpers so both legs and both actuator blocks share a single source of truth. `build_lower_body_xml` now calls each helper once per side instead of duplicating ~45 lines of MJCF. New regression tests assert left/right symmetry of joint/body/actuator/geom/site sets and pin the expected counts.                                                                                                                                                   |
| 2026-04-11 | 1.1.45  | Closed-chain ankle IK in `LowerBodySimulator.setup_initial_pose`: the ankle angles are solved by a closed-form 2-DOF decomposition of the calf's world rotation so each foot's world Z-axis is `(0, 0, 1)` for any feasible hip/knee pose. Raises `ValueError` identifying the offending axis when the required ankle angle exceeds the ±30° joint limit instead of silently clipping. Defaults changed from 30°/120°/20° (infeasible, silently clipped) to 20°/30°/20° (a feasible golf address posture). The PyQt panel catches infeasibility and logs a warning. |
| 2026-04-11 | 1.1.44  | Lower-body simulator DRY/LOD refactor: centralized mj_name2id lookups into a single cache populated in `_cache_indices` (joints, actuators, sites, geoms, bodies), eliminated reflective lookups from hot paths (`step`, `compute_diagnostics`, `inverse_kinematics`, `set_joint_polynomial`, `analyze_induced_acceleration`), and decomposed `compute_diagnostics` into `_collect_tracking_error`, `_collect_joint_torques`, `_collect_ground_reaction_forces`. Added contract test suite locking down the public API surface (`-m contract`).                     |
| 2026-04-11 | 1.1.43  | Added inclined-plane pelvis rotation driver to the lower-body simulator: `set_pelvis_inclined_rotation(target, ...)` wrenches the pelvis free joint via `data.xfrc_applied` each step so the body tracks an inclined rotation axis (spine angle) plus a smoothstep-ramped lateral weight shift during the downswing. New `InclinedPlaneHipRotationTarget.lateral_shift_m`, `lateral_shift_at(t)`, and `target_quaternion_at(t)` with full DbC.                                                                                                                      |
| 2026-04-11 | 1.1.42  | Anatomically-shaped lower-body pelvis: composite of inertial host ellipsoid plus five mass=0 visual-only landmark geoms (sacrum, bilateral iliac wings, bright-red ASIS spheres, pubic symphysis) so pelvic tilt is visually unambiguous in the viewer without any change to dynamics.                                                                                                                                                                                                                                                                              |
| 2026-04-11 | 1.1.41  | Added a full reset control to the lower-body PyQt panel that stops playback, clears history, returns MuJoCo time to zero, preserves loaded golf hip rotation targets, and reapplies the target pose at `t=0`.                                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-11 | 1.1.40  | Added `tools.shared.python.model_generation.editor` compatibility exports (including `TextEditor` alias) to support removing duplicate model editor implementations in downstream repos that consume Tools as a dependency.                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-11 | 1.1.39  | Extended lower-body simulator history playback diagnostics so cached frames expose the configured inclined-plane hip rotation target for scrub-based analysis and verification.                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-11 | 1.1.38  | Added the lower-body inclined-plane hip rotation target profile with deterministic sampling, DbC validation, both-socket simulator application, and diagnostics/history coverage for the first golf lower-body rotation slice.                                                                                                                                                                                                                                                                                                                                      |
| 2026-03-28 | 1.0.0   | Initial specification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-03-29 | 1.0.1   | Document performance improvement in DataChart downsampling algorithm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-03-30 | 1.0.2   | A-N assessment remediation: LoD refactoring in convert_tools_icon.py, launch.py, launch_signal_toolkit.py, verify_launcher.py; DbC input validation added to launch_tool, bootstrap, migrate_file, \_print_environment_info, \_check_launcher_file, \_print_recommendations, \_on_poly_generated; docstrings added to **init** and missing functions in setup_dev.py, remove_broken_scripts.py, migrate_print_to_logging.py, launch_signal_toolkit.py.                                                                                                              |
| 2026-03-31 | 1.0.3   | Fix CI import error in tests/shared/python/test_contracts.py and optimize React rendering in ToolsPanel.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-01 | 1.0.4   | Add keyboard accessibility (focus-within) to video player controls in web application.                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-04-01 | 1.0.5   | Optimize the data processor median filter to reuse a `Float64Array` buffer and preallocate result storage, reducing per-window allocations during large CSV filtering workflows.                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-02 | 1.0.6   | Refactored AnalyticsSuite (computeCorrelation, computeRegression, pearsonCorrelation) to use iterative primitive arrays and eliminate chained .map/.filter mapping overhead, vastly reducing garbage collection pressure.                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-02 | 1.0.7   | Run comprehensive assessments and apply auto-fixes across the repository.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-03 | 1.0.8   | Refactor `linearRegression` and `polynomialRegression` in `useDataProcessor.ts` to replace multiple consecutive `.reduce()` and `.map()` array iteration methods with single-pass `for` loops, improving performance for large datasets.                                                                                                                                                                                                                                                                                                                            |
| 2026-04-10 | 1.0.9   | Optimize Math Functions using single-pass loops.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-10 | 1.1.0   | Add keyboard accessibility and focus management to the Data Processor web application file upload dropzone.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-05-18 | 1.1.1   | Fix command injection vulnerability in MATLAB Quality Utils by escaping single quotes in paths passed to MATLAB and Octave shells.                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-05-18 | 1.1.2   | Optimize PCA mathematical matrix calculations in AnalyticsSuite to use column-wise typed Float64Array to prevent large O(N) allocation overhead.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-05-18 | 1.1.3   | Optimize linear regression calculation in AnalyticsSuite using single-pass loops instead of map/reduce to minimize garbage collection pauses.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-05-19 | 1.1.4   | Add inline error message handling to SignalList to avoid blocking native alert dialogs and added comprehensive focus-visible states across all signal list interface buttons for enhanced keyboard accessibility.                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-04 | 1.1.5   | Replace print statements with logger calls in lower_body_model main entry point to comply with no-print policy and improve production logging.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-04-05 | 1.1.6   | Optimize DataChart point extraction loop to explicitly map selected properties instead of using an object spread on the entire row in `src/data_processing/data_processor/web/src/components/DataChart.tsx`.                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-05 | 1.1.7   | Improve HelpPanel accessibility by adding ARIA expanded states and control links to accordion toggles, and adding explicit focus-visible rings for keyboard users.                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-05 | 1.1.8   | Optimize PlotView WebGL rendering to use Float64Array and bypass map array creation overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-05 | 1.1.9   | Bridge the embedded `src/pendulum_simulator/tests` suite into the top-level `tests/` tree so standard `pytest tests/` collection includes pendulum coverage without double-collecting the same files during root-level pytest runs.                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-05 | 1.1.10  | Standardize vessel drafter `require_positive` usage onto the fleet-wide `(value, name)` argument order while keeping guarded support for the legacy local order and adding regression tests for the signature normalization.                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-05 | 1.1.11  | Deduplicate repeated scalar surface evaluator closures in `analysis_tab.py` by routing matrix and transformed-value cases through shared helper builders, with regression coverage for the new helper paths.                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-05 | 1.1.12  | Expand the embedded-suite discovery policy so root-level pytest ignores bridged `src/` suites by default while `pytest tests/` includes both pendulum and solar-system embedded tests through top-level bridge directories.                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-05 | 1.1.13  | Move pendulum optimizer objective-refresh wiring behind a public `OptimizationWidget` API so `SimulationPanel` no longer reaches through private optimizer button and log internals before optimization runs.                                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-06 | 1.1.14  | Remove developer-machine repository paths from maintenance scripts and eliminate the local sys.path bootstrap fallback from convert_tools_icon.py.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-06 | 1.1.15  | Replace chained array map and filter operations with a single loop in the calculateTrendline algorithm to prevent memory allocation and garbage collection overhead.                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-04-06 | 1.1.16  | Add focus-within styles to video uploader dropzone and missing aria-labels to the volume and seek range inputs in the video processor web application to improve keyboard navigation visibility.                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-06 | 1.1.17  | Optimize Polynomial Regression Matrix Construction in AnalyticsSuite using single-pass loops.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-06 | 1.1.18  | Refactored `applyFilter` inside `useDataProcessor.ts` to pre-allocate buffers and run the mapping in a single loop.                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-06 | 1.1.19  | Split `pressure_drop_interface.py` into facade-oriented `pressure_drop_api`, `pressure_drop_validation`, `pressure_drop_reference`, and `pressure_drop_results` modules while preserving the public interface and extending regression coverage for the pressure-drop calculator.                                                                                                                                                                                                                                                                                   |
| 2026-04-07 | 1.1.20  | Added explicit `focus-visible` keyboard focus indicators to the Video Processor web `ToolsPanel` buttons, color controls, slider, and destructive action buttons so keyboard navigation remains visible throughout the drawing workflow.                                                                                                                                                                                                                                                                                                                            |
| 2026-04-07 | 1.1.21  | Split `model_generation` REST routing from the Flask and FastAPI adapters behind a backward-compatible shim, decomposed the pressure-drop engine into friction-factor, flow-property, fittings, and compressible-flow modules with regression coverage for the preserved calculations, and restored the top-level `contracts` compatibility export for `_resolve_contract_level`.                                                                                                                                                                                   |
| 2026-04-07 | 1.1.22  | Formalize stdout/stderr helper usage for CLI-facing launcher and coverage-gate scripts so terminal output remains explicit while avoiding ad hoc `print()` usage in those entry points.                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-07 | 1.1.23  | Split the data-processor neural-network script exporter, ANOVA analyzer, and vectorized filter engine into smaller domain modules behind backward-compatible facades, and add focused regression tests for the preserved public and compatibility interfaces.                                                                                                                                                                                                                                                                                                       |
| 2026-04-07 | 1.1.25  | Replaced raw `print()` summary emission in `scripts/generate_tools_json.py` with an explicit stdout helper, added regression coverage for the CLI entrypoint's generated-file summary contract, and aligned the humanoid mesh-generator facade with the split backend modules so refreshed type-checking stays green after the backend extraction on `main`.                                                                                                                                                                                                        |
| 2026-04-07 | 1.1.26  | Extracted the double-pendulum golf equations popup string literals into `equations_data.py`, leaving the popup module focused on presentation and control wiring while preserving the existing dialog behavior.                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-07 | 1.1.27  | Optimized `AnalyticsSuite` regression filtering by staging selected x/y series values into `Float64Array` buffers before converting them back to plain arrays for the existing result contract, reducing repeated push-allocation overhead in large regression workloads.                                                                                                                                                                                                                                                                                           |
| 2026-04-07 | 1.1.28  | Optimized `AnalyticsSuite` Pearson correlation by preserving the PR's single-pass accumulation and variance-clamping path while widening the helper to accept pre-allocated `Float64Array` inputs from the newer analytics data flow.                                                                                                                                                                                                                                                                                                                               |
| 2026-04-07 | 1.1.29  | Decomposed the PSA GUI into focused `ui/` modules while tightening the compatibility export surface to immutable `__all__` tuples in both the facade module and the extracted UI package.                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-07 | 1.1.30  | Extracted the public enums/dataclass contracts and low-level helper kernels for `time_series_decomposition` into focused support modules, leaving the main module centered on decomposition orchestration while preserving the existing public import surface through the compatibility facade.                                                                                                                                                                                                                                                                     |
| 2026-04-08 | 1.1.31  | Memoize AnalyticsSuite chart data using useMemo and optimize the scatter regression component with a single-pass loop, drastically reducing React rendering and GC overhead.                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-08 | 1.1.32  | Optimized data array filtering in `useDataProcessor.ts` by replacing `Array.push()` calls with `Float64Array` buffers in `calculateTrendline`, and replacing chained `filter()` passes in `trimTimeRange` with a single-pass `for` loop that avoids creating and resizing intermediate arrays.                                                                                                                                                                                                                                                                      |
| 2026-04-09 | 1.1.33  | Added a loading spinner and `aria-pressed` states to the `VideoEditor.tsx` component in the video processor web application to improve user experience and accessibility during video export operations.                                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-09 | 1.1.35  | Added a shared provider-pack manifest for the pendulum simulator under `src/pendulum_simulator`, plus a repo-local validator and regression tests that keep the manifest aligned with the real package entry point, working directory, Python path, icon asset, and launcher metadata required for future UpstreamDrift shared-launch integration.                                                                                                                                                                                                                  |
| 2026-04-09 | 1.1.34  | Wrapped DataTableView, PlotView, and AnalyticsSuite in `React.memo`, and memoized activeSignals with `useMemo` to prevent expensive visualization re-renders on unrelated UI state changes.                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-10 | 1.1.37  | Add explicit focus-visible styles to the interactive buttons (Upload New Video, Play/Pause, Mute/Unmute) within the `VideoPlayer` component for improved keyboard navigation visibility.                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-12 | 1.1.48  | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead.                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-15 | 1.1.49  | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead.                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-17 | 1.1.50  | Hardened model import security by enforcing HTTPS GitHub host allowlisting for remote model-library fetches, validating user-provided GitHub repository URLs before import, dropping directory components from remote mesh names, and rejecting separator-containing URDF viewer filenames before filesystem resolution.                                                                                                                                                                                                                                            |
| 2026-04-21 | 1.1.67  | Optimized row copying logic in useDataProcessor.ts by replacing `Object.keys()` with a `for...in` loop and `hasOwnProperty`, substantially reducing GC allocation overhead inside tight data processing loops.                                                                                                                                                                                                                                                                                                                                                      |
| 2026-04-21 | 1.1.66  | Refreshed regression test coverage for architecture boundaries, data-processor compatibility, folder archive operations, and upstream-drift contract smoke behavior while keeping the production implementation unchanged.                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-22 | 1.1.90  | Repaired CI dependency bootstrap workflows so shared runners with broken `wheel` metadata upgrade `pip` and `setuptools` separately, then reinstall `wheel` with `--no-deps` before workflow linting and Python test jobs.                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-22 | 1.1.91  | Hardened data-processor normalize and standardize transforms so constant columns raise `TransformationError` instead of silently producing all-NaN output, with regression coverage preserving original data after the failed transform.                                                                                                                                                                                                                                                                                                                            |
| 2026-04-22 | 1.1.89  | Hardened `utils.env_utils` repo-root fallback discovery so shallow path layouts no longer raise import-time index errors, and added regression coverage for shallow fallback computation behavior.                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-22 | 1.1.93  | Enforced finite, non-negative altitude preconditions for the Rust standard-atmosphere model and added operator whitelisting before `DataProcessorEngine.filter_data()` constructs pandas query expressions.                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-22 | 1.1.94  | Updated the shared `DataProcessor.apply_filter()` Butterworth path to use an explicit `sample_rate` or infer it from time-column spacing instead of hard-coding 1000 Hz, with regression coverage for non-1 kHz datasets.                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-22 | 1.1.95  | Canonicalized the Rust universal gas constant by updating `math::R_GAS` to the full CODATA value and having `engineering::R_UNIVERSAL` reuse the same constant.                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-23 | 1.1.102 | Updated Unit Converter `removeCustomUnit` workflow to use an inline soft confirm pattern, eliminating thread-blocking `confirm()` dialogs and improving accessibility with `aria-live`.                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-28 | 1.1.112 | Updated Unit Converter UI to dynamically retarget labels for custom combobox search inputs, ensuring explicit accessible names and resolving click-to-focus gaps.                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-05-02 | 1.1.121 | Preserved `smoothAngles` behavior for fractional moving-average window sizes by dividing optimized mid-window sums by the actual sample span, added a Vitest regression in the golf video-processor web app, hardened the benchmark plugin bootstrap in CI/benchmark workflows against shared-runner cache drift, and restored the CI Standard coverage-policy skip for PRs that touch no Python source or Python tests.                                                                                                                                            |
| 2026-05-01 | 1.1.120 | Hardened the calculator web expression validation gate by rejecting Python object hierarchy, lifecycle, async, import, and control-flow injection markers before SymPy parsing.                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-05-01 | 1.1.119 | Replaced the ODESolverCalculator data-table `.filter().map()` chain with a single-pass `for` loop that pre-allocates a result array and iterates in steps, eliminating O(N) intermediate array allocations and reducing GC pressure during large-dataset renders.                                                                                                                                                                                                                                                                                                   |
| 2026-05-03 | 1.1.122 | Optimized row copying logic in useDataProcessor.ts by replacing the slow `for...in` and `hasOwnProperty` check with `Object.keys()` and a standard `for` loop, eliminating prototype chain crawling overhead.                                                                                                                                                                                                                                                                                                                                                       |
| 2026-05-03 | 1.1.123 | Hardened Folder Packer Pro archive extraction against absolute and parent-traversal member paths, made vessel drafter positive-value contracts accept both legacy and fleet-standard argument order, repaired the production Docker wheel build/install path, expanded Docker context cache exclusions, made CI quality-gate jobs informational, and lengthened Jules issue resolver polling.                                                                                                                                                                       |
| 2026-05-01 | 1.1.118 | Bound the CI Standard workflow's dependency bootstrap to `python -m pip` in both quality-gate and test-matrix jobs so pytest plugins, including `pytest-benchmark`, install into the same interpreter that later runs `python -m pytest`.                                                                                                                                                                                                                                                                                                                           |
| 2026-05-01 | 1.1.117 | Made the shared syngas water vapor-pressure helpers return explicit `float` values so delta `mypy` checks stay green while preserving the `water_fraction` compatibility alias for downstream consumers.                                                                                                                                                                                                                                                                                                                                                            |
| 2026-05-01 | 1.1.116 | Tightened signal generator and acid gas dewpoint precondition handling so short chirp inputs, zero-frequency periodic signals, and non-positive dewpoint partial pressures raise deterministic `ValueError` messages.                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-30 | 1.1.115 | Hardened CI packaging and workflow checks by pinning the setuptools build backend below 82, using the supported package-data wildcard for `py.typed` markers, scanning merge-conflict markers with tracked-file `git grep`, normalizing detect-secrets result comparisons, and tolerating missing or empty benchmark JSON artifacts.                                                                                                                                                                                                                                |
| 2026-04-30 | 1.1.114 | Integrated full-text live search into the Unified Tools Launcher tabs, including name, description, keyword, multi-word, and punctuation-normalized matching, with Ctrl+F focus and Esc clear shortcuts.                                                                                                                                                                                                                                                                                                                                                            |
| 2026-05-24 | 1.1.113 | Fixed a vulnerability in CSRF cookie parsing logic where cookies with values containing an equals sign were previously being truncated. This allows base64 encoded CSRF tokens with padding to be parsed correctly.                                                                                                                                                                                                                                                                                                                                                 |
| 2026-05-11 | 1.1.127 | Replaced `.map()` array allocations in the `rk4Step_golfer` numerical integration function with pre-allocated arrays and standard `for` loops in `physics_golfer.ts` to reduce GC overhead.                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-05-15 | 1.1.180 | Replaced `.map()` array allocations inside `physics_golfer.ts` constraint and torque loops with pre-allocated arrays and standard `for` loops to reduce GC overhead.                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-05-13 | 1.1.141 | Made the migrated Video Analyzer installable and launchable from Tools for issue #2585 by adding package discovery, a `video-analyzer` console script, optional video runtime dependencies, installed-package import paths, and focused packaging/launcher regression tests.                                                                                                                                                                                                                                                                                        |
| 2026-05-13 | 1.1.140 | Registered the migrated Video Processor web surface in the canonical GUI launcher manifest and generated tools catalog, with regression coverage proving shared UpstreamDrift-visible tools expose their expected launch surfaces (#2585).                                                                                                                                                                                                                                                                                                                          |
| 2026-05-12 | 1.1.139 | Refreshed the module-size budget baseline for the updated rotation converter PyQt launcher after the branch was brought current with main.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-05-15 | 1.1.139 | Refactored RK4 expression compilation in ODESolver to pass parameters as a direct array, avoiding spread operator allocation in tight integration loops.                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-05-15 | 1.1.139 | Refactored RK4 expression compilation in ODESolver to pass parameters as a direct array, avoiding spread operator allocation in tight integration loops.                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-05-12 | 1.1.138 | Hardened CI test-matrix dependency setup against stale self-hosted runner NumPy/SciPy binary caches and routed provider-contract tests through the active Python interpreter.                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-05-12 | 1.1.137 | Corrected the coverage policy gate to ratchet from the committed total-coverage baseline until the repository reaches the configured 60% target, while preserving package thresholds and regression checks.                                                                                                                                                                                                                                                                                                                                                         |
| 2026-05-12 | 1.1.136 | Resolved type-checking errors by properly implementing abstract methods (send_message, validate_connection, capabilities) for RustAgentAdapter, and fixed GUI theme and categorization issues in UpstreamDrift chat functionality.                                                                                                                                                                                                                                                                                                                                  |
| 2026-05-19 | 1.1.184 | Replaced `.reduce()` with a standard `for` loop in `calculatePhaseConfidence` to eliminate callback allocation and garbage collection overhead during high-frequency pose frame confidence calculations in the video processor.                                                                                                                                                                                                                                                                                                                                     |
| 2026-05-20 | 1.1.193 | Clarified shared chat provider dropdown ownership by removing stale UpstreamDrift issue references from Tools-owned source and tests, and synchronized the GitHub CLI provider descriptor with the default terminal registry (#3020).                                                                                                                                                                                                                                                                                                                               |

---

<!--
  SPEC MAINTENANCE RULES:

  1. WHEN TO UPDATE: Any PR that adds, removes, or changes functionality
     described in this spec MUST include a corresponding spec update.

  2. WHO UPDATES: The PR author (human or agent) is responsible.

  3. CI ENFORCEMENT: The spec-check workflow will flag PRs where source
     files changed but SPEC.md did not. This is a blocking check.

  4. REVIEW: Spec changes should be reviewed with the same rigor as code.

  5. VERSION: Bump the Spec Version field when making substantive changes.
     Use semver: major (structure change), minor (new features), patch (corrections).
-->

### Performance

- `getTimeDelta` calculations inside tight loops use `Date.parse(dateString)` instead of `new Date(dateString).getTime()` to directly retrieve numeric timestamps without the memory overhead and GC pressure associated with instantiating temporary `Date` objects.
- Data-processing formula evaluation now treats `numexpr` as an optional accelerator rather than a hard runtime dependency. Shared `DataProcessor` and `upstream_drift_tools` formula columns fall back to the pandas Python eval engine when `numexpr` is unavailable, preserving the documented `TransformationError` contract for invalid expressions.
- The application uses `Float64Array` and iterative loops instead of `Array.prototype.map`/`filter`/`reduce` to optimize memory and processing speed for large numerical datasets, including reusable typed-array buffering for median-filter windows in `useDataProcessor.ts`. Chained array functional methods like `reduce` and `map` have been largely replaced with standard iterative loops in mathematical computation methods such as `zScoreFilter`, `linearRegression` and `polynomialRegression`.
- Mathematical matrix calculations such as Principal Component Analysis (PCA) utilize column-wise typed arrays (e.g. `Float64Array` buffers) rather than traditional N x P row-wise arrays, drastically reducing O(N) allocation overheads and mitigating garbage collection pauses on large scale analysis.
- Linear regression and sum-of-squares calculations in `AnalyticsSuite` leverage pre-allocated arrays and single-pass loops to prevent allocation and garbage collection overhead typical of functional `.map()` and `.reduce()` operations in large dataset pipelines.
- The PCA power iteration algorithm in `AnalyticsSuite` has been optimized to remove `.map()` and `.reduce()` from the tight inner loop, using pre-allocated arrays and standard `for` loops to eliminate thousands of allocations per execution.
- PlotView WebGL rendering uses pre-allocated `Float64Array` buffers and single-pass loops instead of `data.map()`, eliminating O(N) intermediate array allocations for large datasets.
- Pearson correlation matrix computations utilize a single-pass loop algorithm, calculating sums concurrently to drastically reduce iteration overhead compared to two-pass implementations, while carefully mitigating numerical instability via clamping.
- Recharts component props in `AnalyticsSuite` are memoized using `useMemo` hooks to provide stable references and prevent expensive internal re-renders.
- Exponential and power trendline calculations use pre-allocated arrays and single-pass loops instead of functional chaining to minimize GC pauses.

### Version 1.1.67

- **Performance**: Optimized array allocations in PCA calculate loop inside `AnalyticsSuite.tsx` by replacing chained `.reduce()` and `.map()` calls with single-pass `for` loops.

### Version 1.1.66

- **Performance**: Optimized row copying logic inside `useDataProcessor.ts` by replacing `Object.keys()` iterations with `for...in` loops and `hasOwnProperty`. This minimizes excessive key array allocations inside data transformation loops.

### Version 1.1.66

- **Security**: Disabled loading and saving of `.pkl` and `.pickle` files natively using pandas due to severe CWE-502 vulnerability. Raises `ValueError` explicitly when format is set to `pickle`.

### Version 1.1.106

- **Performance**: Optimized matrix and loading array copying inside `AnalyticsSuite.tsx` for PCA calculation by replacing `.map()` and array spread operations with single-pass pre-allocated loops, substantially reducing memory allocation overhead.

## 2026-04-20

- Update unit converter clear history button accessibility (ARIA labels, disabled state)

### Version 1.1.111

- **Performance**: Optimized signal statistics and FFT chart data generation in `FunctionGenerator.tsx` by replacing the use of the array spread operator (`...vals`) inside `Math.min`/`Math.max` and chained iterators (`.map().filter()`, `.reduce()`) with single-pass `for` loops. This prevents runtime "Maximum call stack size exceeded" errors and significantly reduces GC overhead.
- **Performance**: Replaced `.map()` and `.push()` with pre-allocated single-pass `for` loops in `pcaScatterData`, `regressionScatterData`, and `regressionResidualsData` within `AnalyticsSuite.tsx` to eliminate dynamic resizing overhead and intermediate object allocations.
- **Security**: Fixed DOM-based Cross-Site Scripting (XSS) vulnerability in `psa_calculator.html` by implementing and applying an `escapeHtml` function to user-controlled inputs before updating `innerHTML`.
- **Performance**: Optimized `computeFFT` inside `FunctionGenerator.tsx` by pre-allocating output arrays and substituting functional iterations (`.map()` and `.reduce()`) with an inline single-pass Hanning window loop. This bypasses intermediary array processing steps and lowers garbage collection occurrences.
- **Performance**: Replaced O(N) chained `.filter().map()` iterators with an $O(N/\text{step})$ `for` loop in `FunctionGenerator.tsx` when preparing data points for time charts to prevent the allocation of intermediate arrays and reduce unnecessary iterations.

## Performance Optimizations

- **Performance**: Optimized `psa_calculator.html` by replacing chained `.map()` and `.reduce()` operations with single-pass `for` loops and pre-allocated arrays, alongside substituting `reduce` with a globally scoped `sumArray` helper function, significantly reducing GC overhead.

- **ODESolverCalculator:** Replaced `.map()` and `Math.max(...values)` with a single-pass `for` loop to prevent "Maximum call stack size exceeded" errors on large dataset arrays generated by the ODE solver.
- **Performance**: Optimized the sliding-window algorithm `smoothAngles` in the video processor's `angleCalculator.ts` by replacing `.slice()` and `.reduce()` inside the loop with a single-pass sum tracker, and splitting the loop into three parts (left, middle, right) to eliminate `Math.min()` and `Math.max()` bounds checking from the hot path.

### Web Frontends

- `data_processor`: Improved performance in tight object loops by replacing `for...in` and `hasOwnProperty` with `Object.keys()` and standard `for` loops in `useDataProcessor.ts`.
- **Performance**: Replaced `.map()` with pre-allocated arrays and single-pass `for` loops across all signal generation functions (`generateSinusoid`, `generateCosine`, `generateSquare`, `generateTriangle`, `generateSawtooth`, `generatePulse`, `generateStep`, `generateExponential`, `generateLinear`, `generateChirp`, `generateConstant`) in `FunctionGenerator.tsx` to minimize garbage collection overhead for large sample arrays.
- **Performance**: Optimized `generatePolynomial` in `FunctionGenerator.tsx` by using Horner's method and pre-allocating output arrays instead of `.map()` with `Math.pow()`, significantly reducing overhead and improving calculation speed.

### Version 1.1.128

- **Performance**: Pre-allocated objects and arrays for the Runge-Kutta 4 (RK4) integration loop in `ODESolverCalculator.tsx` to eliminate thousands of memory allocations per step and reduce severe garbage collection pauses during large ODE simulations.
- **Performance**: Refactored dynamically compiled expressions inside the hot RK4 numerical integration loop to avoid the spread operator (`...args`) and array allocation. Parameters are now passed as a single array and statically destructured within the function body itself.

### Version 1.1.139

- **Performance**: Optimized `detectSwingPhases` inside `src/media_processing/video_processor/apps/web/lib/golf/phaseDetector.ts` by replacing `poseFrames.map(...)` with a standard single-pass `for` loop and a pre-allocated array. This reduces continuous callback allocation and limits garbage collection pauses in hot paths when analyzing multiple video frames.

### Version 1.1.183

- **Performance**: In high-frequency algorithmic optimization loops (like Nelder-Mead iterations), replaced array manipulation operations such as `.map()` and `.slice()` with pre-allocated arrays and standard `for` loops in `src/pendulum_simulator/pendulum-web/src/optimizer.ts` to eliminate continuous array creation and avoid significant garbage collection overhead.

### Version 1.1.184

- **Performance**: In `src/pendulum_simulator/pendulum-web/src/components/AnalysisPlots.tsx`, optimized chart downsampling by replacing multiple instances of `indices.map()` with pre-allocated arrays and explicit `for` loops inside `useMemo` hooks. This drastically reduces array allocation and garbage collection overhead during high-frequency component rendering.
- **Performance**: Replaced `.reduce()` with a standard `for` loop in `calculatePhaseConfidence` inside `src/media_processing/video_processor/apps/web/lib/golf/phaseDetector.ts` to eliminate callback allocation and garbage collection overhead during high-frequency pose frame confidence calculations.

### Version 1.1.185

- **Security**: Fixed a command injection vulnerability in `cli_tools.py`'s `ShellTool._is_command_allowed` by parsing the command with `shlex.split` and blocking shell operators instead of using a naive `.startswith()` string check.

### Version 1.1.186

- **Performance**: In `src/media_processing/video_processor/apps/web/lib/golf/phaseDetector.ts`, replaced `.reduce()` with a standard `for` loop in `calculatePhaseConfidence` to eliminate callback allocation and GC overhead in high-frequency phase detection paths.

### Version 1.1.187

- **UX**: Add accessible toggle states and toast feedback for copy actions in `calculator/static/app.js` and `calculator/templates/index.html`.

### Version 1.1.188

- **Security**: Fixed an information leakage vulnerability in `src/web_applications/health_checks.py`. API endpoints (`/api/health` and `/api/ready`) no longer expose raw exception strings (`str(e)`) in their JSON responses. They now return safe, generic error messages while preserving full traceback details in the backend logs using `logger.exception()`.

### Version 1.1.189

- **Reliability**: Restored source-tree `src.shared.python.logging_pkg` and `src.shared.python.config` compatibility modules so shared AI adapter factories and chat service connection code import cleanly from a Tools source checkout or vendored shared-module install.

## 9. Changelog

### Version 1.1.190

- **2026-05-24**: Add `spellcheck="false"`, `autocorrect="off"`, and `autocapitalize="none"` to math inputs in calculator to improve UX, and add `role="img"` to battery icon span.

- **Performance**: In `src/ode_solver/web/src/components/ODESolverCalculator.tsx`, wrapped `varNames` computation and summary cards rendering in `useMemo`, and replaced `.filter()` with a single-pass `for` loop to prevent O(N) recalculations of array keys and summary min/max loops on every React render.

- **2026-05-22**: Memoize summary statistics calculation and variable names in ODESolverCalculator.
- **2026-05-22**: Keep the model explorer package initializer lint-clean by preserving the module docstring before future imports.
- **2026-05-20**: Suppress shared chat dock WebSocket reconnect scheduling during intentional widget close while retaining reconnects for unexpected disconnects.
- **2026-05-20**: Add accessible toggle states and toast feedback for copy actions in `calculator/static/app.js` and `calculator/templates/index.html`.
- **2026-05-20**: Harden health-check API responses to return generic client-facing errors while logging exception details server-side.
- **2026-05-20**: Restore shared logging and environment helper modules required by AI adapter and chat service connection imports.
- **2026-05-20**: Clarify shared chat provider dropdown ownership by removing stale UpstreamDrift issue references from Tools-owned source and tests, and synchronize the GitHub CLI provider descriptor with the default terminal registry (#3020).
