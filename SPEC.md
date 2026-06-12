# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST_UPDATED: 2026-06-01

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
| **Spec Version**        | 1.1.397                                    |
| **Last Spec Update**    | 2026-06-12                                 |

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
- P1AM power-supply control owns its PLC PID-pass-through writes, tag scaling,
  state-machine controller, and REST routes in a dedicated backend integration
  module so `backend/main.py` remains below the module-size budget
- Sidekick tabs declare versioned per-tab settings schemas and persist
  materialized settings by stable tab or duplicate instance id behind the
  selected-tab settings action
- Shared chat history rows use wrapped readable item widgets with transparent
  icon-only archive, restore, and delete controls available without right-click
- Shared chat dock close control lives in the persistent status header instead
  of the terminal provider control row
- Shared chat dock delegates workspace slash-commands (`/ws.read`, `/ws.write`,
  `/plot`) and AI provider/model/thinking settings to headless, Qt-free
  controllers (`WorkspaceCommandHandler`, `AiSettingsController`) per ADR-0022,
  enabling unit tests without a QApplication
- Shared unified tools sidebar widgets provide optional dockable/tear-off host
  integration for project file browsing, workspace variables, chat, terminal,
  calculator, unit conversion, and notes tabs
- Sidekick runtime tabs embed real utility surfaces for chat status, workspace
  Python execution, symbolic calculator evaluation, and project-persistent
  notes instead of placeholder panels
- Sidekick sidebar configuration extends the shared sidebar with persisted
  left/right docking, minimized state, tab order, hidden tabs, popped-out tab
  tracking, duplicate tab instances, and host-provided tab definitions
- Sidekick agent action dispatch lives canonically under
  `src/shared/python/sidekick/agent`, with audited action registration,
  headless host/subtab ports, and an optional thunk dispatcher compatible with
  the shared AI main-thread tool dispatcher for GUI-affine actions
- Sidekick agent canonical modules are protected by focused unit coverage for
  audit sinks, feature catalog discovery/search, host capability dispatch,
  planner validation/export, and subtab action dispatch so the per-file
  Sidekick coverage gate can block untested drift
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
- Sidekick Python REPL registry preconditions accept both canonical
  `sidekick` and deprecated `upstream_drift_tools` `WorkspaceRegistry` module
  identities during the compatibility migration while preserving explicit
  TypeError failures for missing or unrelated registry objects
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
  Cline CLI, Gemini CLI, and GitHub CLI, including probe command metadata with
  diagnostic redaction helpers
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
| Movement Optimizer       | `src/optimizer_gui/`                                                                   | Standalone PyQt6 movement optimizer exposing Adam optimization plus side-view swingset policy training and segmented chain whip-dynamics analysis tabs with adjustable segment lengths, masses, chain counts, and provider metadata for UpstreamDrift launcher tiles                                                                                                                                                                                                                 |
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
- GUI discovery (`gui_launcher.registry.auto_discover_guis`) isolates each
  `gui_registration.py`: any per-file failure (import error, malformed
  `GUI_INFO`) is logged and skipped rather than aborting discovery for every
  tool, and the returned count reflects only successful registrations.
- Saved-state JSON persistence is atomic. `utils.file_utils.atomic_write_text`
  (temp file + `fsync` + `os.replace`) backs `safe_write_json`, and
  `StateManager` routes all state writes through it, so a crash / disk-full
  mid-write leaves the prior file intact rather than truncated.
- Sidekick data I/O closes sqlite connections via `contextlib.closing` on both
  the read and write paths, bounding the connection lifecycle to the call even
  when the query/`to_sql` raises.
- The web-app launcher uses a bounded socket readiness probe (no fixed sleep)
  before opening the browser, and reaps the dev-server child on Ctrl-C
  (terminate → wait → kill) returning a non-zero exit code so no child outlives
  the call.
- The Sidekick calculator plot evaluator routes expressions through the
  AST-validated `safe_eval` (attribute/dunder traversal rejected at parse time)
  instead of raw `eval`.
- AI CLI adapters resolve binaries via `shutil.which` plus home-relative
  fallbacks (no hardcoded usernames); fleet scripts derive their repo/repos root
  from `--repos-root` / env / `__file__` and exit non-zero when it cannot be
  determined (headless-safe).

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

**Cross-repo import contract:**

Downstream consumers (e.g. UpstreamDrift's `external_tools_adapter`) import
this repository by placing the **repository root** on `sys.path` and importing
packages under the `src.` namespace (`import src.<package>`). Top-level `src/`
packages MUST therefore be importable with only the repository root on
`sys.path` — they may not depend on the test-only `pythonpath` shims (`src`,
`src/shared/python`) or on the editable-install finder being present. Concretely:

- Package `__init__` modules use package-relative imports (`from .x import ...`)
  rather than bare, ambiguous-root names (`from <pkg>.x import ...`).
- Optional heavy runtime dependencies (e.g. `cv2`, `mediapipe`, `sidekick`) are
  imported lazily (PEP 562 `__getattr__`) so that importing a package — and
  reaching its version/type metadata or declared console-script entry point —
  never requires those optional dependencies.

This contract is enforced by the subprocess import-contract tests
(`tests/test_src_package_import_contract.py`,
`tests/video_analyzer/test_video_analyzer_import_contract.py`), which reproduce
the consumer's clean `sys.path` so a regression turns CI red here rather than
crashing the consumer at runtime.

Rotation-converter NumPy boundaries and the video-analyzer DbC shim remain
mypy-clean under the changed-file CI profile while preserving runtime
validation through explicit `require`/`ensure` checks and stable fallback
imports.

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
Changed Python test files must contain at least one AST-visible behavioral
assertion, exception assertion, or unittest/mock-style assertion call unless
they match the explicit fixture/support-only assertion allowlist.
Critical numerical contracts are additionally guarded by property-based tests
(Hypothesis) that assert invariants — round-trip identity, linearity, and
boundary/failure behavior — rather than only example outputs. The flow-rate
conversion API (`calc_backend`) carries such a suite
(`test_calc_backend_properties.py`); `hypothesis` is a declared `dev`
dependency. New adversarial coverage targets invalid inputs, non-finite values,
and missing fields so a regression fails CI here rather than downstream.
When CI detects changes under `src/shared/python/sidekick/`, its focused Python
test slice includes the dedicated Sidekick state-manager suites before the
per-file Sidekick coverage gate runs, so coverage enforcement measures the
module's own regression tests instead of an unrelated reduced slice.

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

| Scope                  | Minimum                              | Current                                                                         | Enforced By                             |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------- | --------------------------------------- |
| Overall                | 60% target, current-baseline ratchet | 24.48% baseline                                                                 | CI (`scripts/check_coverage_policy.py`) |
| Core tools             | 75%                                  | ~81%                                                                            | CI                                      |
| Plugin system          | 80%                                  | ~85%                                                                            | CI                                      |
| Codemap package        | 90%                                  | 97.72% focused coverage                                                         | CI (`scripts/check_coverage_policy.py`) |
| File watcher fallback  | 95%                                  | 99.46% focused coverage                                                         | CI (`scripts/check_coverage_policy.py`) |
| Upstream drift shim    | 100%                                 | 100% focused coverage                                                           | CI (`scripts/check_coverage_policy.py`) |
| Folder packer ops      | 90%                                  | 92.95% focused coverage                                                         | Focused pytest coverage                 |
| Model-gen URDF/inertia | 80%                                  | primitives 91%, spatial 100%, urdf_parser 85%, format_utils 75%, validation 93% | Focused pytest coverage                 |

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
- [x] Movement Optimizer launches standalone and exposes tested swingset and
      chain-dynamics tabs with 100% focused model coverage

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
- Web API authentication/authorization: the P1AM control backend now gates all
  mutating endpoints behind an `X-API-Key` credential (`auth_config.py`); other
  web apps' APIs may still lack auth.

## 12. Change Log

<!-- prettier-ignore-start -->

| Date | Version | Changes |
| ---- | ------- | ------- |
| 2026-06-12 | 1.1.397 | test(conversion, #3384 #3388 #3389): add shared conversion-service policy coverage for normalization, validation, custom-unit warnings, gas-flow dispatch, syngas/performance helpers, and singleton conversion helpers so `src/shared/python/sidekick/calculators/conversion/service.py` stays above the changed-file coverage gate without changing production behavior. |

| 2026-06-12 | 1.1.394 | chore(consolidation): refresh the quality-consolidation branch after the scientific-accuracy merge so the shared Sidekick process-calculator constants, signal calculus guards, and API baseline remain aligned with current main while preserving the data-processor facade split. |
| 2026-06-11 | 1.1.391 | fix(sidekick): keep the Python REPL worker owned by the widget until its QThread has fully stopped, avoiding Linux/offscreen teardown aborts from premature deleteLater scheduling. |
| 2026-06-11 | 1.1.390 | test(ci): keep the Sidekick Python REPL widget below the fleet file-size budget after QThread teardown hardening. |
| 2026-06-11 | 1.1.387 | test(ci): stabilize optional CoolProp symbol patching and data_processor nested-package imports across CI Python environments. |
| 2026-06-11 | 1.1.386 | fix(thermo, #3381 #3382): correct the Buck water vapor-pressure exponent, tighten dew-point regression coverage against published reference points, and add pressure-dependent ideal-gas entropy in the simplified steam vapor fallback. |
| 2026-06-11 | 1.1.385 | fix(calc-backend, #3341): require forward time spans for ODE solver and thermal-profile requests, convert diverging ODE and thermal integrations into 422 validation errors before non-finite values reach JSON responses, and add contract/API regressions for reversed spans and divergent systems. |
| 2026-06-11 | 1.1.384 | fix(steam, #3337 #3338): enforce saturation temperature and pressure preconditions before backend fallback, reject out-of-range simplified saturation states instead of extrapolating Antoine correlations, preserve unknown CoolProp quality as NaN instead of saturated-liquid quality, and map steam API validation failures to HTTP 400. |
| 2026-06-11 | 1.1.383 | fix(unit-converter, #3336 #3339): make gas-flow conversions fail loudly for unknown gas species across Sidekick and web converter surfaces, and align the sidekick compressibility-factor calculation with the Abbott/Pitzer second-virial form used by pressure-drop calculations. |
| 2026-06-11 | 1.1.377 | fix(ci): use an actionlint-compatible relative npm cache path for Tauri jobs while keeping installs isolated from the runner user's shared npm cache. |
| 2026-06-11 | 1.1.376 | fix(ci): isolate Tauri npm caches under the per-job runner temp directory and prefer fresh registry metadata so corrupted shared npm cache entries cannot fail `npm ci`. |
| 2026-06-11 | 1.1.375 | fix(ci): set `fail-fast: false` on the CI Standard `tests` Python matrix. Only `tests (3.11)` is a required check; under the default `fail-fast: true` an infra crash in the non-required 3.10/3.12 lanes (SIGABRT/exit-134 from the Qt headless multi-widget segfault or an OOM kill on a saturated self-hosted runner) cancelled the required 3.11 lane before it ran, leaving consolidation PR #3380 permanently BLOCKED. Decoupling the lanes lets 3.11 report independently. |
| 2026-06-11 | 1.1.374 | fix(ci): keep the Sidekick extended Qt-heavy unit suite on Python 3.11/3.12 while excluding it from the Python 3.10 compatibility lane, where PyQt aborts the interpreter on saturated self-hosted runners. |
| 2026-06-11 | 1.1.373 | fix(ci): make the workflow validation PyYAML fallback explicit for mypy so quality-gate checks accept both full and lean runner environments. |
| 2026-06-11 | 1.1.372 | fix(ci): make workflow lint validation tolerate lean runner environments where PyYAML cannot be fetched by adding stdlib fallback checks for workflow structure and blocking quality gates, while still using PyYAML when present. |
| 2026-06-11 | 1.1.371 | fix(ci): keep the Python 3.10 CI Standard lane focused on core compatibility tests for large consolidation PRs while Python 3.11/3.12 continue to run the full changed-test slice, avoiding 3.10 runner OOM kills during collection. |
| 2026-06-11 | 1.1.370 | fix(ci): remove the network-dependent `actions/setup-python` bootstrap from Topology Governance because the topology checker is a stdlib-only script and can run with the fleet runner's existing `python3`, avoiding transient PyPI/setup-python failures. |
| 2026-06-11 | 1.1.369 | fix(ci): make the Python 3.10 CI Standard test lane override repo-level pytest-xdist auto-parallelism with `-n 0` so saturated self-hosted runners report deterministic test results instead of xdist worker crash exhaustion. |
| 2026-06-11 | 1.1.368 | test(ci): keep data-processor tkinter fallbacks from leaking a partial `tkinter` stub into folder-tool collection by preferring real tkinter when available and installing a complete fallback with `ttk`, `messagebox`, and `filedialog` modules only when needed. |
| 2026-06-11 | 1.1.367 | fix(ci/runner): make Tauri Linux checks discover an available local Node 24, 22, or 20 toolcache on mixed self-hosted runners instead of failing on runners without the exact Node 24.16.0 path. |
| 2026-06-11 | 1.1.364 | test(ci): keep retired data-processor skip sentinels compatible with the ruff B011 guard by using truthy documentation assertions instead of optimized-away `assert False` statements. |
| 2026-06-11 | 1.1.363 | fix(ci/runner): harden `ci-standard.yml` Linux apt setup by clearing corrupted apt package-cache binaries alongside stale lock files before `apt-get update`, allowing self-hosted runners to recover from cache rename failures. |
| 2026-06-11 | 1.1.362 | fix(ci): align `ci-standard.yml` with the fleet's known-good `mypy==1.13.0` workflow pin so quality-gate dependency installation remains reproducible on self-hosted runners. |
| 2026-06-11 | 1.1.361 | test(ci): satisfy the changed-test behavioral assertion gate in the Tools consolidation branch by adding benchmark output postconditions, making retired data-processor skip sentinels explicit, and documenting the shared numerical helper as support-only in the assertion allowlist. |
| 2026-06-11 | 1.1.360 | fix(ci): restore Tools consolidation CI by replacing the coverage tracked-package regex generator with a shell-safe Python expression, adding changed-file mypy annotations for the multi-parameter PyQt meshgrid arrays, and resolving signal-toolkit integration bounds to concrete floats before validation/result construction. |
| 2026-06-11 | 1.1.354 | fix(consolidation, #3314 #3315 #3350 #3356 #3358): restore and relocate truncated test coverage across shared calculators, signal tooling, GUI launchers, folder tooling, data processing, rotation conversion, and integration surfaces; unify humanoid anthropometry under the shared implementation; propagate P1AM E-STOP clear commands through the backend API; refresh assessment artifacts and CI baselines for the consolidated changes. |
| 2026-06-11 | 1.1.354 | test(sidekick, #3339 #3340): add focused pressure-drop gas-property coverage for strict unknown-species DbC paths, physical-value helper contracts, complete gas-property calculation keys, and ideal-gas compressibility fallback so the changed gas helper module is covered by the Sidekick per-file coverage gate. |
| 2026-06-10 | 1.1.353 | fix(hooks, #1361): align the pre-push mypy hook with changed-file delta CI by adding `--follow-imports=skip`, so clean pushes are checked against the pushed source files without failing on unrelated pre-existing imported `ai/` debt. Added an ops regression test that keeps the hook on the pre-push stage, filename-passing mode, `src/` scope, and no-follow-import behavior. |
| 2026-06-10 | 1.1.352 | test(sidekick): keep action-audit timestamp fixtures compatible with the Python 3.10 CI lane by using `timezone.utc` with a scoped pyupgrade suppression instead of the Python 3.11-only `datetime.UTC` alias. |
| 2026-06-10 | 1.1.351 | test(sidekick): keep action-audit redaction fixtures covered while marking synthetic sensitive-key values with detect-secrets allowlist pragmas, so the security scan remains strict without treating redaction test data as leaked material. |
| 2026-06-10 | 1.1.348 | test(ai): keep the #3310 GUI-thread dispatcher coverage mypy-clean under the changed-file gate by annotating the offscreen Qt fixture, worker-thread test parameters, dispatcher thunks, decorator-registered tool dispatch, and exception helper while preserving the main-thread marshalling behavior under test. |
| 2026-06-10 | 1.1.347 | fix(ci/runner): split Tauri build matrix display labels from `runs-on` targets so Windows jobs no longer render as `Array`, and run Windows Rust path/tool-home setup through PowerShell while preserving bash setup on Linux. |
| 2026-06-10 | 1.1.346 | fix(ci/runner, #3308): restore the Tauri 30-minute check timeout on current main after #3307 accidentally reverted the runner hardening while adding the ShellTool command-injection fix. |
| 2026-06-10 | 1.1.345 | fix(ci/runner, #3305): isolate Tauri `RUSTUP_HOME` and `CARGO_HOME` under each job's `RUNNER_TEMP` so parallel self-hosted jobs do not race on the shared `$HOME/.rustup` toolchain and lose `rustc` mid-clippy. |
| 2026-06-10 | 1.1.344 | fix(ci/runner, #3304): disable Tauri Rust `target/` cache restoration while keeping cargo registry/git caching after a fast-I/O runner hit a stale dep-info fingerprint (`time-*.d` missing) during clippy. |
| 2026-06-10 | 1.1.343 | fix(ci/runner, #3304): raise Tauri Rust stack reservations to 512 MiB after function-generator and data-processor clippy on OGLaptop explicitly requested `RUST_MIN_STACK=536870912`, with workflow regression coverage for the stack contract. |
| 2026-06-10 | 1.1.342 | fix(ci/runner, #3304): route Rust-heavy Tauri check and Linux build jobs to the `d-sorg-fleet-fast-io` runner label so PR validation avoids OGLaptop slots that repeatedly hit rustc stack faults while keeping local self-hosted execution. |
| 2026-06-10 | 1.1.341 | fix(ci/runner, #3304): raise Rust stack reservations to 256 MiB after rotation-converter clippy on OGLaptop explicitly requested `RUST_MIN_STACK=268435456`, keeping all Tauri app checks on the same fleet-safe stack setting. |
| 2026-06-10 | 1.1.340 | fix(ci/runner, #3304): raise Rust stack reservations to 128 MiB for local self-hosted Tauri and wheel builds after OGLaptop rustc clippy failures explicitly requested `RUST_MIN_STACK=134217728`. |
| 2026-06-10 | 1.1.339 | fix(ci/runner, #3300): expose `$HOME/.cargo/bin` before Rust toolchain setup in self-hosted Rust jobs so fleet runners use their preinstalled rustup instead of attempting fragile bootstrap installs when non-login shells omit cargo from PATH. |
| 2026-06-10 | 1.1.338 | fix(ci, #3300): raise Rust runner stack reservations to 64 MiB for local self-hosted Tauri and wheel builds after rustc SIGSEGV failures explicitly requested `RUST_MIN_STACK=67108864` on the fleet. |
| 2026-06-10 | 1.1.337 | fix(ci/test-contract, #3300): recognize repo-level `tests/<package>/test_*.py` directories as satisfying the minimum test contract for changed `src/<package>` packages, with regression coverage so package-scoped tests like `tests/plant_simulator/test_dataset.py` are accepted without weakening the quality gate. |
| 2026-06-10 | 1.1.336 | fix(ci/review-comments, #3300): keep the review-comment-to-issue converter checkout shallow because the job uses GitHub API reads plus local archive commits, avoiding full-history fetches on self-hosted runners where stale/corrupt loose objects can make checkout fail before the workflow logic runs. |
| 2026-06-10 | 1.1.335 | fix(ci/runner-health, #3300): serialize the Tauri desktop app check/build matrices and cap Cargo jobs with non-incremental, no-debug builds so self-hosted runners do not compile multiple Tauri Rust dependency graphs concurrently and trigger rustc SIGSEGV/paging-pressure failures. |
| 2026-06-11 | 1.1.359 | chore(consolidation): finish the open-PR consolidation by centralizing Catppuccin stylesheet imports, preserving calc-backend dependency direction, and tightening restored test/type annotations for the changed-file quality gates. |
| 2026-06-11 | 1.1.358 | fix(thermo, #3345): keep saturation-pressure lookups resilient by falling back to the Antoine equation when the optional Cantera water backend raises while preserving explicit failures for invalid fallback inputs. |
| 2026-06-11 | 1.1.357 | fix(ode, #3349): preserve the consolidated `t_span` bounds guard in the Sidekick ODE solver while keeping the merged implementation syntactically valid. |
| 2026-06-11 | 1.1.356 | fix(test, #3315): restore truncated test coverage across P1AM, pendulum, shared-tool, and architecture suites; preserve HMI emergency-stop propagation tests; and reconcile the humanoid/URDF anthropometry consolidation with the shared ratio helpers. |
| 2026-06-11 | 1.1.355 | fix(dry, #3346): remove reintroduced root-level `urdf_builder_gui` duplicate modules and add a regression test that asserts the root package does not shadow the canonical `src/shared/python/urdf_builder_gui` implementation. |
| 2026-06-10 | 1.1.334 | fix(ci/rust, #3291 #3294 #3295): split PyO3 `python` test features from maturin-only `extension-module` wheel linkage so `cargo test --features python` no longer emits Python extension-module binaries while wheel builds still opt into extension-module linking. |
| 2026-06-10 | 1.1.333 | fix(bug/ci, #3294 #3295): declare pendulum `Golfer` dynamics native-only with construction-time `RuntimeError` guidance and an explicit workspace exclude for `pendulum-core`; remove `plant_simulator`'s silent random-data path so `SCADADataset` loads real SQLite `taglog` rows unless synthetic data is explicitly requested; and keep the affected native wrappers mypy-clean under the changed-file quality gate. |
| 2026-06-10 | 1.1.332 | fix(ci, #3298): keep the P1AM project import helper mypy-clean under the changed-file quality gate by typing parsed SCADA tags as `TagDefinition` at the parser boundary and preserving the endpoint's documented `dict[str, Any]` response contract when imports are skipped. |
| 2026-06-11 | 1.1.354 | fix(dbc): harden optimized-mode validation for signal-toolkit derivative guards and Sidekick gas-flow conversion internals. `signal_toolkit` optimized-mode subprocess coverage now preserves the repo shared-python import path, and gas-flow ACFM invariant checks use explicit exceptions instead of runtime `assert` statements so guard behavior remains deterministic under `python -O`. |
| 2026-06-10 | 1.1.331 | fix(ci, #3298): avoid a detect-secrets Secret Keyword false positive in the P1AM backend auth helper by renaming the public header-name constant away from token-like wording and constructing the `X-API-Key` header name without changing the HTTP authentication contract. |
| 2026-06-10 | 1.1.331 | fix(daemon, #3291): stop `start-gaai-daemon.sh` from writing `~/.claude/settings.json` or globally suppressing Claude Code dangerous-mode prompts; document that any safety override must be configured deliberately outside the launcher, and add a dry-run regression test proving existing global Claude settings are preserved. |
| 2026-06-09 | 1.1.329 | fix(security, #3288 #3289 #3292): remove the P1AM HMI hardcoded default Admin password and accepted hardcoded SHA-256 hashes, fail closed when no credential is configured, and verify admin passwords with a salted PBKDF2-HMAC-SHA256 KDF (`ADMIN_PASSWORD_HASH`/`ADMIN_PASSWORD`) instead of bare SHA-256; add server-side `X-API-Key` authentication/authorization to the P1AM control backend (`auth_config.py`) so every state-mutating endpoint and the live WebSocket require an operator key and destructive/elevated operations (estop clear, tag writes, PID tuning, MPC, alicat setpoint/gas, project import) require an admin key, failing closed (503) unless `P1AM_DEV_NO_AUTH=1`, with E-stop activation intentionally left open and the Docker default bind changed to loopback; and harden `/api/project/import` against unbounded uploads (streamed size cap -> 413), zip bombs (member-count/per-file/total-size/compression-ratio limits before extraction), and partial DB wipes (atomic delete+insert in one transaction). |
| 2026-06-09 | 1.1.329 | fix(security, #3290 #3293): add static complexity limits to `shared.python.safe_eval.validate_expression` (max expression length, max AST node count, bounded `Pow` exponent and nested-`Pow` chain depth, and rejection of oversized string/bytes constants) so pow/repetition bombs such as `9**9**9**9` fail fast instead of hanging or exhausting memory in the calc-backend ODE-solver path; and replace the web calculator's substring blocklist with a structural AST allowlist gate (`TI89Calculator._ast_security_gate`) that runs before `sympy.parse_expr`, rejecting attribute access, lambdas, comprehensions, and the walrus operator by structure rather than enumeration. Adds bypass/DoS regression tests. |
| 2026-06-09 | 1.1.328 | fix(ci): satisfy the changed-file quality gate by explicitly annotating access-policy registry results under skipped-import mypy, add Python 3.10 `tomli` support for metadata contract tests, assert calc-backend pressure-drop values through the standardized response `data` payload, and keep Sidekick standard responses importable from the repo package path without top-level path shims. |
| 2026-06-09 | 1.1.327 | fix(compatibility-ci): route remaining Python 3.10-exercised `StrEnum` imports through compatibility shims, make those shims type-check as native `StrEnum` under mypy while retaining Python 3.10 fallbacks, keep the integrations dashboard empty-state property explicitly typed as `bool`, and pass `.secrets.baseline` explicitly to the detect-secrets audit test so the 3.10 CI matrix validates the canonical baseline instead of failing on CLI argument parsing. |
| 2026-06-09 | 1.1.326 | ci(coverage): keep total coverage floors as a full-suite ratchet while changed-file scoped PR runs enforce only the tracked coverage-policy packages touched by the diff; added regression coverage for the scoped/full-suite split. |
| 2026-06-09 | 1.1.325 | test(calc-backend): add an adversarial route-list contract test ensuring every endpoint advertised by `/api/calc/endpoints` is backed by a registered FastAPI route, strengthening the #3262 calc_backend test-quality audit follow-up. |
| 2026-06-09 | 1.1.324 | fix(ci): invoke detect-secrets through `python -m detect_secrets` in the secret scanning workflow so runners where the console script is not on PATH still execute the installed package. |
| 2026-06-09 | 1.1.323 | fix(ci): avoid detect-secrets false positives from immutable workflow digest pins and workflow-pinning test fixtures without changing the committed secrets baseline. |
| 2026-06-09 | 1.1.323 | test(tools): add changed-test assertion and changed-Python policy guards for the A-O audit follow-up, blocking assertion-light Python test changes and undocumented changed-file policy regressions with focused tests, allowlists, CI integration, and development notes for issues #3262 and #3263. |
| 2026-06-09 | 1.1.322 | fix(ci): fold #3255 pinning into the consolidated branch by requiring third-party workflow actions to use immutable 40-character SHAs, allowing first-party `actions/*` and `github/*` tag refs as the explicit trust boundary, blocking `curl|sh` installers and unversioned global npm installs without a baseline, keeping wasm-pack on a pinned release archive with SHA-256 verification, and pinning Jules CLI installs to `@0.1.42`. |
| 2026-06-09 | 1.1.321 | fix(ci): add a blocking workflow pinning ratchet, replace wasm-pack `curl | sh` installers with a pinned release archive plus SHA-256 verification, add pip retry/timeout settings for CI dependency installs, add a blocking quality-gate verifier for core Ruff/format/mypy PR gates, and split Sidekick data I/O format detection into a dedicated registry module with property/adversarial coverage. |
| 2026-06-09 | 1.1.320 | fix(policy): remove the broken `dwsim-model` console entry, stop allowing the committed coverage baseline to lower the configured coverage floor, align root package docs with the Python 3.11 metadata floor, constrain Sidekick data I/O advertised formats to implemented handlers with focused round-trip coverage, and require the NPM publish job to use the protected `npm` environment. |
| 2026-06-04 | 1.1.318 | test(gui-launcher): add focused unit coverage for shared GUI launcher factory helpers, including launcher construction, generated launch scripts, registered-tool dispatch, missing registry entries, missing PyQt6 configs, module import errors, missing `GUI_INFO`, and successful `GUI_INFO` launch delegation, raising `src/shared/python/gui_launcher/launcher_factories.py` focused coverage from 15.52% to 98.28%; also preserve the declared integer return contract for delegated PyQt6 launch helpers. |
| 2026-06-04 | 1.1.317 | test(gui-launcher): add focused unit coverage for the shared GUI registry, including singleton access, registration validation, lookup/listing/category behavior, helper registration, GUI_INFO conversion, auto-discovery of registration modules, missing paths, import-error handling, and empty legacy modules, raising `src/shared/python/gui_launcher/registry.py` focused coverage from 0.00% to 97.96% without changing production behavior. |
| 2026-06-04 | 1.1.316 | test(gui-launcher): add focused unit coverage for the shared GUI manifest loader, including bundled manifest loading, custom manifest parsing, debug logging, missing files, malformed YAML, missing `tools` mappings, non-sequence `tools` values, and empty manifests, raising `src/shared/python/gui_launcher/manifest_loader.py` focused coverage from 0.00% to 100.00% without changing production behavior. |
| 2026-06-04 | 1.1.315 | fix(compatibility-tests): keep shared Python compatibility coverage importable on Python 3.10 by asserting the UTC fallback through `datetime.timezone.utc`, avoiding Python 3.11-only `enum.StrEnum` references, and preserving Ruff and mypy cleanliness. |
| 2026-06-03 | 1.1.314 | test(compatibility): add focused unit coverage for shared Python compatibility helpers, including Python 3.11+ standard-library alias exports and isolated Python 3.10 fallback behavior for UTC and StrEnum compatibility, raising `src/shared/python/compatibility.py` focused coverage from 0.00% to 100.00% without changing production behavior. |
| 2026-06-03 | 1.1.313 | test(deprecation): add focused unit coverage for shared deprecation helpers, including decorator configuration validation, metadata preservation, warning text variants, method-qualified warnings, and wrapped callable result propagation, raising `src/shared/python/deprecation.py` focused coverage from 0.00% to above 90% without changing production behavior. |
| 2026-06-03 | 1.1.312 | test(logging): add focused unit coverage for shared logging helpers, including package exports, sensitive-value redaction, stream/file logging setup, quiet-library defaults, file and rotating handlers, deterministic seeding, and execution-time telemetry, raising `src/shared/python/logging_pkg` focused coverage from 0.00% to above 90% without changing production behavior. |
| 2026-06-03 | 1.1.311 | test(config): add focused unit coverage for shared environment configuration helpers, including package exports, missing/default/required reads, whitespace handling, boolean parsing, integer/float parsing, bounds errors, and structured `EnvironmentError` details, raising `src/shared/python/config` focused coverage from 0.00% to above 90% without changing production behavior. |
| 2026-06-03 | 1.1.310 | test(chat-export): add focused pure-Python coverage for shared chat export contracts, scanner-safe secret redaction fixtures, markdown/text/html file exporters, and injected clipboard copy modes, raising `src/shared/python/chat/export` focused coverage from 0.00% to 92.79% without changing production behavior. |
| 2026-06-09 | 1.1.310 | perf(p1am frontend): optimize array aggregations and string operations in LadderExplorer.tsx by replacing chained .map().filter() operations with a single-pass loop and using useMemo to prevent main thread lag. |
| 2026-06-03 | 1.1.309 | fix(p1am-power-supply): move the power-supply controller/router and PID-pass-through integration out of `backend/main.py`, keep the split power-supply tests importable under pytest importlib mode, make the controller enums Python 3.10-compatible and mypy-clean, remove stale mypy suppressions from the invalid-input tests, and preserve the module-size budget without relaxing CI gates. |
| 2026-06-03 | 1.1.308 | test(folder-packer): add focused workflow coverage for `folder_packer_pro.operations`, including pack/unpack start validation, worker dispatch, scan dispatch, filesystem exception handling, failed unpack results, encrypted package inspection, and missing package warnings; raises focused module coverage from 74.27% to 92.95% without changing production behavior. |
| 2026-06-03 | 1.1.307 | test(model_generation): add focused edge-case coverage for `model_generation.library.unified_loader`, including load-result naming, preference corruption and persistence failures, manifest cache fallbacks, bundled missing-file reporting, unknown-extension fallback ordering, inline XML conversion dispatch, and malformed MJCF `LoadResult` handling; fixes malformed MJCF loads so they return a failed `LoadResult` instead of escaping parse exceptions, while keeping the loader source under the file-size budget. |
| 2026-06-03 | 1.1.306 | test(upstream-drift): ratchet the legacy `upstream_drift_tools` compatibility shim coverage gate to 100% after focused shim contract tests verified full line and branch coverage, and update the coverage-policy regression tests so the high-water mark is enforced in CI without changing production behavior. |
| 2026-06-03 | 1.1.305 | test(model_generation): add focused coverage for `model_generation.library._rate_limiter`, including rate-limit header parsing, success logging, request header propagation, capped exponential backoff, terminal 429 handling, non-429 HTTP passthrough, and retried network failures; raises the focused module coverage from 53.12% toward the phase-2 model-generation coverage target without changing production behavior. |
| 2026-06-03 | 1.1.304 | test(financial-calculator): add focused PyQt6 contract coverage, split across line-budgeted GUI test modules, for financial calculator import isolation, theme-manager test isolation, successful engine result/projection mapping, notes-dock toggling, summary label rendering, projection table rendering, and calculate-button refresh behavior, raising `src/financial_calculator/python/financial_calculator/ui/pyqt6/main_window.py` focused coverage to 95.28% and the focused `src/financial_calculator` package coverage to 90.53% without changing production behavior. |
| 2026-06-03 | 1.1.303 | test(codemap): add focused headless coverage for the `codemap-mcp` server entrypoint, including `CODEMAP_REPO_ROOT` discovery, missing optional `mcp` dependency handling, server run dispatch, and fake FastMCP tool delegation for search, symbol lookup, callers, imports, and repo summary; raises `src/shared/python/codemap/mcp_server.py` focused coverage from 0.00% to 100.00% and `src/shared/python/codemap` focused package coverage from 94.39% to 97.72% without changing production behavior. |
| 2026-06-03 | 1.1.302 | fix(ai-skills): run shared AI skills runner coroutine tests through explicit `asyncio.run(...)` calls and handle Python 3.10 `asyncio.TimeoutError` in the runner timeout boundary so timeout failures are consistently classified as structured `timeout` audit events. |
| 2026-06-03 | 1.1.301 | test(ai-skills): add focused contract and failure-path coverage for the shared AI skills runtime, including concrete-skill descriptor enforcement, duplicate instance registration, structured execution-error audit classification, and required descriptor field normalization, raising `src/shared/python/ai/skills` focused coverage from 90.42% to 96.17% without changing production behavior. |
| 2026-06-03 | 1.1.300 | test(codemap): add focused CLI coverage for rebuild, search, who-calls, export, and info command paths using mocked API/indexer seams plus real SQLite JSONL/gzip export verification, raising `src/shared/python/codemap` focused package coverage to 94.39% and adding a 90% tracked coverage policy gate. |
| 2026-06-03 | 1.1.299 | test(file-watcher): add focused deterministic coverage for the Python watchdog fallback covering constructor contracts, callback dispatch failures, debounce coalescing, ignore rules, fake watchdog lifecycle handling, missing optional dependencies, and no-op flush branches, raising `src/shared/python/file_watcher/_fallback.py` focused coverage to 99.46% with a 95% file-level coverage policy gate. |
| 2026-06-03 | 1.1.298 | test(signal-toolkit): add focused deterministic LMS/RLS adaptive filter coverage for pure NumPy fallback behavior, optional Rust-kernel dispatch, output metadata, and signal preconditions, raising `src/shared/python/signal_toolkit/adaptive_filter.py` focused coverage to 95.24% with a 95% file-level coverage policy gate. |
| 2026-06-03 | 1.1.297 | test(model-generation): add 49 focused handler tests for `rest_api_routes.ModelGenerationAPI` covering route count, health/info shape, security headers, all missing-field 400 guards for every endpoint, inertia success branches (box/sphere/cylinder/capsule) with wrong-dimension-count errors, validate/parse success and error paths, library and editor handlers; fix `library_get_model` and `library_add_model` using `ModelEntry.model_id` (non-existent attribute) to use the correct `ModelEntry.id`. |
| 2026-06-03 | 1.1.296 | fix(programmatic-pid): guard DXF-producing `PIDDocument.export_dxf` tests on optional `ezdxf` availability so lean CI environments skip only the dependency-backed export assertions while retaining construction, validation, and precondition coverage. |
| 2026-06-03 | 1.1.294 | test(safe-eval): add a 99% file-level coverage policy gate for `src/shared/python/safe_eval.py`, backed by existing focused safe evaluator tests that cover validation, namespace allowlists, stripped builtins, scalar math, and NumPy math paths at 100% line and branch coverage. |
| 2026-06-03 | 1.1.293 | test(safe-pandas): add focused validation coverage for overlong formulas, syntax errors, unsupported operators, and maximum allowed exponent boundaries, raising `src/shared/python/safe_pandas_eval.py` focused coverage to 100% and adding a 99% file-level coverage policy gate. |
| 2026-06-02 | 1.1.292 | test(notes): add focused PyQt6 coverage for the shared notes dock widget save/reload/clear, recycle/restore, floating/redock, and initialization guard paths, raising the `src/shared/python/notes` package coverage policy gate from 48% to 95% without changing production behavior. |
| 2026-06-02 | 1.1.291 | fix(sidekick): keep conversion service helper boundaries explicit under CI changed-file mypy analysis by coercing skipped-import helper and mixin conversion results back to `float` without changing runtime conversion behavior. |
| 2026-06-02 | 1.1.290 | fix(sidekick): restore custom unit conversion by adding user-defined units to the normalized lookup map, keep invalid temperature validation failures non-fatal as documented, and add focused edge coverage for `sidekick.calculators.conversion.service` singleton helpers, normalization/cache paths, validation guards, category dispatch, and compatible-unit lookup, raising focused service coverage to 99.09% and adding a 90% file-level coverage policy gate. |
| 2026-06-02 | 1.1.289 | fix(ui): route the Windows AppUserModelID platform check through a runtime helper so Linux changed-file mypy does not mark the Windows ctypes branch unreachable while preserving the same taskbar identity behavior. |
| 2026-06-02 | 1.1.288 | fix(sidekick): restore tab hover highlight (`QTabBar::tab:!selected:hover` QSS), fix the active-tab settings button and Configure-Tabs list by preserving `TabCollection` live aliases, add tested `set_app_user_model_id`/`apply_window_icon` helpers for Windows taskbar identity, and fix the Unified Launcher icon path to use `assets/`. |
| 2026-06-02 | 1.1.287 | fix(codemap): add focused headless coverage for the codemap watcher daemon, including watchdog import failures, supported-path filtering, moved-path handling, debounce flushes, deleted-file cleanup, shutdown resource cleanup, and CLI option forwarding; deleted events now reach the existing DB cleanup path instead of being filtered out after the file disappears. |
| 2026-06-02 | 1.1.286 | test(codemap): add focused headless coverage for the codemap indexer, including supported-file walking, `.gitignore` and fallback ignore handling, unchanged-file hash skips, incremental reprocessing and deletion, unreadable/parser-skipped files, per-file error collection, manifest writing, git helper parsing/fallbacks, and preferred blake3 hashing, raising `src/shared/python/codemap/indexer.py` focused coverage from 16.24% to 98.98% without changing production behavior. |
| 2026-06-02 | 1.1.285 | fix(codemap): add focused public API coverage for repo-root discovery, query sanitization, FTS search filtering, symbol lookup, caller lookup, import parsing, neighbor traversal, repo summaries, malformed JSON fallbacks, and default-root caching; fix one-hop `neighbors()` so outbound callees are resolved and returned as documented, raising `src/shared/python/codemap/api.py` focused coverage to 96.93%. |
| 2026-06-02 | 1.1.284 | fix(ai): keep OpenAI and Anthropic system-prompt assembly mypy-clean under the changed-file CI profile by casting the shared prompt builder result back to the documented `str` contract when imported through the skipped-follow-imports namespace, without changing runtime prompt behavior. |
| 2026-06-02 | 1.1.283 | fix(ai-ui): keep the merged #3205 AI/UI hardening mypy-clean under the normal pre-push hook by removing stale system-prompt `no-any-return` ignores, routing BitNet generic errors through the shared classifier, and typing optional headless PyQt UI exports through private nullable export variables without changing runtime behavior. |
| 2026-06-02 | 1.1.282 | fix(codemap): add focused SQLite schema coverage for canonical index paths, DB initialization, local `.codemap/.gitignore` handling, schema-version fallbacks, idempotent initialization, and FTS insert/update/delete synchronization; fix the external-content FTS column contract by replacing the legacy `co` alias with `calls_out` and migrating existing v1 FTS tables, raising `src/shared/python/codemap/db.py` focused coverage from 31.82% to 100.00%. |
| 2026-06-02 | 1.1.281 | feat(a11y): improve the Unit Converter web app's theme-toggle and custom-unit validation accessibility. The theme button now keeps `aria-pressed` synchronized with the active dark/light state, and custom unit validation messages are announced via dynamic `aria-describedby` while preserving existing input hints. |
| 2026-06-02 | 1.1.280 | fix(tools): consolidated A–O review fixes resolving issues #3173/#3174/#3175/#3176/#3179/#3183/#3184/#3185/#3186/#3187/#3188 — AI adapter/tool-bridge/CLI-tools hardening, model_generation FastAPI/URDF roundtrip fixes, sidekick syngas_compression calculator de-duplication, theme color fallback drift guard, UI headless import safety, plus chat routing lifecycle, programmatic PID pipeline, and humanoid builder assembly coverage. |
| 2026-06-02 | 1.1.279 | test(codemap): add focused headless coverage for the codemap parser dispatcher, including case-insensitive extension mapping, unsupported-path handling, all registered language dispatch routes, missing-extractor fallback, and public re-export registry stability, raising `src/shared/python/codemap/parsers.py` focused coverage from 58.06% to 100.00% without changing production behavior. |
| 2026-06-02 | 1.1.278 | test(codemap): add focused headless coverage for shared tree-sitter parser helpers, including byte/text extraction helpers, child lookup, line range conversion, unsupported-language handling, successful parser construction/cache reuse, missing optional-language caching, and initialization-failure warning behavior, raising `src/shared/python/codemap/_ts_common.py` focused coverage from 66.18% to 100.00% without changing production behavior. |
| 2026-06-02 | 1.1.277 | test(codemap): add focused headless coverage for the Rust tree-sitter extractor, including parser-independent `use` imports, top-level functions, structs, typed and untyped impl blocks, nested modules, nested impl methods, unavailable-parser fallback, and incomplete-item guards, raising `src/shared/python/codemap/_lang_rust.py` focused coverage from 8.43% to 98.80% without changing production behavior. |
| 2026-06-02 | 1.1.276 | test(codemap): add focused headless coverage for the JavaScript and TypeScript tree-sitter extractors, including parser-independent import extraction, functions, exported/ambient declarations, class and abstract-class methods, variable-assigned function forms, TS/TSX language dispatch, unavailable-parser fallback, and incomplete-node guards, raising `src/shared/python/codemap/_lang_js.py` focused coverage from 7.08% to 96.46% without changing production behavior. |
| 2026-06-02 | 1.1.275 | test(codemap): make the focused Python parser coverage test independent of the optional `tree_sitter_python` wheel by driving extraction through a parser-shaped fake tree, preserving the existing `src/shared/python/codemap/_lang_python.py` 97.95% focused coverage target while keeping Python 3.10 CI deterministic. |
| 2026-06-02 | 1.1.274 | test(codemap): add focused headless coverage for the Python tree-sitter extractor, including real import/symbol/docstring/signature/call extraction, unavailable-parser fallback, missing-name guards, parser-shaped fake definition nodes, call fallback handling, import edge cases, and block recursion, raising `src/shared/python/codemap/_lang_python.py` focused coverage from 7.53% to 97.95% without changing production behavior. |
| 2026-06-02 | 1.1.273 | test(codemap): add focused headless coverage for the Markdown tree-sitter extractor, including parser-independent ATX heading extraction from byte input, long heading truncation, unavailable-parser fallback, raw heading fallback text, and blank heading skipping, raising `src/shared/python/codemap/_lang_markdown.py` focused coverage from 0.00% to 91.43% without changing production behavior. |
| 2026-06-02 | 1.1.272 | test(plot-engine): add focused headless coverage for the Matplotlib renderer, including line/scatter styling, trendline success and failure paths, 3D surface rendering, contour and heatmap options, histogram styling, filter-comparison difference plots, PNG export, validation guards, and helper defaults, raising `src/shared/python/plot_engine/matplotlib_renderer.py` focused coverage from 8.38% to 100.00% without changing production behavior. |
| 2026-06-02 | 1.1.271 | test(plot-engine): add focused headless coverage for the Plotly converter JSON contract, including typed dispatch for line/scatter, surface, contour, heatmap, histogram, and filter-comparison specs, style/layout serialization, trendline naming and failure handling, required-input guards, and helper defaults, raising `src/shared/python/plot_engine/plotly_converter.py` focused coverage from 0% to 94.77% without changing production behavior. |
| 2026-06-02 | 1.1.270 | fix(calc_backend,signal_toolkit): iterate the scrubber router's column area -> liquid flux -> flooding velocity -> diameter solve to convergence so `liquid_mass_flux` is self-consistent with the solved cross-section instead of an assumed 1 m2 basis (#3181); and restore Design-by-Contract `ValueError` guards on `Integrator.integrate`/`compute_integral` that reject NaN, inverted (`lower > upper`), and out-of-range integration bounds via explicit checks that survive `python -O` (#3182). Regression tests live in dedicated, fully type-annotated files (`calc_backend/tests/test_scrubber_convergence_3181.py`, `signal_toolkit/tests/test_bound_validation_3182.py`) to keep the delta-CI mypy surface clean. |
| 2026-06-02 | 1.1.269 | fix(scripting): add an AST escape pre-screen (`_screen_source_for_escapes`) to the `ConsoleEnvironment` sandbox so user source is rejected before compile/exec when it accesses dunder attributes (`__class__`/`__bases__`/`__subclasses__`/`__globals__` traversal) or constructs dunder names at runtime via `getattr`/`setattr`/`delattr`/`vars`/`type`/`globals`/`locals` with a non-literal or dunder name argument; raises a new `SecurityError`, wires the screen into `execute()` and `refresh_user_functions()`, and documents the authoritative out-of-process trust boundary with the in-process screen as defense-in-depth (#3180). |
| 2026-06-02 | 1.1.268 | test(plot-engine): add focused PyQt6 widget coverage for constructor theme wiring, spec rendering and signal emission, refresh/theme-change rerendering, export dialog/save behavior, empty-export guards, and image byte delegation, raising `src/shared/python/plot_engine/pyqt6_widget.py` focused coverage from 0% to 96.81% without changing production behavior. |
| 2026-06-02 | 1.1.267 | test(plot-engine): add focused headless coverage for plot engine protocol contracts, including runtime structural conformance for renderers, converters, and theme color providers plus explicit protocol stub coverage, raising `src/shared/python/plot_engine/protocols.py` focused coverage to 100% without changing production behavior. |
| 2026-06-02 | 1.1.266 | test(plot-engine): add focused headless coverage for trendline computation, including linear NaN filtering, polynomial degree capping and zero equations, exponential and power fits, optimizer fallback behavior, insufficient-data validation, unknown trend types, R-squared edge cases, and helper validation paths, raising `src/shared/python/plot_engine/trendline.py` focused coverage to 100% without changing production behavior. |
| 2026-06-02 | 1.1.265 | test(plot-engine): add focused headless coverage for contour data preparation, including scatter interpolation grid shape/value behavior, NaN filtering, insufficient-point validation, correlation matrix defaults, custom labels, and dimensionality validation, raising `src/shared/python/plot_engine/contour.py` focused coverage to 100% without changing production behavior. |
| 2026-06-02 | 1.1.264 | test(notes): add focused headless coverage for the shared notes dock integration helper, covering custom/default dock areas, dock construction, parent propagation, and invalid host validation, raising `src/shared/python/notes/integration.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.263 | test(notes): add focused headless coverage for shared notes markdown card storage, including markdown metadata round trips, create/update/list ordering, recycle/restore, settings persistence, legacy text-note migration, index helpers, and validation/error paths, raising `src/shared/python/notes/card_store.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.262 | test(notes): add focused headless coverage for shared notes models and storage validation, normalization, save/load/clear, recycle/restore/purge, index ordering, and error paths, raising `src/shared/python/notes/models.py` and `src/shared/python/notes/storage.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.261 | test(theme): add focused PyQt-light ThemeManager coverage for singleton reset, inherited app-context preferences, theme queries, stylesheet fallback, registered window application, custom theme persistence/loading/deletion, and validation/error paths, raising `src/shared/python/theme/theme_manager.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.259 | test(theme): add focused PyQt zoom controller coverage for configuration validation, persisted zoom loading, font scaling, step/reset helpers, install/uninstall, keyboard shortcuts, and Ctrl+wheel handling, raising `src/shared/python/theme/zoom.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.260 | test(theme): add focused stylesheet generator coverage for complete QSS section output, minimal embedding styles, required theme color validation, and public exports, raising `src/shared/python/theme/stylesheets.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.255 | fix(folder-packer-pro): keep the headless `operations.py` messagebox fallback typed under mypy by assigning the optional Tk import through an `Any`-typed alias while preserving the unavailable-messagebox runtime guard. |
| 2026-06-01 | 1.1.254 | test(theme): add focused headless coverage for shared matplotlib style helpers, including themed figure/axes/legend styling, default color fallbacks, canvas redraw behavior, global rcParams, palette cycling, and styled figure creation without changing production behavior. |
| 2026-06-01 | 1.1.253 | test(theme): add focused headless coverage for shared icon SVG registry rendering, unknown-icon validation, argument type guards, external SVG recoloring, and missing-file handling, raising `src/shared/python/theme/icon_utils.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.252 | test(theme): add focused coverage for shared theme typography constants, CSS font-stack exports, PyQt font-family selection, explicit-family handling, italic flags, font weights, and missing-size validation, raising `src/shared/python/theme/typography.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.251 | test(theme): add focused coverage for shared theme color validation, normalization, RGBA conversion, matplotlib palette mapping, JSON loader fallback/error paths, and Qt color conversion, raising `src/shared/python/theme/colors.py` focused coverage above 99% without changing production behavior. |
| 2026-06-01 | 1.1.250 | test(theme): add focused coverage for shared theme style constants and parameterized stylesheet helpers, raising `src/shared/python/theme/style_constants.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.249 | fix(mcp): keep config-loader preset application and npx package detection typed under the CI mypy delta profile while preserving the Python 3.10 MCP compatibility and config writer coverage changes. |
| 2026-06-01 | 1.1.248 | fix(mcp): keep MCP contracts importable on Python 3.10 by using a `str`/`Enum` transport type, keep config-loader merge validation and npx package detection mypy-clean, remove the Windows shell wrapper from the npm preset probe, and add focused deterministic coverage for the pure `config_writer` MCP server JSON writer/reader. |
| 2026-06-01 | 1.1.247 | test(mcp): add focused deterministic coverage for the pure `config_writer` MCP server JSON writer/reader, including Claude Desktop serialization, duplicate and invalid server validation, malformed environment placeholder rejection, missing/malformed file handling, flat and `mcpServers` read normalization, invalid-entry filtering, and the `load` alias. |
| 2026-06-01 | 1.1.246 | fix(performance-utils): make `OptimizedFileScanner` cache entries expire by both TTL and root directory mtime so changed directories are rescanned within the 60-second cache window, and handle top-level directory enumeration errors consistently with inaccessible child directories. Added focused deterministic coverage for scanner cache invalidation, TTL reuse/expiry, worker error suppression, hashing paths, and chunked/lazy memory utilities. |
| 2026-06-01 | 1.1.245 | fix(folder-packer-pro): guard the `operations.py` messagebox import so headless Linux runners without Tk shared libraries can import the operation mixins while GUI runtime behavior stays unchanged when Tk is available. |
| 2026-06-01 | 1.1.244 | fix(folder-packer-pro): teach `inspect_package()` to read uncompressed unencrypted archives instead of mislabeling them as encrypted, and add focused headless coverage for `folder_packer_pro` file operations, pack/unpack engine behavior, archive path traversal rejection, cancellation/error handling, and operation mixin workflows. |
| 2026-06-01 | 1.1.243 | test(data-processing): add focused coverage for the shared pandas formula validator and `DataProcessor.apply_formula` integration, pinning accepted arithmetic/boolean grammar, unsafe syntax rejection, complexity/exponent guards, and rejection logging without formula text leakage. |
| 2026-06-01 | 1.1.242 | fix(model-generation): harden the headless `model_generation` CLI library commands by parsing category/source filters into library enums, using `ModelEntry.id` in list/add output, defaulting adds to `ModelCategory.OTHER`, trimming comma-separated tags, and keeping the typed CLI dispatch path mypy-clean. Added focused CLI tests covering parser wiring, library list/add behavior, invalid filters, and inertia dimension errors. Also keeps Sidekick workspace facade name listing typed under both local and CI mypy import modes. |
| 2026-05-31 | 1.1.240 | fix(sidekick): harden calculator workspace adapter typed boundaries so changed-file mypy checks keep `Path`, `bool`, and `list[str]` return contracts when helper modules are skipped during CI analysis. |
| 2026-05-31 | 1.1.241 | fix(sidekick): harden calculator workspace adapter typed boundaries so changed-file mypy checks keep `Path`, `bool`, and `list[str]` return contracts when helper modules are skipped during CI analysis. |
| 2026-05-31 | 1.1.239 | test(sidekick): harden the Sidekick per-file coverage gate so only `src/shared/python/sidekick/` production modules are enforced, excluding changed test files from missing-coverage failures. CI now runs the full Sidekick unit suite when Sidekick source changes, and the split runtime/default-tab modules have focused contract coverage for chat bridges, plot requests, fallback diagnostics, tab definitions, and optional-tab placeholders. |
| 2026-05-31 | 1.1.238 | fix(security, #3143 #3144): rewrite wave_solver.py to use argv lists with shell=False (no shell string from issue title/body), make --dangerously-skip-permissions opt-in, and gate destructive git/gh actions (git reset --hard, issue close, gh pr merge --auto) behind an explicit --allow-mutations flag with a dry-run default; replace P1AM backend wildcard CORS (`["*"]` + credentials) with an env-driven allowlist (cors_config.resolve_cors_settings) that defaults to local dev origins, never pairs `*` with credentials, and fails closed in production without an explicit allowlist. |
| 2026-05-31 | 1.1.238 | fix(sidekick): Completed the #3141 monolith-decomposition follow-up by splitting runtime tab, default-tab, calculator workspace, runtime settings, and chat settings responsibilities into focused modules while preserving the historical import surface through facade modules. Added focused alias-contract and coverage-gate regression tests so hosts keep stable live tab collections and changed Sidekick files cannot silently bypass coverage enforcement. |
| 2026-05-31 | 1.1.237 | fix(sidekick): #3138 TabCollection.set_definitions()/sync_order_from_widget() now mutate their backing dict/list in place instead of reassigning, so UnifiedToolsSidebar's live \_tab_definitions/\_tab_ids/\_tab_widgets aliases stay current (fixes duplicate/pop-out/redock/settings flows); PythonReplWidget.execute() now waits on its worker thread and delivers output deterministically without a spinning event loop (fixes REPL output). #3139 check_sidekick_coverage.py fails when a changed Sidekick file is missing from coverage XML or when an enforced run counts zero files, closing the vacuous-pass gap. #3140 removed two stale TDD-pending xfail markers now that the package-rename import-boundary contracts pass. Part of #3141 (monolith decomposition deferred to a focused follow-up). check_sidekick_coverage.py now parses coverage.xml via defusedxml.ElementTree (matching check_coverage_policy.py) to satisfy bandit B314. |
| 2026-05-31 | 1.1.236 | perf(golf): optimize calculateTempoQuality in phaseDetector.ts by replacing the two chained .filter().reduce() passes with a single-pass for loop, eliminating intermediate array allocations while preserving the tempo score. |
| 2026-05-31 | 1.1.235 | feat(a11y, p1am frontend): add `aria-pressed` to custom toggle buttons in ControlDashboard (PID loop selector) and RoutingMatrix (input/output route cells) so screen readers announce active state. |
| 2026-05-30 | 1.1.233 | perf(golf): optimize array iterations in swingAnalyzer by replacing chained .filter().reduce() with single-pass for loops in calculateTempoMetrics and calculateSwingScores; ci: remove the retired fix-brick.yml toolcache-repair workflow (consolidates #3124 and #3129). |
| 2026-05-30 | 1.1.232 | feat(ux, #3115): improve accessibility of the ODE Solver UI by explicitly linking labels to inputs and textareas using htmlFor and unique IDs, add spellcheck="false" and disabled autocorrect. |
| 2026-05-30 | 1.1.231 | perf(p1am frontend, #3126): optimize array aggregations in AlarmsHeader.tsx by replacing chained .filter() and .reduce() operations with a single-pass loop. |
| 2026-05-30 | 1.1.230 | Fix CI failures on PR #3123: re-export \_QS_ORG/\_QS_APP/\_QS_VISIBLE_TABS_KEY from sidebar, fix apply_state \_dock_widget AttributeError (now uses \_dock_chrome.dock_widget), add waitUntil to MockQtBot, fix F6 isVisible→isHidden for headless tests, fix F10 duplicate-pin test to use subdirectory, add runtime_tabs.py and registry.py to monolith baseline, bump SPEC version. |
| 2026-05-30 | 1.1.229 | chore: remove stale type-ignore suppression comments in data_explorer_service, project_file_explorer, runtime_tabs; add explicit bool() cast on eventFilter return in os_terminal to satisfy mypy no-any-return. |
| 2026-05-30 | 1.1.228 | F4: Patched TabCollection.replace() to correctly update internal id mapping when swapping widgets; fixes stale id→widget reference after atomic swap. |
| 2026-05-30 | 1.1.227 | F4: Decomposed UnifiedToolsSidebar god class. Extracted TabCollection (id↔widget↔order bookkeeping), DockChromeController (collapse/minimize/dock-area/title-bar/shortcuts), and VisibilityPersistence (project-root-scoped QSettings read/write). Sidebar is now a thin coordinator that delegates to these three collaborators. Backward-compatible shims (\_tab_ids/\_tab_widgets/\_tab_definitions) preserved for mixins. Added test_sidekick_f4_collaborators.py with tests for all three. |
| 2026-05-30 | 1.1.226 | F6: PythonReplWidget now executes user scripts on a background QThread (\_ReplWorker) so the GUI stays responsive. Added \_cancel_button (best-effort terminate), \_status_label ('Running...'), \_set_running() toggle helper, and \_on_execution_finished() slot that syncs the namespace back to the registry on completion. |
| 2026-05-30 | 1.1.225 | F2: Added Ctrl+C interrupt button (writes 0x03 to PTY), Stop/restart button, command history ring (Up/Down navigate, newest-first, deduplicates), and eventFilter on input QLineEdit in SidekickOsTerminalWidget. |
| 2026-05-30 | 1.1.224 | F8: Added replace_tab_widget() to UnifiedToolsSidebar for atomic chat-dock retry swaps that keep both QTabWidget and \_tab_widgets bookkeeping in sync. F9: Rewrote registry.update_from() to merge via public set()/\_set_repr_entry() so name validation runs and subscribers are notified; same fix applied to load_json(). |
| 2026-05-30 | 1.1.223 | F10: Quick-access folder pins in ProjectFileExplorer now persist to and restore from QSettings (project-root-scoped key); duplicates are rejected. F11: Hoisted a shared `resolve_columns` helper in `data_explorer_service` to eliminate the duplicated column-validation logic in `data_processor_tab`. |
| 2026-05-30 | 1.1.222 | F1: Fixed Windows PTY double-submit by writing b"\n" instead of os.linesep. F3: Fixed PTY output chunk-stripping by using raw QTextEdit.append. F5: Consolidated QSettings writes into \_persist_visible_tabs with explicit org/app names. F7: Implemented singleton help dialog to prevent duplicate windows. |
| 2026-05-29 | 1.1.215 | Hardened the Sidekick C3D reader to validate the header magic byte before invoking ezc3d, so mislabeled or truncated files raise a typed `ValueError` instead of surfacing parser internals; added focused regression coverage for invalid headers and updated C3D reader tests to use temp files with valid magic bytes. |
| 2026-05-27 | 1.1.214 | Fixed HistorySidebar initialization, updated theme manager colors, and synchronized Tools baseline hashes. |
| 2026-05-27 | 1.1.210 | Added P1AM analog I/O calibration helper script and interactive Modbus CLI procedure documentation. |
| 2026-05-27 | 1.1.209 | Simplified HistorySidebar implementation to reduce lines of code under 500 lines to satisfy the file size budget check constraint. |
| 2026-05-27 | 1.1.201 | Added Sidekick Chat controls to create new chat or load conversation history, integrated HistorySidebar in horizontal QSplitter, added toolbar/status buttons, WebSocket session_created handler, and comprehensive tests. |
| 2026-05-23 | 1.1.200 | Added `sidekick.bootstrap` import to the deprecated `upstream_drift_tools` compatibility shim to preserve legacy import paths. |
| 2026-05-26 | 1.1.200 | Kept the optional session-scoped PyQt `qapp` pytest fixture in `tests/conftest.py` ruff-compliant by normalizing the guarded local import block, so PR-local test harness changes stop tripping the CI quality gate on import-order formatting alone. |
| 2026-05-22 | 1.1.199 | Fixed mypy TYPE_CHECKING import guards in sidekick process calculators (syngas_compression_calculator, acid_gas_dewpoint_calculator, pressure_drop_interface, syngas_compression_engine) and calculator_state_mixin to use `if TYPE_CHECKING:` conditional imports for optional PyQt6/matplotlib dependencies, eliminating incompatible-assignment and no-redef errors across Qt-installed and Qt-absent environments. |
| 2026-05-22 | 1.1.198 | Tightened local hook behavior for consolidated task branches so pre-push fleet guardrails inspect the unpushed commit range before falling back to the full repository, and changed the Bandit pre-push hook to scan the Python files selected by pre-commit instead of re-scanning existing repository-wide baseline debt. |
| 2026-05-21 | 1.1.195 | Resolved shared AI/chat unit-test failures by tightening Rust adapter optional-backend behavior, removing obsolete phase-one integration coverage, and updating Ollama, Rust adapter, and AI memory manager tests to use deterministic mocks for terminal-provider and event-loop contracts. |
| 2026-05-20 | 1.1.192 | Fixed shared Sidekick chat dock shutdown so an intentional widget close suppresses the WebSocket reconnect timer while unexpected disconnects retain the existing retry path; added focused regression coverage for both lifecycle branches. |
| 2026-05-20 | 1.1.191 | Hardened Sidekick test-health coverage so the Jupyter tab availability positive path simulates an importable optional `nbformat` module without requiring the package in the base environment, while the missing-dependency negative path remains covered. Marked the Sidekick dock close-affordance Qt tests as serial/offscreen and skipped them inside Windows xdist workers so the serial lane keeps coverage without crashing parallel workers. |
| 2026-05-20 | 1.1.190 | Added shared Sidekick/chat launcher integration contracts: `ChatServiceBase.condense_to_memory()` now persists explicit memory candidates through the shared memory manager, `UnifiedToolsSidebar.open_tab()` focuses visible and hidden tabs with `os_terminal` compatibility, ChatDockWidget exposes readiness diagnostics, and Qt chat imports gained subprocess-backed PyQt6 runtime diagnostics with focused regression coverage. |
| 2026-05-18 | 1.1.185 | Added `htmlFor` and `id` mapping to range inputs in `SwingComparison.tsx` (`src/media_processing/video_processor/apps/web`) to improve screen reader accessibility. |
| 2026-05-18 | 1.1.184 | Optimized Nelder-Mead optimization loop in pendulum simulator by replacing map and slice with pre-allocated arrays and standard for loops to minimize GC pauses. |
| 2026-05-17 | 1.1.183 | Pre-allocated the `results` array in the `solveODESystem` hot RK4 integration loop (`src/ode_solver/web/src/lib/odeSolver.ts`) to eliminate continuous memory reallocation overhead and garbage collection pauses during large numerical simulations. |
| 2026-05-15 | 1.1.181 | Split AI settings local-provider configuration widgets so Ollama keeps its host/model discovery controls, Cline shows its own endpoint test UI, BitNet shows an installation-root hint tied to the main model selector, and CLI-backed providers no longer render misleading Ollama-specific fields; added focused PyQt6 regression coverage for the provider-specific widget contracts. |
| 2026-05-15 | 1.1.179 | Added a markdown-backed shared notes card store with stable path-safe IDs, metadata round trips, validated note and board colors, reversible markdown-card recycling/restoration, legacy `project.notes.txt` migration, import-safe backend coverage, and a lightweight Sidekick Notes color-control contract that reuses the shared store. |
| 2026-05-15 | 1.1.178 | Added an optional Sidekick Function Generator tab with import-safe PyQt6 launcher integration, shared default-tab/help metadata, design-token aliases, and focused sidebar regression coverage. |
| 2026-05-15 | 1.1.176 | Added Sidekick calculator workspace management with isolated calculator-local variables, explicit local-to-global promotion, scoped local/global JSON workspace persistence helpers, focused regression coverage for merge, replace, malformed-file rollback, and duplicate-facade separation behavior, stabilized Sidekick data explorer dtype summaries across pandas string dtype changes, and kept calculator-tab expression evaluation inside the shared safe math evaluator so headless imports do not require Flask or tool-specific calculator packages. |
| 2026-05-14 | 1.1.175 | Added a lazy optional Sidekick Data Processor tab that stays hidden by default, reports missing UI/runtime dependencies without crashing Sidekick, and exports validated selected Data Processor results into the shared workspace registry with focused import/runtime regression coverage. |
| 2026-05-14 | 1.1.174 | Added a Sidekick Data Explorer tab with project-scoped file validation, bounded CSV/TSV/JSON/Parquet/Excel preview service limits, schema/null-count sample summaries, preview-to-workspace export, and a structured Data Processor handoff request contract plus focused backend/UI regression coverage. |
| 2026-05-14 | 1.1.173 | Added a bounded Sidekick workspace command line to the calculator tab for explicit local/global variable assignment, inspection, deletion, clear, and load/save operations, reusing the shared command-history and workspace persistence contracts while keeping workspace mutations separate from arbitrary terminal execution. |
| 2026-05-14 | 1.1.172 | Added a pure-Python Sidekick help registry for default tabs and shared context-menu actions, wired default-tab help metadata into the shared sidebar, exposed a Help action in the tab context menu, added hover hints to compact terminal/notes controls, documented custom-tab help requirements in the sidebar README, and expanded the shared UI regression suite to enforce the new help contract. |
| 2026-05-14 | 1.1.171 | Added Sidekick named state profile storage helpers with path-safe save/load contracts, atomic malformed-profile rejection, explicit clear-data warning confirmation, sidebar wrapper methods, README guidance, and focused regression coverage. |
| 2026-05-14 | 1.1.170 | Added validated Sidekick calculator startup import preferences with default optional NumPy/SciPy aliases, JSON sidebar-state persistence, transaction-safe import execution, missing-dependency diagnostics in the calculator tab, and focused backend/UI regression coverage. |
| 2026-05-14 | 1.1.169 | Added calculator-local Sidekick workspace save/load wiring with an explicit scoped persistence controller, JSON path validation, atomic save, merge-versus-confirmed-replace load behavior, malformed-file rollback, and UI button coverage that keeps calculator workspace persistence separate from the global sidebar workspace registry. |
| 2026-05-14 | 1.1.168 | Added a Sidekick file explorer navigation controller with normalized current path state, back/forward/up history, injectable common-location discovery, project-boundary containment, and predictable disabled-state flags, then wired the project explorer widget to expose a compact navigation bar and common-locations sidebar. |
| 2026-05-14 | 1.1.165 | Optimized the ODE solver RK4 integration loop by moving state and derivative buffers from keyed objects to indexed arrays, extracted the solver and presets into a pure module, and added Vitest coverage for analytical decay, coupled oscillator order, and solver preconditions. |
| 2026-05-14 | 1.1.164 | Improved calculator bounds/value input accessibility by labeling the grouped lower-bound, upper-bound, and evaluation-point controls with a shared group name plus explicit accessible names for each field. |
| 2026-05-14 | 1.1.163 | Optimized the pressure-drop calculator gas-composition hot paths by replacing repeated object-entry/value reductions with single-pass keyed loops for mixture molecular weight, total composition, and normalized composition construction. |
| 2026-05-14 | 1.1.162 | Refactored Sidekick default tab construction into a focused helper module so `UnifiedToolsSidebar` stays below the changed-file LOC budget while preserving the runtime tab behavior introduced in 1.1.161. |
| 2026-05-14 | 1.1.161 | Replaced remaining Sidekick runtime placeholders with embedded utility widgets: chat status/optional PyQt chat dock loading, a workspace-aware Python terminal with optional numpy/pandas/scipy aliases, a TI-89 symbolic calculator tab that publishes results into workspace state, and project-persistent notes with explicit save and debounced autosave. Added widget contract coverage for the runtime tabs. |
| 2026-05-14 | 1.1.160 | Added runtime Sidekick theme reapplication APIs so existing PyQt sidebar instances can switch shared themes or explicit design-token sets without being reconstructed. |
| 2026-05-14 | 1.1.159 | Added shared-theme-name resolution to the Sidekick host factory/install helpers so PyQt hosts can opt into canonical theme definitions without hand-building design tokens. |
| 2026-05-14 | 1.1.156 | Added shared PyQt6 responsive sizing and application zoom utilities for issue #2647. The theme package now exposes text-aware minimum width helpers, readable form-layout configuration, scroll-area wrapping, a persisted application zoom event filter for Ctrl+wheel/Ctrl+plus/Ctrl+minus/Ctrl+0, and scaled UI tokens for downstream QSS/layout regeneration; package discovery now includes the `shared*` namespace so these fleet imports ship with `ud-tools`. |
| 2026-05-14 | 1.1.155 | Added the canonical Sidekick design-token bridge with pure-Python token exports, CSS-variable and QSS mapping helpers, stable Qt object names/selectors, default shared sidebar styling, and focused tests for token contract and backend import safety. |
| 2026-05-13 | 1.1.154 | Expanded the shared sidebar into the Sidekick toolkit with configurable tab definitions, persisted left/right dock placement, minimized state, tab ordering, hidden tabs, popped-out tab tracking, redock and duplicate-tab APIs, and tests for flexible host workflows while preserving the existing `install_tools_sidebar` contract. |
| 2026-05-13 | 1.1.153 | Added the shared `upstream_drift_tools.ui.tools_sidebar` package with a Qt-binding-compatible dockable sidebar, project file explorer, workspace registry/state persistence, public `create_tools_sidebar` and `install_tools_sidebar` APIs, and focused backend/import/widget contract tests for downstream host integration. |
| 2026-05-13 | 1.1.152 | Improved chat layout by moving the shared dock Close button into the persistent status header, replacing clipped history-list text with wrapped row widgets, and adding transparent icon-only archive, restore, and delete actions directly on chat-history rows. |
| 2026-05-13 | 1.1.151 | Hardened shared chat dock terminal lifecycle controls so Start is disabled while a terminal session is pending or active, Stop is enabled only for active sessions, and shell/provider selectors are locked while the selected terminal agent session is running. |
| 2026-05-13 | 1.1.150 | Improved the shared chat dock terminal interface by populating shell/provider selectors from the terminal provider registry, adding an explicit terminal Stop action wired to the existing WebSocket stop protocol, and adding an in-dock Close button so embedded chat windows can be dismissed from inside the chat UI. |
| 2026-05-13 | 1.1.149 | Added shared AI chat memory management with a Tools-scoped `user_memory.json` store, explicit archived-conversation preference extraction, project-root `AGENTS.md` prompt inclusion, bounded prompt-memory formatting across provider adapters, and focused regression coverage so archived chats inform future sessions without becoming opaque model training data. |
| 2026-05-13 | 1.1.148 | Added data-driven shared chat terminal-provider descriptors for Claude Code, Codex, Cline CLI, and Gemini CLI, plus default registry builders, install/auth probe command metadata, and command redaction helpers so downstream UIs can enumerate terminal agents without copying provider lists or logging secret-like command values. |
| 2026-05-13 | 1.1.144 | Added a native BitNet direct subprocess adapter for shared AI chat provider resolution, exposing local 1.58b models through the adapter factory and settings metadata without requiring an external FastAPI server. |
| 2026-05-13 | 1.1.143 | Synchronized Signal Toolkit Matplotlib canvas theming for issue #2582 by applying the active fleet plot theme after axes are created, keeping legacy `setup_dark_theme()` wired to the shared theme manager, and adding regression coverage for themed axes and spines. |
| 2026-05-13 | 1.1.142 | Registered the migrated Video Analyzer PyQt6 surface in the generator-backed tools catalog and surface contract so issue #2585 is visible through both the canonical GUI manifest and generated launcher outputs. |
| 2026-05-13 | 1.1.141 | Made the migrated Video Analyzer installable and launchable from Tools for issue #2585 by adding package discovery, a `video-analyzer` console script, optional video runtime dependencies, installed-package import paths, and focused packaging/launcher regression tests. |
| 2026-05-13 | 1.1.140 | Tightened the shared chat package contract for issue #2592 by exporting the documented model/list/index facade symbols, adding a `chat` optional dependency group and compatibility matrix, fixing installed-package lazy Qt loading, validating model/index status payloads, and removing product-specific defaults from reusable AI assistant GUI metadata. |
| 2026-05-12 | 1.1.135 | Added Rust `tools-core.signal` moving-average and exponential-smoothing kernels with PyO3 numpy vector-in/vector-out endpoints, filling the remaining smoothing-filter slice after the LMS/RLS migration. |
| 2026-05-12 | 1.1.134 | Promoted LMS/RLS adaptive filters to native Rust implementations via PyO3 bindings, eliminating Python-side vectorization overhead for high-frequency signal processing pipelines (PR #2575). |
| 2026-05-11 | 1.1.132 | Fixed `signal_toolkit.calculus` import: replaced bare `from src.shared.python.contracts import require` (broken because the repo root is not on `pytest`'s pythonpath) with the sibling-module try/except pattern used in `core.py`, and cast `Differentiator.differentiate`'s return to `np.asarray(dy)` to keep mypy `no-any-return` clean. Unblocks `tests (3.x)` matrix on `main`. |
| 2026-05-11 | 1.1.131 | Added shared `codemap` package (`src/shared/python/codemap/`) — tree-sitter symbol index over SQLite FTS5 with a 6-function pydantic query API (`search_code`, `get_symbol`, `who_calls`, `imports_of`, `neighbors`, `repo_summary`), CLI (`codemap rebuild/search/who-calls/export/info`), `watchdog` daemon (`codemap-watch`), and FastMCP server (`codemap-mcp`) so external coding agents inherit the same data the in-app chat consumes. `.codemap/` is gitignored; embedding layer deferred to a follow-up. |
| 2026-05-11 | 1.1.130 | Hardened `signal_toolkit.calculus.Differentiator.differentiate` with an explicit `require(order >= 1, ...)` precondition so non-positive derivative orders raise `PreconditionError` instead of silently producing an empty derivative loop. |
| 2026-05-11 | 1.1.129 | Added dynamic focus shifting to inline form validation within the Calculator app. This prevents keyboard focus traps by focusing the first invalid input (`.focus()`) and marking it with `aria-invalid="true"`. |
| 2026-05-07 | 1.1.128 | Pre-compiled ODE Solver derivative expressions outside the RK4 loop while preserving the existing non-finite fallback behavior, so singular or overflowing user formulas still collapse to `0` instead of poisoning the integration state with `NaN` or `Infinity`. |
| 2026-05-05 | 1.1.125 | Optimized polynomial evaluation using Horners method in `pendulum-web` physics engines (`physics.ts`, `physics_triple.ts`, `physics_golfer.ts`). |
| 2026-05-04 | 1.1.124 | Documented production-readiness hardening for generated data-processing batch scripts, shared pandas formula allowlist validation, model-generation mesh upload size and filename checks with cleanup, and MakeHuman generated-script serialization plus the `mesh_generator_makehuman.py` compatibility shim. |
| 2026-04-26 | 1.1.111 | Improved accessibility for the calculator clear button's soft confirm state. Added `aria-live="polite"` to the parent row and dynamically toggled the `aria-label` between "Clear all fields" and "Confirm clear all fields" to keep screen reader users informed of the required secondary action. |
| 2026-04-25 | 1.1.107 | Fixed StrEnum import compatibility for Python 3.10 by routing `steam_engine_calculator` and `video_processor` API modules through the existing `utils.compatibility` backport facade, eliminating import-time failures on the 3.10 CI interpreter. |
| 2026-04-25 | 1.1.106 | Added dynamic focus shifting to inline form validation within the Unit Converter app's Custom Units modal. This prevents keyboard focus traps by focusing the first invalid input (`.focus()`) and marking it with `aria-invalid="true"`. |
| 2026-04-23 | 1.1.103 | Tightened the shared `model_generation` unified-loader conversion contract so malformed MJCF/URDF XML parse failures are wrapped as `ConversionError`, converter-raised `ConversionError` instances propagate unchanged, and regression tests lock the typed error/logging behavior. |
| 2026-04-23 | 1.1.101 | Hardened model-generation REST routing so unexpected route-handler programming errors propagate to the framework adapter instead of being flattened into JSON 500 responses by the route facade, with regression coverage for the propagation contract. |
| 2026-04-23 | 1.1.100 | Extended the Python 3.10 UTC compatibility contract across document-processing, folder-packing, shared model-generation, upstream-drift UI/state, folder-tool analysis, and launcher timestamp paths by using `timezone.utc` instead of the Python 3.11-only `datetime.UTC` alias while preserving timezone-aware datetime behavior. |
| 2026-04-23 | 1.1.99 | Kept shared data-processing result timestamps timezone-aware while preserving Python 3.10 compatibility by using `timezone.utc` rather than the Python 3.11-only `datetime.UTC` alias, keeping the data-processing import contract green across the supported CI interpreter matrix. |
| 2026-04-25 | 1.1.105 | Narrowed `ConsoleEnvironment.refresh_user_functions()` to re-raise `KeyboardInterrupt` and `SystemExit` while still logging expected user-code failures from the persisted scripting library, and added focused regression coverage for both reload paths. |
| 2026-04-23 | 1.1.98 | Documented the rotation converter API exception-boundary tests that keep invalid quaternion parsing mapped to HTTP 422 while allowing unexpected reference-frame runtime failures to propagate for diagnostics instead of being silently swallowed. |
| 2026-04-23 | 1.1.97 | Security and robustness remediation pass from adversarial review: tightened exception boundaries and error propagation for shared rotation conversion, scripting runtime, and model-generation loaders; hardened data-processing and state-management paths against invalid inputs and silent failures; and aligned related test coverage for the updated failure-handling contracts. |
| 2026-04-23 | 1.1.96 | Hardened ODE and signal generation preconditions so direct RK4 calls reject fewer than two output points, chirp generation rejects single-point time arrays, and sawtooth/triangle/square generation reject non-positive frequencies with clear `ValueError` messages instead of division-by-zero failures. |
| 2026-04-22 | 1.1.92 | Fixed Design by Contract runtime toggling so contract primitives, decorators, invariant checks, and validation helpers read the canonical contract state instead of stale module-level compatibility aliases; added regression coverage for alias/state divergence. |
| 2026-04-22 | 1.1.91 | Security hardening (closes #2219): removed starred argument unpacking from the safe mathematical expression evaluator AST allowlist and added regression coverage so expressions such as `sum(*x)` are rejected before execution. |
| 2026-04-22 | 1.1.88 | Test-enforcement fix (closes #2211): restricted GH1732 logging-consistency excluded-directory matching to the top-level `src/<segment>` only, and added regression coverage proving nested path segments named like excluded dirs remain in sweep scope. |
| 2026-04-22 | 1.1.87 | Documented the `signal_toolkit` package organization for adaptive filters: `AdaptiveFilter` now lives in `adaptive_filter.py` while remaining available from the package root and legacy `filters` module. |
| 2026-04-22 | 1.1.85 | Implementation (closes #2200): added a flat Asteroid Jumper controller snapshot DTO and routed the renderer through it to remove nested state traversal from the draw path. |
| 2026-04-22 | 1.1.84 | Documentation (closes #2200): reviewed deep object traversal hotspots in launchers, Matplotlib/Qt UI code, assessment scripts, Rust ball-flight physics, and Asteroid Jumper controller code, documenting framework/path/import/value-object boundaries that do not require DTO or facade extraction. |
| 2026-04-22 | 1.1.83 | Optimized statistical calculation in data processor using Welford's algorithm to compute variance in a single pass. |
| 2026-04-19 | 1.1.82 | Removed QTimer.singleShot startup races and leaky lambda captures in shared chat dock and syngas compression calculator UI code by routing deferred initialization through named callbacks and stored helper methods (PR #2163). |
| 2026-04-19 | 1.1.81 | Aligned dependency metadata with the supported Python and toolchain baseline: Python package metadata now starts at Python 3.11, lint/type configuration shares that floor, Black was removed from the canonical format path, and the reproducible requirements lock includes the pytest timeout and benchmark plugins declared by the development manifests (PR #2161). |
| 2026-04-19 | 1.1.80 | Hardened model-generation archive extraction and URDF mesh resolution by normalizing archive member paths, rejecting traversal or absolute members before extraction, and preserving unsafe mesh references as text instead of resolving them to local files (PR #2157). |
| 2026-04-19 | 1.1.79 | Consolidated stale Tools PR fixes covering shared rotation primitives, data processor background worker error surfacing and UI offload, PDF renamer API-key/CORS hardening, narrower exception fallbacks, shared GUI boundary checks, and lower-body manifest registration; also tightened NumPy return typing for the rotation modern robotics helpers checked by quality-gate (PR #2149). |
| 2026-04-19 | 1.1.78 | Optimized `TimeRangePanel.tsx` in `data-processor-web` by computing time-column ranges in a single pass and avoiding `Math.min`/`Math.max` spread calls that can overflow the call stack on large datasets (PR #2156). |
| 2026-04-19 | 1.1.77 | Hardened model-generation library GitHub discovery and downloads by validating generated GitHub API URLs, rejecting non-HTTPS model source URLs, and skipping untrusted subdirectory URLs before network retrieval (PR #2146). |
| 2026-04-21 | 1.1.76 | Added screen-reader-only context to dynamic video progress text and pose detection counters so numeric readouts expose their meaning to assistive technology; decorative pulsing dots are now hidden from screen readers (PR #2138). |
| 2026-04-21 | 1.1.75 | Optimized `calculateStatistics` in `useDataProcessor.ts` by extracting numbers into a dynamically resizing `Float64Array` during the first pass to eliminate a second pass over the original array of objects (PR #2137). |
| 2026-04-21 | 1.1.74 | Disabled pickle-backed reads, writes, and file-dialog discovery in shared data-processing helpers and upstream drift tooling to prevent arbitrary code execution through unsafe deserialization (PR #2139). |
| 2026-04-21 | 1.1.73 | Improved exception handling and signal re-raising in rotation converter UI threads, scripting environment, and model library imports by capturing background thread exceptions, adding structured logging, and re-raising with context (PR #2088). |
| 2026-04-21 | 1.1.72 | Enhanced data processor exception handling by wrapping background threading tasks with try-except blocks that log exceptions and propagate errors to the main thread instead of silently failing (PR #2084). |
| 2026-04-21 | 1.1.71 | Hardened data-processing file I/O by disabling pickle reads and writes by default, removing pickle extensions from GUI-supported file discovery paths, and requiring an explicit trusted-legacy override for pickle use. |
| 2026-04-21 | 1.1.70 | Test configuration hygiene: registered the complete CLAUDE.md marker set in `pytest.ini`, enabled strict xfail handling, and added a contract-test backbone for the ODE solver, pressure-drop calculator, and rotation-converter calc backend request/response models. |
| 2026-04-21 | 1.1.69 | Stopped the bot CI trigger workflow from using stale external credentials for repository checkout and PR/check API operations so bot-authored PRs use repo-scoped workflow credentials for required check discovery. |
| 2026-04-21 | 1.1.68 | Restricted Data Processor web row-copy paths to own enumerable properties via a shared `Object.keys` helper and added regression coverage to prevent inherited prototype keys from being copied into processed rows. |
| 2026-04-21 | 1.1.67 | Filter deleted test files out of the CI changed-test list so PRs that intentionally remove stale tests do not pass non-existent paths to pytest. |
| 2026-04-21 | 1.1.66 | Hardened asteroid-jumper physics validation so non-finite timesteps and physics parameters are rejected with explicit `ValueError`s instead of propagating NaN or infinity through simulation state. |
| 2026-04-21 | 1.1.66 | Simplified root pytest addopts in `pyproject.toml` by removing benchmark and xdist-specific defaults so repository-level test runs do not require those plugins outside focused plugin test contexts. |
| 2026-04-17 | 1.1.64 | Optimized `applyFilter` loop in `useDataProcessor.ts` by replacing the object spread operator with manual property copying to eliminate significant garbage collection overhead during large dataset processing. |
| 2026-04-17 | 1.1.63 | Hardened model-generation GitHub repository downloads by requiring HTTPS retrievals and validating mesh output paths so API-provided mesh names cannot escape the destination directory; kept the unit-converter development WSGI debugger disabled unless `FLASK_DEBUG=1` is explicitly set. |
| 2026-04-17 | 1.1.62 | Enhanced video editor UX by replacing native alert dialogs with inline accessible errors and ensuring proper focus styles. |
| 2026-04-17 | 1.1.61 | Replaced runtime `assert` validation in asteroid-jumper physics, rotation-converter UI helpers, and scripting console execution with explicit exceptions so invalid caller input remains guarded under optimized Python. |
| 2026-04-16 | 1.1.60 | Hardened launcher process handling by validating tool names, cleaning up spawned process groups, surfacing explicit model-conversion errors, and regression-testing temporary-file cleanup paths. |
| 2026-04-16 | 1.1.59 | Removed stale root-level debug artifacts (`.ci_trigger.py`, `MUJOCO_LOG.TXT`, `error_log.txt`, `wave_log.txt`, and the empty marker file ending in `Last`), added root-scoped ignore rules for those paths, and locked the hygiene policy with regression tests. |
| 2026-04-16 | 1.1.58 | Hardened GitHub archive extraction in the model-generation repository helper by validating zip members before unpacking so repository downloads cannot escape the destination directory. |
| 2026-04-16 | 1.1.55 | Replaced object spread operator with manual property copy in `integrateSignals` and `differentiateSignals` loops in `useDataProcessor.ts`; wrapped UI components (`AdvancedPanel`, `ExportPanel`, `FilterPanel`, `ResamplePanel`) in `React.memo()` to prevent unnecessary re-renders. |
| 2026-04-15 | 1.1.56 | Refreshed the data processor regression-preparation optimization spec after CI retriggers so the PR-level SPEC freshness gate sees a documentation update on the latest source-changing branch head. |
| 2026-04-16 | 1.1.57 | Improved the accessibility and semantics of the `AudioRecorder` component in the Video Processor app. Added `aria-label`s to recording control buttons, formatted recording duration for screen readers, hid purely visual elements from screen readers, and enhanced keyboard navigation by adding `focus-visible` styling to all buttons. |
| 2026-04-15 | 1.1.55 | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead. |
| 2026-04-16 | 1.1.53 | Added `aria-label` and `title` to the dynamically generated "Remove" button (`×`) in the unit converter Custom Units list for screen reader accessibility. |
| 2026-04-13 | 1.1.52 | Added visually hidden `sr-only` span before the raw timer text in `AudioRecorder.tsx` to provide screen reader context and added `aria-hidden` to purely decorative pulsing red dot. |
| 2026-04-13 | 1.1.51 | Added `tools.shared.python.model_generation.editor` compatibility namespace so downstream repos can import the text editor via `tools.shared.python` without duplicating the module; added `-p no:xvfb` to pytest addopts so the test suite runs on headless self-hosted runners that lack Xvfb; applied ruff formatting fixes across GUI stylesheets and multiline string literals. |
| 2026-04-12 | 1.1.51 | Replace remaining `print()` calls with `logging` across `src/` modules and disable xvfb pytest plugin to fix CI timeout on headless runners. |
| 2026-04-13 | 1.1.48 | Wrapped the `SignalList` and `StatisticsPanel` components in `React.memo()` to prevent expensive re-render cascades in the data processor web application during UI tab navigation. |
| 2026-04-12 | 1.1.47 | Added the shared `tools.mypy_autofix_agent` module and `mypy-autofix` console entry point so downstream fleet repositories can call one maintained mypy autofix implementation instead of carrying duplicated script copies; kept `tools.setup_logging` lazy so CLI startup does not import optional heavy dependencies. |
| 2026-04-11 | 1.1.46 | Lower-body builder DRY refactor: extracted `_build_leg_xml(side, ...)` and `_build_leg_actuators_xml(side)` helpers so both legs and both actuator blocks share a single source of truth. `build_lower_body_xml` now calls each helper once per side instead of duplicating ~45 lines of MJCF. New regression tests assert left/right symmetry of joint/body/actuator/geom/site sets and pin the expected counts. |
| 2026-04-11 | 1.1.45 | Closed-chain ankle IK in `LowerBodySimulator.setup_initial_pose`: the ankle angles are solved by a closed-form 2-DOF decomposition of the calf's world rotation so each foot's world Z-axis is `(0, 0, 1)` for any feasible hip/knee pose. Raises `ValueError` identifying the offending axis when the required ankle angle exceeds the ±30° joint limit instead of silently clipping. Defaults changed from 30°/120°/20° (infeasible, silently clipped) to 20°/30°/20° (a feasible golf address posture). The PyQt panel catches infeasibility and logs a warning. |
| 2026-04-11 | 1.1.44 | Lower-body simulator DRY/LOD refactor: centralized mj_name2id lookups into a single cache populated in `_cache_indices` (joints, actuators, sites, geoms, bodies), eliminated reflective lookups from hot paths (`step`, `compute_diagnostics`, `inverse_kinematics`, `set_joint_polynomial`, `analyze_induced_acceleration`), and decomposed `compute_diagnostics` into `_collect_tracking_error`, `_collect_joint_torques`, `_collect_ground_reaction_forces`. Added contract test suite locking down the public API surface (`-m contract`). |
| 2026-04-11 | 1.1.43 | Added inclined-plane pelvis rotation driver to the lower-body simulator: `set_pelvis_inclined_rotation(target, ...)` wrenches the pelvis free joint via `data.xfrc_applied` each step so the body tracks an inclined rotation axis (spine angle) plus a smoothstep-ramped lateral weight shift during the downswing. New `InclinedPlaneHipRotationTarget.lateral_shift_m`, `lateral_shift_at(t)`, and `target_quaternion_at(t)` with full DbC. |
| 2026-04-11 | 1.1.42 | Anatomically-shaped lower-body pelvis: composite of inertial host ellipsoid plus five mass=0 visual-only landmark geoms (sacrum, bilateral iliac wings, bright-red ASIS spheres, pubic symphysis) so pelvic tilt is visually unambiguous in the viewer without any change to dynamics. |
| 2026-04-11 | 1.1.41 | Added a full reset control to the lower-body PyQt panel that stops playback, clears history, returns MuJoCo time to zero, preserves loaded golf hip rotation targets, and reapplies the target pose at `t=0`. |
| 2026-04-11 | 1.1.40 | Added `tools.shared.python.model_generation.editor` compatibility exports (including `TextEditor` alias) to support removing duplicate model editor implementations in downstream repos that consume Tools as a dependency. |
| 2026-04-11 | 1.1.39 | Extended lower-body simulator history playback diagnostics so cached frames expose the configured inclined-plane hip rotation target for scrub-based analysis and verification. |
| 2026-04-11 | 1.1.38 | Added the lower-body inclined-plane hip rotation target profile with deterministic sampling, DbC validation, both-socket simulator application, and diagnostics/history coverage for the first golf lower-body rotation slice. |
| 2026-03-28 | 1.0.0 | Initial specification |
| 2026-03-29 | 1.0.1 | Document performance improvement in DataChart downsampling algorithm |
| 2026-03-30 | 1.0.2 | A-N assessment remediation: LoD refactoring in convert_tools_icon.py, launch.py, launch_signal_toolkit.py, verify_launcher.py; DbC input validation added to launch_tool, bootstrap, migrate_file, \_print_environment_info, \_check_launcher_file, \_print_recommendations, \_on_poly_generated; docstrings added to **init** and missing functions in setup_dev.py, remove_broken_scripts.py, migrate_print_to_logging.py, launch_signal_toolkit.py. |
| 2026-03-31 | 1.0.3 | Fix CI import error in tests/shared/python/test_contracts.py and optimize React rendering in ToolsPanel. |
| 2026-04-01 | 1.0.4 | Add keyboard accessibility (focus-within) to video player controls in web application. |
| 2026-04-01 | 1.0.5 | Optimize the data processor median filter to reuse a `Float64Array` buffer and preallocate result storage, reducing per-window allocations during large CSV filtering workflows. |
| 2026-04-02 | 1.0.6 | Refactored AnalyticsSuite (computeCorrelation, computeRegression, pearsonCorrelation) to use iterative primitive arrays and eliminate chained .map/.filter mapping overhead, vastly reducing garbage collection pressure. |
| 2026-04-02 | 1.0.7 | Run comprehensive assessments and apply auto-fixes across the repository. |
| 2026-04-03 | 1.0.8 | Refactor `linearRegression` and `polynomialRegression` in `useDataProcessor.ts` to replace multiple consecutive `.reduce()` and `.map()` array iteration methods with single-pass `for` loops, improving performance for large datasets. |
| 2026-04-10 | 1.0.9 | Optimize Math Functions using single-pass loops. |
| 2026-04-10 | 1.1.0 | Add keyboard accessibility and focus management to the Data Processor web application file upload dropzone. |
| 2026-05-18 | 1.1.1 | Fix command injection vulnerability in MATLAB Quality Utils by escaping single quotes in paths passed to MATLAB and Octave shells. |
| 2026-05-18 | 1.1.2 | Optimize PCA mathematical matrix calculations in AnalyticsSuite to use column-wise typed Float64Array to prevent large O(N) allocation overhead. |
| 2026-05-18 | 1.1.3 | Optimize linear regression calculation in AnalyticsSuite using single-pass loops instead of map/reduce to minimize garbage collection pauses. |
| 2026-05-19 | 1.1.4 | Add inline error message handling to SignalList to avoid blocking native alert dialogs and added comprehensive focus-visible states across all signal list interface buttons for enhanced keyboard accessibility. |
| 2026-04-04 | 1.1.5 | Replace print statements with logger calls in lower_body_model main entry point to comply with no-print policy and improve production logging. |
| 2026-04-05 | 1.1.6 | Optimize DataChart point extraction loop to explicitly map selected properties instead of using an object spread on the entire row in `src/data_processing/data_processor/web/src/components/DataChart.tsx`. |
| 2026-04-05 | 1.1.7 | Improve HelpPanel accessibility by adding ARIA expanded states and control links to accordion toggles, and adding explicit focus-visible rings for keyboard users. |
| 2026-04-05 | 1.1.8 | Optimize PlotView WebGL rendering to use Float64Array and bypass map array creation overhead. |
| 2026-04-05 | 1.1.9 | Bridge the embedded `src/pendulum_simulator/tests` suite into the top-level `tests/` tree so standard `pytest tests/` collection includes pendulum coverage without double-collecting the same files during root-level pytest runs. |
| 2026-04-05 | 1.1.10 | Standardize vessel drafter `require_positive` usage onto the fleet-wide `(value, name)` argument order while keeping guarded support for the legacy local order and adding regression tests for the signature normalization. |
| 2026-04-05 | 1.1.11 | Deduplicate repeated scalar surface evaluator closures in `analysis_tab.py` by routing matrix and transformed-value cases through shared helper builders, with regression coverage for the new helper paths. |
| 2026-04-05 | 1.1.12 | Expand the embedded-suite discovery policy so root-level pytest ignores bridged `src/` suites by default while `pytest tests/` includes both pendulum and solar-system embedded tests through top-level bridge directories. |
| 2026-04-05 | 1.1.13 | Move pendulum optimizer objective-refresh wiring behind a public `OptimizationWidget` API so `SimulationPanel` no longer reaches through private optimizer button and log internals before optimization runs. |
| 2026-04-06 | 1.1.14 | Remove developer-machine repository paths from maintenance scripts and eliminate the local sys.path bootstrap fallback from convert_tools_icon.py. |
| 2026-04-06 | 1.1.15 | Replace chained array map and filter operations with a single loop in the calculateTrendline algorithm to prevent memory allocation and garbage collection overhead. |
| 2026-04-06 | 1.1.16 | Add focus-within styles to video uploader dropzone and missing aria-labels to the volume and seek range inputs in the video processor web application to improve keyboard navigation visibility. |
| 2026-04-06 | 1.1.17 | Optimize Polynomial Regression Matrix Construction in AnalyticsSuite using single-pass loops. |
| 2026-04-06 | 1.1.18 | Refactored `applyFilter` inside `useDataProcessor.ts` to pre-allocate buffers and run the mapping in a single loop. |
| 2026-04-06 | 1.1.19 | Split `pressure_drop_interface.py` into facade-oriented `pressure_drop_api`, `pressure_drop_validation`, `pressure_drop_reference`, and `pressure_drop_results` modules while preserving the public interface and extending regression coverage for the pressure-drop calculator. |
| 2026-04-07 | 1.1.20 | Added explicit `focus-visible` keyboard focus indicators to the Video Processor web `ToolsPanel` buttons, color controls, slider, and destructive action buttons so keyboard navigation remains visible throughout the drawing workflow. |
| 2026-04-07 | 1.1.21 | Split `model_generation` REST routing from the Flask and FastAPI adapters behind a backward-compatible shim, decomposed the pressure-drop engine into friction-factor, flow-property, fittings, and compressible-flow modules with regression coverage for the preserved calculations, and restored the top-level `contracts` compatibility export for `_resolve_contract_level`. |
| 2026-04-07 | 1.1.22 | Formalize stdout/stderr helper usage for CLI-facing launcher and coverage-gate scripts so terminal output remains explicit while avoiding ad hoc `print()` usage in those entry points. |
| 2026-04-07 | 1.1.23 | Split the data-processor neural-network script exporter, ANOVA analyzer, and vectorized filter engine into smaller domain modules behind backward-compatible facades, and add focused regression tests for the preserved public and compatibility interfaces. |
| 2026-04-07 | 1.1.25 | Replaced raw `print()` summary emission in `scripts/generate_tools_json.py` with an explicit stdout helper, added regression coverage for the CLI entrypoint's generated-file summary contract, and aligned the humanoid mesh-generator facade with the split backend modules so refreshed type-checking stays green after the backend extraction on `main`. |
| 2026-04-07 | 1.1.26 | Extracted the double-pendulum golf equations popup string literals into `equations_data.py`, leaving the popup module focused on presentation and control wiring while preserving the existing dialog behavior. |
| 2026-04-07 | 1.1.27 | Optimized `AnalyticsSuite` regression filtering by staging selected x/y series values into `Float64Array` buffers before converting them back to plain arrays for the existing result contract, reducing repeated push-allocation overhead in large regression workloads. |
| 2026-04-07 | 1.1.28 | Optimized `AnalyticsSuite` Pearson correlation by preserving the PR's single-pass accumulation and variance-clamping path while widening the helper to accept pre-allocated `Float64Array` inputs from the newer analytics data flow. |
| 2026-04-07 | 1.1.29 | Decomposed the PSA GUI into focused `ui/` modules while tightening the compatibility export surface to immutable `__all__` tuples in both the facade module and the extracted UI package. |
| 2026-04-07 | 1.1.30 | Extracted the public enums/dataclass contracts and low-level helper kernels for `time_series_decomposition` into focused support modules, leaving the main module centered on decomposition orchestration while preserving the existing public import surface through the compatibility facade. |
| 2026-04-08 | 1.1.31 | Memoize AnalyticsSuite chart data using useMemo and optimize the scatter regression component with a single-pass loop, drastically reducing React rendering and GC overhead. |
| 2026-04-08 | 1.1.32 | Optimized data array filtering in `useDataProcessor.ts` by replacing `Array.push()` calls with `Float64Array` buffers in `calculateTrendline`, and replacing chained `filter()` passes in `trimTimeRange` with a single-pass `for` loop that avoids creating and resizing intermediate arrays. |
| 2026-04-09 | 1.1.33 | Added a loading spinner and `aria-pressed` states to the `VideoEditor.tsx` component in the video processor web application to improve user experience and accessibility during video export operations. |
| 2026-04-09 | 1.1.35 | Added a shared provider-pack manifest for the pendulum simulator under `src/pendulum_simulator`, plus a repo-local validator and regression tests that keep the manifest aligned with the real package entry point, working directory, Python path, icon asset, and launcher metadata required for future UpstreamDrift shared-launch integration. |
| 2026-04-09 | 1.1.34 | Wrapped DataTableView, PlotView, and AnalyticsSuite in `React.memo`, and memoized activeSignals with `useMemo` to prevent expensive visualization re-renders on unrelated UI state changes. |
| 2026-04-10 | 1.1.37 | Add explicit focus-visible styles to the interactive buttons (Upload New Video, Play/Pause, Mute/Unmute) within the `VideoPlayer` component for improved keyboard navigation visibility. |
| 2026-04-12 | 1.1.48 | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead. |
| 2026-04-15 | 1.1.49 | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead. |
| 2026-04-17 | 1.1.50 | Hardened model import security by enforcing HTTPS GitHub host allowlisting for remote model-library fetches, validating user-provided GitHub repository URLs before import, dropping directory components from remote mesh names, and rejecting separator-containing URDF viewer filenames before filesystem resolution. |
| 2026-04-21 | 1.1.67 | Optimized row copying logic in useDataProcessor.ts by replacing `Object.keys()` with a `for...in` loop and `hasOwnProperty`, substantially reducing GC allocation overhead inside tight data processing loops. |
| 2026-04-21 | 1.1.66 | Refreshed regression test coverage for architecture boundaries, data-processor compatibility, folder archive operations, and upstream-drift contract smoke behavior while keeping the production implementation unchanged. |
| 2026-04-22 | 1.1.90 | Repaired CI dependency bootstrap workflows so shared runners with broken `wheel` metadata upgrade `pip` and `setuptools` separately, then reinstall `wheel` with `--no-deps` before workflow linting and Python test jobs. |
| 2026-04-22 | 1.1.91 | Hardened data-processor normalize and standardize transforms so constant columns raise `TransformationError` instead of silently producing all-NaN output, with regression coverage preserving original data after the failed transform. |
| 2026-04-22 | 1.1.89 | Hardened `utils.env_utils` repo-root fallback discovery so shallow path layouts no longer raise import-time index errors, and added regression coverage for shallow fallback computation behavior. |
| 2026-04-22 | 1.1.93 | Enforced finite, non-negative altitude preconditions for the Rust standard-atmosphere model and added operator whitelisting before `DataProcessorEngine.filter_data()` constructs pandas query expressions. |
| 2026-04-22 | 1.1.94 | Updated the shared `DataProcessor.apply_filter()` Butterworth path to use an explicit `sample_rate` or infer it from time-column spacing instead of hard-coding 1000 Hz, with regression coverage for non-1 kHz datasets. |
| 2026-04-22 | 1.1.95 | Canonicalized the Rust universal gas constant by updating `math::R_GAS` to the full CODATA value and having `engineering::R_UNIVERSAL` reuse the same constant. |
| 2026-04-23 | 1.1.102 | Updated Unit Converter `removeCustomUnit` workflow to use an inline soft confirm pattern, eliminating thread-blocking `confirm()` dialogs and improving accessibility with `aria-live`. |
| 2026-04-28 | 1.1.112 | Updated Unit Converter UI to dynamically retarget labels for custom combobox search inputs, ensuring explicit accessible names and resolving click-to-focus gaps. |
| 2026-05-02 | 1.1.121 | Preserved `smoothAngles` behavior for fractional moving-average window sizes by dividing optimized mid-window sums by the actual sample span, added a Vitest regression in the golf video-processor web app, hardened the benchmark plugin bootstrap in CI/benchmark workflows against shared-runner cache drift, and restored the CI Standard coverage-policy skip for PRs that touch no Python source or Python tests. |
| 2026-05-01 | 1.1.120 | Hardened the calculator web expression validation gate by rejecting Python object hierarchy, lifecycle, async, import, and control-flow injection markers before SymPy parsing. |
| 2026-05-01 | 1.1.119 | Replaced the ODESolverCalculator data-table `.filter().map()` chain with a single-pass `for` loop that pre-allocates a result array and iterates in steps, eliminating O(N) intermediate array allocations and reducing GC pressure during large-dataset renders. |
| 2026-05-03 | 1.1.122 | Optimized row copying logic in useDataProcessor.ts by replacing the slow `for...in` and `hasOwnProperty` check with `Object.keys()` and a standard `for` loop, eliminating prototype chain crawling overhead. |
| 2026-05-03 | 1.1.123 | Hardened Folder Packer Pro archive extraction against absolute and parent-traversal member paths, made vessel drafter positive-value contracts accept both legacy and fleet-standard argument order, repaired the production Docker wheel build/install path, expanded Docker context cache exclusions, made CI quality-gate jobs informational, and lengthened Jules issue resolver polling. |
| 2026-05-01 | 1.1.118 | Bound the CI Standard workflow's dependency bootstrap to `python -m pip` in both quality-gate and test-matrix jobs so pytest plugins, including `pytest-benchmark`, install into the same interpreter that later runs `python -m pytest`. |
| 2026-05-01 | 1.1.117 | Made the shared syngas water vapor-pressure helpers return explicit `float` values so delta `mypy` checks stay green while preserving the `water_fraction` compatibility alias for downstream consumers. |
| 2026-05-01 | 1.1.116 | Tightened signal generator and acid gas dewpoint precondition handling so short chirp inputs, zero-frequency periodic signals, and non-positive dewpoint partial pressures raise deterministic `ValueError` messages. |
| 2026-04-30 | 1.1.115 | Hardened CI packaging and workflow checks by pinning the setuptools build backend below 82, using the supported package-data wildcard for `py.typed` markers, scanning merge-conflict markers with tracked-file `git grep`, normalizing detect-secrets result comparisons, and tolerating missing or empty benchmark JSON artifacts. |
| 2026-04-30 | 1.1.114 | Integrated full-text live search into the Unified Tools Launcher tabs, including name, description, keyword, multi-word, and punctuation-normalized matching, with Ctrl+F focus and Esc clear shortcuts. |
| 2026-05-24 | 1.1.113 | Fixed a vulnerability in CSRF cookie parsing logic where cookies with values containing an equals sign were previously being truncated. This allows base64 encoded CSRF tokens with padding to be parsed correctly. |
| 2026-05-11 | 1.1.127 | Replaced `.map()` array allocations in the `rk4Step_golfer` numerical integration function with pre-allocated arrays and standard `for` loops in `physics_golfer.ts` to reduce GC overhead. |
| 2026-05-15 | 1.1.180 | Replaced `.map()` array allocations inside `physics_golfer.ts` constraint and torque loops with pre-allocated arrays and standard `for` loops to reduce GC overhead. |
| 2026-05-13 | 1.1.141 | Made the migrated Video Analyzer installable and launchable from Tools for issue #2585 by adding package discovery, a `video-analyzer` console script, optional video runtime dependencies, installed-package import paths, and focused packaging/launcher regression tests. |
| 2026-05-13 | 1.1.140 | Registered the migrated Video Processor web surface in the canonical GUI launcher manifest and generated tools catalog, with regression coverage proving shared UpstreamDrift-visible tools expose their expected launch surfaces (#2585). |
| 2026-05-12 | 1.1.139 | Refreshed the module-size budget baseline for the updated rotation converter PyQt launcher after the branch was brought current with main. |
| 2026-05-15 | 1.1.139 | Refactored RK4 expression compilation in ODESolver to pass parameters as a direct array, avoiding spread operator allocation in tight integration loops. |
| 2026-05-15 | 1.1.139 | Refactored RK4 expression compilation in ODESolver to pass parameters as a direct array, avoiding spread operator allocation in tight integration loops. |
| 2026-05-12 | 1.1.138 | Hardened CI test-matrix dependency setup against stale self-hosted runner NumPy/SciPy binary caches and routed provider-contract tests through the active Python interpreter. |
| 2026-05-12 | 1.1.137 | Corrected the coverage policy gate to ratchet from the committed total-coverage baseline until the repository reaches the configured 60% target, while preserving package thresholds and regression checks. |
| 2026-05-12 | 1.1.136 | Resolved type-checking errors by properly implementing abstract methods (send_message, validate_connection, capabilities) for RustAgentAdapter, and fixed GUI theme and categorization issues in UpstreamDrift chat functionality. |
| 2026-05-19 | 1.1.184 | Replaced `.reduce()` with a standard `for` loop in `calculatePhaseConfidence` to eliminate callback allocation and garbage collection overhead during high-frequency pose frame confidence calculations in the video processor. |
| 2026-05-20 | 1.1.193 | Clarified shared chat provider dropdown ownership by removing stale UpstreamDrift issue references from Tools-owned source and tests, and synchronized the GitHub CLI provider descriptor with the default terminal registry (#3020). |

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

<!-- prettier-ignore-end -->

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

### Version 1.1.393

- 2026-06-12: refactor(data-processor) — move the shared
  `RustBulkDataEngine` compatibility facade into `bulk_facade.py` while
  preserving `data_processor.rust_engine` re-exports, keeping the CI
  changed-file size budget green without changing runtime behavior.

### Version 1.1.392

- 2026-06-12: fix(data-processor) — expose `DataProcessorRustError` and a
  `RustBulkDataEngine` compatibility facade from the shared data-processor
  fallback package so source-tree import order cannot shadow the full
  data-processor package and break `data_processor.core.data_loader`.

### Version 1.1.333

- 2026-06-10: feat(ai) — marshal GUI-affine chat tools onto the main
  thread. `Tool` gains an opt-in `requires_main_thread` flag; `ToolRegistry`
  gains `set_main_thread_dispatcher` and routes flagged tools through it in
  `execute` (running inline when no dispatcher is installed, so headless use
  is unaffected). `MainThreadToolDispatcher` (ai/gui) marshals a tool thunk
  from the background `StreamWorker` thread onto its owning GUI thread via a
  queued signal, returning the result synchronously and re-raising errors on
  the caller; same-thread calls run inline. Decorator-registered tools can
  opt in through `ToolRegistry.register(..., requires_main_thread=True)`, so
  the normal shared-tool registration path preserves GUI-thread affinity.
  `AIAssistantPanel` installs the dispatcher on the global registry at
  startup and uses explicit boundary return types so skipped-import mypy runs
  remain type-clean at the panel boundary. Additive and backward compatible
  for existing downstream registrations.

### Version 1.1.332

- 2026-06-10: fix(ci) — keep the P1AM project import helper mypy-clean under
  the changed-file quality gate by typing parsed SCADA tags as `TagDefinition`
  at the parser boundary and preserving the project import endpoint's
  documented `dict[str, Any]` response contract when imports are skipped.

### Version 1.1.331

- 2026-06-10: fix(daemon, #3291) — stop `start-gaai-daemon.sh` from
  writing `~/.claude/settings.json` or globally suppressing Claude Code
  dangerous-mode prompts; document that any safety override must be configured
  deliberately outside the launcher, and add a dry-run regression test proving
  existing global Claude settings are preserved.

- 2026-06-10: ci(security) — harden
  `.github/workflows/anti-phantom-merge.yml` so the privileged
  `pull_request_target` label path never checks out untrusted PR head code.
  Full git-diff phantom checks continue to run only on `pull_request`; label
  events validate the admin override through GitHub API calls and emit notices
  for ignored non-admin overrides. Added an ops regression that scans all
  `pull_request_target` workflows for unguarded `actions/checkout` steps using
  `github.event.pull_request.head.sha`, preserving the invariant across future
  workflow edits.

### Version 1.1.330

- 2026-06-10: test(ci) — include existing Sidekick state-manager regression
  suites in Sidekick-changed CI slices before the per-file Sidekick coverage
  gate runs, keeping changed-file coverage enforcement aligned with the module
  that triggered the gate; restored JSON serialization of simple object
  class-level defaults while keeping instance attributes authoritative; keep the
  state-manager import side-effect subprocess on the same shared `utils` source
  path used by clean CI runners.
- 2026-06-09: fix(import-contracts) — keep rotation-converter NumPy helpers,
  screw-axis animation callbacks, and the video-analyzer DbC shim mypy-clean
  under changed-file CI by adding explicit typed array boundaries and
  non-redefining contract import fallbacks without changing runtime validation
  behavior; declare the import-contract subprocess bootstrap as assertion-free
  test support so the changed-test assertion ratchet continues to block real
  assertion-light test cases without forcing fake assertions into helpers.

### Version 1.1.329

- 2026-06-09: feat(movement optimizer) — restore the standalone PyQt6 swingset policy-training and segmented-chain whip-dynamics tabs, mypy-compatible typed model and Qt UI modules, focused model and UI tests, launcher bootstrap repair, and provider metadata so UpstreamDrift can discover the Movement Optimizer tile from remote main.
- 2026-06-09: fix(sidekick) — preserve the Python REPL `WorkspaceRegistry`
  contract across canonical and deprecated compatibility import paths so
  legacy `upstream_drift_tools` callers pass the same runtime precondition as
  canonical Sidekick callers without weakening the TypeError guard for
  unrelated registry objects.

### Version 1.1.320

- 2026-06-09: refactor(tools, #3261, #3262, #3263) — replaced the duplicated `scripts/mypy_autofix_agent.py` implementation with a compatibility wrapper that delegates to the canonical `src.tools.mypy_autofix_agent` entrypoint, preserving direct script execution while reducing audit-reported DRY debt. Added focused tests that guard the delegation contract and CLI help path.

### Version 1.1.265

- 2026-06-01: feat(ai): Added `AdapterReviewerLLMClient`, a production `ReviewerLLMClient` backed by `BaseAgentAdapter` that builds a structured JSON prompt, runs `send_message` off the event loop, and parses verdict/reasoning/confidence (malformed JSON → `abstain`, confidence clamped to [0,1]). Wired it as the default via `peer_review.registry.default_llm_client()`, which selects the production client when an adapter is available and falls back to `StubReviewerLLMClient` offline (#3177). Added behavioral tests for the four CLI adapters (`claude_code`/`codex_cli`/`gemini_cli` via mocked `subprocess.run`, `cline` via mocked httpx) covering success, non-zero-exit/timeout/missing-binary error classification, `_strip_telemetry`, and `validate_connection` paths (#3178).

### Version 1.1.258

- 2026-06-01: test(theme): Added focused FastAPI router coverage for built-in/custom theme listing, active-theme retrieval and updates, custom-theme save/delete errors, Pydantic request models, and registration guards, raising `src/shared/python/theme/api.py` focused coverage to 100%.

### Version 1.1.257

- 2026-06-01: test(theme): Normalized font manager and responsive theme tests to import through the exported `src.shared.python.theme` package path so the provider-contract suite passes under importlib mode.

### Version 1.1.256

- 2026-06-01: test(theme): Added focused font manager coverage for QSettings persistence, singleton reuse, font-change signaling, application font application, and no-application warning behavior; fixed PyQt6 font database enumeration to use the static API and tightened adjacent theme helper return typing for strict mypy.

### Version 1.1.255

- 2026-06-01: test(theme): Added focused responsive PyQt helper coverage for maximum-width clamping, invalid contracts, generic widget text derivation, and zero/negative scroll-area width handling.

### Version 1.1.233

- 2026-06-01: fix(ux) — improve accessibility of pendulum simulator model selector tabs by removing conflicting `aria-pressed` attributes and replacing them with standard `role="tablist"`, `role="tab"`, `role="tabpanel"`, and `aria-selected` attributes.

### Version 1.1.232

- 2026-05-30: feat(ux, #3115) — improve accessibility of the ODE Solver UI by explicitly linking `label`s to `input`s and `textarea`s using `htmlFor` and unique `id`s generated by `React.useId()`. Add `spellcheck="false"` and disabled autocorrect on math text areas to prevent mobile OS interference with equations.

### Version 1.1.231

- 2026-06-09: ci(secret-scan) — run the detect-secrets baseline scan through `python -m detect_secrets` so the configured Python environment, not the runner PATH, controls the installed scanner entrypoint.
- 2026-06-09: perf(p1am frontend) — optimize array aggregations and string operations in `LadderExplorer.tsx` by replacing chained `.map().filter()` operations with a single-pass `for` loop and using `useMemo` to prevent main thread lag.
- 2026-05-30: perf(p1am frontend, #3126) — optimize array aggregations in `AlarmsHeader.tsx` by replacing chained `.filter()` and `.reduce()` operations with a single-pass `for` loop, eliminating intermediate array allocations and minimizing garbage collection overhead during high-frequency alarm updates.

### Version 1.1.230

- 2026-05-30: fix(sidekick): PyQt test worker crash fix and Module Size budget baseline adjustment (#3104, #3115)

### Version 1.1.221

- 2026-05-30: test(rust-engine, #3114) — the #2989 bulk-I/O contract suite now runs in CI after the Data Processor embedding PR fixed its import path; guard the parquet round-trip cases (`test_csv_to_parquet`, `test_parquet_destination`, and the `parquet_file` fixture) with a skipif on parquet-engine (pyarrow/fastparquet) availability so the lean CI test image skips them instead of failing, honoring the file's "runs in CI without native deps" contract. CSV contract cases continue to run unconditionally.

### Version 1.1.220

- 2026-05-29: fix(sidekick conversions, #3101) — reconcile `flow_rate_converter` with the DRY constants layer: `ton`/`ton/hr` now means a short ton (907.18 kg) fleet-wide (metric is `tonne`), STP is the IUPAC 0°C/1 bar definition, the gas constant and standard conditions import from `unit_constants`, `Nm3/hr` spellings are recognized, `convert_via_table` raises `ValueError` on unknown units, `_normalize_unit` raises `UnknownUnitError` (O(1)) instead of silently echoing, and the temperature path validates finiteness. Restored the four empty conversion test stubs with known-value and round-trip assertions.
- 2026-05-29: fix(sidekick process numerics, #3103) — remove the duplicated compressible-flow solver (`_flow_calculations` now imports the canonical `compressible_flow`) and the malformed in-sqrt expansion factor; solve WGS extent directly from the equilibrium constant so reported K and composition are self-consistent and guard `T>0`; replace precondition `assert`s with `ValueError` in flare/financial; raise on laminar `Re<=0`; return ideal-gas Z=1 for unknown-only compositions; flag compressible-solver non-convergence; clarify the acid-gas °C Antoine convention.
- 2026-05-29: fix(sidekick PSA UI, #3105) — refresh the sensitivity plot when components change (dirty flag + re-plot when visible); resolve the pre-calc tab trigger via `indexOf` instead of a magic index; size the O2 hazard band from the plotted data max so it can't collapse to the default y-limit.
- 2026-05-29: fix(sidekick widget/state layer, #3102) — wrap Data Processor engine ops (filter/query/aggregate/add/transform/rename/drop/fit) in `try/except DataProcessingError` so bad input shows a warning instead of crashing; validate corrupt saved-state shape (via an `Any`-typed alias so the runtime guard stays reachable) and broaden the load except; parent the auto-save `QTimer` to the host widget and guard `auto_save_state` against teardown; route unit-converter save/delete through `_get_row_by_index`; add a public `UnitConversionService.get_compatible_units`.

### Version 1.1.241

- 2026-06-01: Performance — Optimized correlation matrix calculation in AnalyticsSuite by precomputing column sums for the fast-path (no NaNs) and utilizing fast `x !== x` NaN checks.

### Version 1.1.219

- 2026-05-30: test(data-processor) — added skip guards to rust engine contract tests to handle missing parquet engines (pyarrow/fastparquet) gracefully in minimal test environments.

### Version 1.1.217

- 2026-05-30: feat(sidekick) — consolidated Sidekick quality and cleanup issues (#3106). Replaced global instantiation of `state_manager` with lazy-loading module `__getattr__` wrapper and deprecation warning to prevent eager directory creation on import. Added support for native matplotlib rendering of LaTeX formulas to crisp QPixmaps in `latex_renderer.py`, falling back to monospace text on missing dependencies. Added type annotations to state manager tests.
- 2026-05-30: feat(ui) — added a reusable `HoverCopyTextBrowser` widget with hover-triggered copy-to-clipboard overlay buttons and integrated it into the double pendulum simulator diagnostics/analysis tabs and error notifications. Excluded pendulum simulator from pre-push mypy checks.
- 2026-05-30: fix(sidekick) — validate C3D header magic bytes in `c3d_reader.py` and package-relative standard response import fixes.

### Version 1.1.216

- 2026-05-29: chore(sidekick) — type-gate hardening surfaced by the changed-file CI mypy run: `register_shortcuts` now resolves `QShortcut`/`QKeySequence` through the active `qt_compat` binding instead of a PyQt6/PyQt5 dual-import fork, `_default_tab_definitions` returns an annotated local rather than `cast()`, and `SidekickThemeSettings.__post_init__` widens the persisted-`font` branch through `Any` so the runtime-dict reconstruction type-checks. Clears pre-existing `no-redef`/`unused-ignore`/`redundant-cast`/`no-any-return`/`arg-type` findings on `sidebar.py` and `theme_settings.py`.
- 2026-05-29: feat(sidekick) — wired working ⚙ settings into the Chat, Terminal, Python REPL, and Workspace tabs (previously the gear was disabled on every tab except Data Explorer). New shared `appearance.py` (`PanelAppearance` value object + `panel_qss` generator) gives the terminal/REPL/workspace always-on visible borders and user-adjustable colours; the Workspace now shows an empty-state hint instead of blank white space and the REPL gained input/output labels. `chat_settings.py` adds provider/model/reasoning/agent-mode/auto-condense config plus keyring-backed API-key management. `runtime_tab_settings.py` adds per-tab appearance panels (native colour pickers) and a configurable preloaded scientific-package bundle (numpy/scipy/pandas/matplotlib/sympy) for the Python REPL, reusing the validated `CalculatorStartupConfig`. Added narrow `UnifiedToolsSidebar.tab_widget(tab_id)` accessor for live application (LOD). 130+ new tests; sidekick API stability baseline regenerated (purely additive).

### Version 1.1.215

- 2026-05-29: fix(sidekick) — validate C3D header magic bytes before handing files to ezc3d, returning a clear `ValueError` for truncated or mislabeled files and covering the pre-parser failure path with focused tests.

### Version 1.1.212

- 2026-05-29: chore(ci) — scope coverage policy package thresholds to tracked packages changed in the PR while preserving total coverage and Sidekick-specific coverage gates.
- 2026-05-29: fix(sidekick) — make the standard response API import its shared StrEnum helper via the repo package path while preserving top-level Sidekick compatibility (#3106).
- 2026-05-27: fix(p1am) — extend interlock contract to 4 limits (lolo/low/high/hihi) in SafetyInterlock to align with host. Chunk Modbus client read/write routing into 64-register packets to satisfy pymodbus's request size caps.
- 2026-05-27: chore(ci) — use `sudo rm -rf` in Python tool cache cleanup to ensure complete removal of corrupted files, and add cleanup step to topology-governance, detect-secrets, and cross-repo integration workflows.

### Version 1.1.208

- 2026-05-27: Optimized object allocations in `themeApi.ts` and `themeDefinitions.ts` by replacing `Object.fromEntries(Object.entries().map())` chains with single-pass loops to reduce memory allocation overhead on startup.
- 2026-05-27: feat(FilterPanel) — Added `useId` to dynamically generate linked IDs for form labels, select dropdowns, and inputs in `FilterPanel.tsx` via `htmlFor`, improving screen reader navigation. Also added `aria-invalid` and `aria-describedby` to announce validation error states clearly.

### Version 1.1.206

- 2026-05-26: feat(chat) — restored shared chat dock keybindings (Enter→submit, Shift+Enter→newline, busy-queue with steering), port-aware default WS URL (`UD_CHAT_WS_URL` / `GOLF_API_PORT` env), Ollama latency tuning (`keep_alive: "30m"`, `num_ctx: 4096`, native `tools` field), `_chat_dock_widget_qt.py` refactored into `_qt/` submodules (2091→1049 lines), and the animated "AI is thinking" indicator.
- 2026-05-26: Chat dock resolves its default WebSocket URL per-instance, keeps the Steer action queue-only, and preserves typed import-safe runtime diagnostics for the optional Qt chat surface.

### Version 1.1.204

- 2026-05-26: Optimized Nelder-Mead loop in `optimizer.ts` to mutate pre-allocated arrays in-place to avoid GC overhead.

### Version 1.1.190

- **Performance**: In `src/ode_solver/web/src/components/ODESolverCalculator.tsx`, extracted the entire `resultsPanel` (containing heavy Recharts and data table elements) into a `useMemo` block to prevent the entire SVG tree from re-rendering synchronously on every keystroke in the textarea, eliminating severe UI input lag.

### Version 1.1.190

- **Performance**: In `src/ode_solver/web/src/components/ODESolverCalculator.tsx`, wrapped `varNames` computation and summary cards rendering in `useMemo`, and replaced `.filter()` with a single-pass `for` loop to prevent O(N) recalculations of array keys and summary min/max loops on every React render.

- **2026-05-22**: Memoize summary statistics calculation and variable names in ODESolverCalculator.
- **2026-05-22**: Keep the model explorer package initializer lint-clean by preserving the module docstring before future imports.
- **2026-05-20**: Suppress shared chat dock WebSocket reconnect scheduling during intentional widget close while retaining reconnects for unexpected disconnects.
- **2026-05-20**: Add accessible toggle states and toast feedback for copy actions in `calculator/static/app.js` and `calculator/templates/index.html`.
- **2026-05-20**: Harden health-check API responses to return generic client-facing errors while logging exception details server-side.
- **2026-05-20**: Restore shared logging and environment helper modules required by AI adapter and chat service connection imports.
- **2026-05-20**: Clarify shared chat provider dropdown ownership by removing stale UpstreamDrift issue references from Tools-owned source and tests, and synchronize the GitHub CLI provider descriptor with the default terminal registry (#3020).
- **2026-05-30**: Resolve mypy type check errors in core data loader, signal processing, and Sidekick embedding.

## 1.1.241 - Replaced chained .filter() array passes with single-pass for-loops in Golf UI components

- **2026-06-09**: fix(sidekick) — regenerate `sidekick_api_baseline.json` to include the `data_processing/formats.py` module added in ed6e415fa (only addition; no public API removals or signature changes), and correct `test_json_serializer` to set the Dummy attribute on the instance `__dict__` so it matches `_json_serializer`'s `hasattr(obj, "__dict__")` branch.
- **2026-06-10**: Fix command injection bypasses in `cli_tools.py` `ShellTool._is_command_allowed` by explicitly parsing executable names using `pathlib.Path` and parsing arguments with assignment flags.
- **2026-06-11**: fix(conversion) — `convert_gas_flow_scfm_acfm` now validates inputs as a true precondition: a non-positive/non-finite `compressibility_factor` raises `ValueError` (instead of silently passing through on ACFM→SCFM), and an explicitly supplied non-positive/non-finite `actual_temp_K`/`actual_pressure_kPa` raises `ValueError` instead of being coerced to the standard-condition default via the falsy-`0.0` `or` idiom. Reconciles the #3344 gas-flow guard refactor with the restored #3367/#3342 compressibility validation tests during PR consolidation (#3380).
- **2026-06-11**: test(p1am) — make the P1AM backend functional suite (`src/p1am_control_system/backend/tests/test_backend.py`) robust to cross-module test-collection order. An autouse fixture now re-asserts `P1AM_DEV_NO_AUTH=1` and the `get_session` dependency override per-test (restoring prior values on teardown), so the sibling security suite's import-time `os.environ.pop("P1AM_DEV_NO_AUTH")` and competing `app.dependency_overrides` no longer leak 503 auth-gate / `no such table` failures into these tests (#3289/#3292, surfaced during #3380 consolidation).
