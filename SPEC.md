# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-04-09

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
| **Primary Language(s)** | Python 3.10+, Rust, JavaScript, TypeScript |
| **License**             | MIT                                        |
| **Current Version**     | N/A                                        |
| **Spec Version**        | 1.1.48                                     |
| **Last Spec Update**    | 2026-04-12                                 |

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
- Support multiple Python versions (3.10, 3.11, 3.12) with comprehensive test matrix

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

| Component                | Location                                                                               | Purpose                                                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| UnifiedToolsLauncher     | `UnifiedToolsLauncher.py`                                                              | PyQt6 GUI for tool discovery and execution                                                                                             |
| Plugin System            | `src/python/plugin_system/`                                                            | Discover, load, and manage plugins                                                                                                     |
| Shared Utilities         | `src/python/shared_utilities/`                                                         | Common functions, decorators, error handling                                                                                           |
| Pressure Drop Calculator | `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/` | Facade-driven gas pressure-drop workflows with extracted API, validation, reference, results, and engine-domain helper modules         |
| Model Generation API     | `src/shared/python/model_generation/api/`                                              | Route facade with framework-specific Flask and FastAPI adapters behind a compatibility shim                                            |
| Engineering Tools        | `src/tools/`                                                                           | 45+ specialized calculation and processing tools                                                                                       |
| Data Processing          | `src/data_processing/`                                                                 | Pipelines, transformers, validators, and facade-based data-processor core modules for exporter, ANOVA, and vectorized filter workflows |
| Document Processing      | `src/document_processing/`                                                             | PDF extraction, text processing                                                                                                        |
| Media Processing         | `src/media_processing/`                                                                | Audio and video utilities                                                                                                              |
| Scientific Modeling      | `src/scientific_modeling/`                                                             | Thermal, mechanical, chemical simulations                                                                                              |
| Web Services             | `src/web_applications/api/`                                                            | FastAPI endpoints and integrations                                                                                                     |
| Rust Kernels             | `rust_core/`                                                                           | High-performance mathematical operations                                                                                               |
| MATLAB Integration       | `matlab/`                                                                              | Wrapped MATLAB scientific code                                                                                                         |
| Fleet Theme System       | `src/python/shared_utilities/theme/`                                                   | Consistent UI theming across tools                                                                                                     |

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

### Output Data

| Output              | Format        | Destination               | Description                      |
| ------------------- | ------------- | ------------------------- | -------------------------------- |
| Calculation results | JSON/CSV/HDF5 | User's disk, API response | Tool output matching schema      |
| Cached results      | SQLite/HDF5   | `.cache/`                 | Memoized expensive calculations  |
| Logs                | JSON/text     | `logs/`                   | Tool execution logs with timings |
| Reports             | HTML/PDF      | User's disk               | Generated analysis reports       |

### Configuration

**Environment Variables:**

- `TOOLS_PLUGIN_PATH` — Colon-separated paths to plugin directories
- `TOOLS_THEME` — Default theme name (light/dark/custom)
- `TOOLS_CACHE_DIR` — Cache directory for results
- `TOOLS_LOG_LEVEL` — Logging verbosity (DEBUG/INFO/WARNING)
- `TOOLS_RUST_WORKERS` — Number of Rust kernel worker threads

**Config Files:**

- `config/tools_registry.yml` — Available tools and metadata
- `config/theme_config.yml` — Theme settings and customization
- `config/plugin_config.yml` — Plugin discovery paths and settings
- `config/web_api_config.yml` — FastAPI server configuration

## 7. Testing Specification

### Testing Strategy

Test pyramid with unit tests at the base, integration tests for tool interactions, acceptance tests for end-to-end workflows. Markers organize tests by category: unit, integration, acceptance, contract, and slow. GUI and Rust components tested separately.

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

### Coverage Requirements

| Scope         | Minimum | Current | Enforced By                |
| ------------- | ------- | ------- | -------------------------- |
| Overall       | 60%     | ~72%    | CI (`--cov-fail-under=60`) |
| Core tools    | 75%     | ~81%    | CI                         |
| Plugin system | 80%     | ~85%    | CI                         |

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
| 2026-04-12 | 1.1.48  | Added `tools.shared.python.model_generation.editor` compatibility namespace so downstream repos can import the text editor via `tools.shared.python` without duplicating the module; added `-p no:xvfb` to pytest addopts so the test suite runs on headless self-hosted runners that lack Xvfb; applied ruff formatting fixes across GUI stylesheets and multiline string literals.                                                                                                                                                                                |
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

- The application uses `Float64Array` and iterative loops instead of `Array.prototype.map`/`filter`/`reduce` to optimize memory and processing speed for large numerical datasets, including reusable typed-array buffering for median-filter windows in `useDataProcessor.ts`. Chained array functional methods like `reduce` and `map` have been largely replaced with standard iterative loops in mathematical computation methods such as `zScoreFilter`, `linearRegression` and `polynomialRegression`.
- Mathematical matrix calculations such as Principal Component Analysis (PCA) utilize column-wise typed arrays (e.g. `Float64Array` buffers) rather than traditional N x P row-wise arrays, drastically reducing O(N) allocation overheads and mitigating garbage collection pauses on large scale analysis.
- Linear regression and sum-of-squares calculations in `AnalyticsSuite` leverage pre-allocated arrays and single-pass loops to prevent allocation and garbage collection overhead typical of functional `.map()` and `.reduce()` operations in large dataset pipelines.
- The PCA power iteration algorithm in `AnalyticsSuite` has been optimized to remove `.map()` and `.reduce()` from the tight inner loop, using pre-allocated arrays and standard `for` loops to eliminate thousands of allocations per execution.
- PlotView WebGL rendering uses pre-allocated `Float64Array` buffers and single-pass loops instead of `data.map()`, eliminating O(N) intermediate array allocations for large datasets.
- Pearson correlation matrix computations utilize a single-pass loop algorithm, calculating sums concurrently to drastically reduce iteration overhead compared to two-pass implementations, while carefully mitigating numerical instability via clamping.
- Recharts component props in `AnalyticsSuite` are memoized using `useMemo` hooks to provide stable references and prevent expensive internal re-renders.
