# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-03-28

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

| Field | Value |
|-------|-------|
| **Repository Name** | `Tools` |
| **GitHub URL** | `https://github.com/D-sorganization/Tools` |
| **Owner** | D-sorganization |
| **Primary Language(s)** | Python 3.10+, Rust, JavaScript, TypeScript |
| **License** | MIT |
| **Current Version** | N/A |
| **Spec Version** | 1.1.1 |
| **Last Spec Update** | 2026-05-18 |

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

| Component | Location | Purpose |
|-----------|----------|---------|
| UnifiedToolsLauncher | `UnifiedToolsLauncher.py` | PyQt6 GUI for tool discovery and execution |
| Plugin System | `src/python/plugin_system/` | Discover, load, and manage plugins |
| Shared Utilities | `src/python/shared_utilities/` | Common functions, decorators, error handling |
| Engineering Tools | `src/tools/` | 45+ specialized calculation and processing tools |
| Data Processing | `src/data_processing/` | Pipelines, transformers, validators |
| Document Processing | `src/document_processing/` | PDF extraction, text processing |
| Media Processing | `src/media_processing/` | Audio and video utilities |
| Scientific Modeling | `src/scientific_modeling/` | Thermal, mechanical, chemical simulations |
| Web Services | `src/web_applications/api/` | FastAPI endpoints and integrations |
| Rust Kernels | `rust_core/` | High-performance mathematical operations |
| MATLAB Integration | `matlab/` | Wrapped MATLAB scientific code |
| Fleet Theme System | `src/python/shared_utilities/theme/` | Consistent UI theming across tools |

## 5. Desired Functionality

### Core Features

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F1 | UnifiedToolsLauncher (PyQt6 GUI) | ✅ | Main entry point with tool discovery, search, favorites, and launch |
| F2 | 45+ engineering calculation tools | ✅ | Diverse tools for calculations, conversions, analysis |
| F3 | Rust math primitives | ✅ | Performance-critical numerical operations in Rust |
| F4 | Shared upstream_drift_tools library | ✅ | Common utilities for drift detection and analysis |
| F5 | Plugin discovery system | ✅ | Auto-discover tools via plugin registry, support dynamic loading |
| F6 | FastAPI web interfaces | 🔄 | RESTful API for programmatic access to tools |
| F7 | MATLAB scientific tools | ✅ | Integration with MATLAB code and wrappers |
| F8 | Fleet theme system | ✅ | Consistent theming across all PyQt6 GUIs |

### API / Interface Contract

**GUI:**
- `UnifiedToolsLauncher()` — Main launcher application
- Tools accessed via discoverable plugin interface
- Search, filter, favorites, recent tools in UI

**CLI:**
- `python -m tools <tool_name> [args]` — Command-line invocation
- `python -m tools --list` — List available tools
- `python -m tools --help <tool_name>` — Tool-specific help

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

| Input | Format | Source | Schema |
|-------|--------|--------|--------|
| Tool parameters | JSON/YAML/Python dict | User input, CLI args, API calls | Per-tool schema (JSON-schema) |
| Configuration files | YAML | `config/` | Tool registry, theme config, plugin paths |
| Scientific data | CSV/HDF5/NetCDF | Files, databases | Domain-specific formats |
| MATLAB models | .m/.mat files | `matlab/` | MATLAB simulation parameters |

### Output Data

| Output | Format | Destination | Description |
|--------|--------|-------------|-------------|
| Calculation results | JSON/CSV/HDF5 | User's disk, API response | Tool output matching schema |
| Cached results | SQLite/HDF5 | `.cache/` | Memoized expensive calculations |
| Logs | JSON/text | `logs/` | Tool execution logs with timings |
| Reports | HTML/PDF | User's disk | Generated analysis reports |

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

| Category | Location | Framework | Markers |
|----------|----------|-----------|---------|
| Unit | `tests/unit/` | pytest | `@pytest.mark.unit` |
| Integration | `tests/integration/` | pytest | `@pytest.mark.integration` |
| Acceptance | `tests/acceptance/` | pytest | `@pytest.mark.acceptance` |
| Contract | `tests/contract/` | pytest | `@pytest.mark.contract` |
| GUI | `tests/gui/` | pytest-qt | `@pytest.mark.gui` |
| DWSIM | `tests/dwsim/` | pytest | `@pytest.mark.dwsim` |
| Slow | `tests/slow/` | pytest | `@pytest.mark.slow` |

### Coverage Requirements

| Scope | Minimum | Current | Enforced By |
|-------|---------|---------|-------------|
| Overall | 60% | ~72% | CI (`--cov-fail-under=60`) |
| Core tools | 75% | ~81% | CI |
| Plugin system | 80% | ~85% | CI |

### Required Test Scenarios

- [ ] Tool instantiation returns valid object with correct schema
- [ ] UnifiedToolsLauncher starts and displays available tools
- [ ] Plugin discovery finds all registered tools
- [ ] Calculation produces deterministic results for same inputs
- [ ] Web API endpoint validates input and returns JSON response
- [ ] Rust kernel outperforms pure Python equivalent by 10x+
- [ ] Theme system applies consistently across all GUI tools
- [ ] Data processing pipeline handles malformed input gracefully

## 8. Quality Standards

### Code Quality Tools

| Tool | Version | Purpose | Blocking? |
|------|---------|---------|-----------|
| ruff | latest | Linting + formatting | Yes |
| mypy | latest | Type checking | Yes |
| pytest | latest | Testing framework | Yes |
| bandit | latest | Security scanning | Yes |
| pip-audit | latest | Dependency vulnerability scanning | Yes |

### Design Principles

- **TDD**: Yes — tests written before/with implementation for core tools
- **Design by Contract (DbC)**: Yes — schema validation, precondition/postcondition checks
- **DRY**: Yes — shared_utilities module reduces duplication across tools
- **Orthogonality**: Yes — tools are independent, composable, minimal coupling

### CI/CD Pipeline

| Workflow | Trigger | Purpose | Blocking? |
|----------|---------|---------|-----------|
| `ci-standard.yml` | Push/PR | Unit tests, linting, type checking | Yes |
| `test-matrix.yml` | Push/PR | Test on Python 3.10/3.11/3.12 | Yes |
| `integration-tests.yml` | Push/PR | Integration and contract tests | Yes |
| `gui-tests.yml` | Push/PR | GUI rendering and interaction tests | Yes |
| `rust-build.yml` | Push/PR | Rust kernel compilation and benches | Yes |
| `dwsim-tests.yml` | Manual trigger | DWSIM simulation tests (long-running) | No |
| `security-scan.yml` | Daily | bandit + pip-audit | Yes |
| `performance-bench.yml` | Weekly | Benchmark Rust kernels vs alternatives | No |
| `build-release.yml` | Tag push | Build wheels, binaries, docs | Yes |

## 9. Dependencies

### Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | latest | Numerical computing |
| scipy | latest | Scientific functions |
| pandas | latest | Data frames and manipulation |
| matplotlib | latest | Plotting and visualization |
| sympy | latest | Symbolic mathematics |
| pydantic | latest | Data validation |
| PyYAML | latest | YAML parsing |
| defusedxml | latest | Safe XML parsing |
| PyQt6 | latest | GUI toolkit |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | latest | Testing framework |
| pytest-cov | latest | Coverage reporting |
| pytest-xdist | latest | Parallel test execution |
| pytest-timeout | latest | Test timeout enforcement |
| pytest-benchmark | latest | Performance benchmarking |
| pytest-qt | latest | PyQt6 testing utilities |
| mypy | latest | Type checking |
| ruff | latest | Linting and formatting |
| bandit | latest | Security scanning |
| pip-audit | latest | Dependency audit |

### Optional Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| urdf | urdfpy, trimesh | Robot URDF parsing and manipulation |
| signal | scipy.signal extensions | Signal processing tools |
| process | thermodynamics libs | Process engineering calculations |
| robotics | PyBullet, ikpy | Robotics simulation and kinematics |
| gui | PyQt6, plotly | GUI and interactive visualization |
| theme | custom theme libs | Advanced theming and styling |
| pid | control, slycot | PID controller design |
| cad | CadQuery, Fusion 360 API | CAD generation and integration |
| dwsim | DWSIM COM integration | Process simulation (Windows only) |

### Fleet Dependencies

| Repo | Relationship | Description |
|------|-------------|-------------|
| Repository_Management | Depends on | Consumes templates, workflows, skills |
| Tools_Private | Depends by / Depends on | Shares test patterns, assessment framework |
| [Other fleet repos] | Depends by | Optional integration with Tools utilities |

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

| Artifact | Format | Destination |
|----------|--------|-------------|
| Python wheel | `.whl` | PyPI / `dist/` |
| Source distribution | `.tar.gz` | PyPI / `dist/` |
| Rust binaries | `.so`/`.pyd` | Embedded in wheel |
| Documentation | HTML | `docs/_build/html/` |
| Test reports | HTML/JSON | `reports/` |

## 11. Roadmap & Open Issues

### Current Phase

Active development with stable core, continuous tool expansion, and web API in progress.

### Planned Work

| Priority | Item | Issue/PR | Target Date |
|----------|------|----------|-------------|
| P0 | Complete FastAPI web interfaces (F6) | TBD | Q2 2026 |
| P1 | Add 10 more scientific tools | TBD | Q2 2026 |
| P1 | Optimize Rust kernels for multi-core | TBD | Q3 2026 |
| P2 | Plugin marketplace / registry | TBD | Q3 2026 |
| P2 | Cloud deployment templates | TBD | Q4 2026 |

### Known Limitations

- DWSIM integration Windows-only (COM interface limitation)
- Some MATLAB tools require MATLAB runtime installed
- Large datasets may cause GUI slowdowns without optimization
- Plugin system doesn't yet support hot-reloading
- Web API authentication/authorization not yet implemented

## 12. Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-03-28 | 1.0.0 | Initial specification |
| 2026-03-29 | 1.0.1 | Document performance improvement in DataChart downsampling algorithm |
| 2026-03-30 | 1.0.2 | A-N assessment remediation: LoD refactoring in convert_tools_icon.py, launch.py, launch_signal_toolkit.py, verify_launcher.py; DbC input validation added to launch_tool, bootstrap, migrate_file, _print_environment_info, _check_launcher_file, _print_recommendations, _on_poly_generated; docstrings added to __init__ and missing functions in setup_dev.py, remove_broken_scripts.py, migrate_print_to_logging.py, launch_signal_toolkit.py. |
| 2026-03-31 | 1.0.3 | Fix CI import error in tests/shared/python/test_contracts.py and optimize React rendering in ToolsPanel. |
| 2026-04-01 | 1.0.4 | Add keyboard accessibility (focus-within) to video player controls in web application. |
| 2026-04-01 | 1.0.5 | Optimize the data processor median filter to reuse a `Float64Array` buffer and preallocate result storage, reducing per-window allocations during large CSV filtering workflows. |
| 2026-04-02 | 1.0.6 | Refactored AnalyticsSuite (computeCorrelation, computeRegression, pearsonCorrelation) to use iterative primitive arrays and eliminate chained .map/.filter mapping overhead, vastly reducing garbage collection pressure. |
| 2026-04-02 | 1.0.7 | Run comprehensive assessments and apply auto-fixes across the repository. |
| 2026-04-03 | 1.0.8 | Refactor `linearRegression` and `polynomialRegression` in `useDataProcessor.ts` to replace multiple consecutive `.reduce()` and `.map()` array iteration methods with single-pass `for` loops, improving performance for large datasets. |
| 2026-04-10 | 1.0.9 | Optimize Math Functions using single-pass loops. |
| 2026-04-10 | 1.1.0 | Add keyboard accessibility and focus management to the Data Processor web application file upload dropzone. |
| 2026-05-18 | 1.1.1 | Fix command injection vulnerability in MATLAB Quality Utils by escaping single quotes in paths passed to MATLAB and Octave shells. |
| 2026-05-19 | 1.1.2 | Refactor `gaussianFilter` in `useDataProcessor.ts` to replace array `.push()` calls with pre-allocated Float64Array for kernels and mapped indexes to improve performance. |

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
