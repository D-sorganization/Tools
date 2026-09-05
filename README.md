# Tools

[![CI Standard](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml/badge.svg)](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A monorepo of engineering and scientific desktop applications, shared Python
libraries, and command-line utilities. It covers process and mechanical
calculators, signal and data processing, robotics model generation, motion
analysis, and project file management. Every graphical tool is reachable from a
single PyQt6 launcher; the underlying libraries are importable on their own and
are shared with other D-sorganization repositories.

- **Audience**: engineers and researchers who want working desktop tools, and
  developers who want the libraries behind them.
- **Platform**: Windows, macOS, and Linux. Python 3.11 or 3.12 (Version **3.11+** required).
- **Status**: actively developed. Interfaces outside `src/shared/` may change
  between releases.

## Contents

| Section                                                 | Purpose                                        |
| ------------------------------------------------------- | ---------------------------------------------- |
| [Tool catalog](#tool-catalog)                           | Every application in the repository, by domain |
| [Command-line entry points](#command-line-entry-points) | Installed console scripts                      |
| [Installation](#installation)                           | Set up a working environment                   |
| [Running the tools](#running-the-tools)                 | Launch the GUI and individual applications     |
| [Repository layout](#repository-layout)                 | Where code lives and why                       |
| [Documentation](#documentation)                         | Guides, architecture, and reference material   |
| [Contributing](#contributing)                           | Branching, testing, and review requirements    |
| [Security](#security)                                   | Vulnerability reporting                        |
| [License](#license)                                     | MIT terms                                      |

## Tool catalog

Every graphical application below is a launcher tile: `python UnifiedToolsLauncher.py`
reads `tools.json`, and `python launch.py --list` reads the same registry. Both are
generated from the one source of truth, the `GUI_INFO` dict in each tool's
`src/<tool>/gui_registration.py`, by `python scripts/generate_tools_json.py`
(`--check` in docs governance keeps this table, `tools.json` and
`tool_surface_contract.json` in step). Do not edit the table by hand.

<!-- tool-catalog:start -->

| Tool                       | Category             | Surfaces    | Maturity     | What it does                                                                                                                                                                                                                                                                                      | Help                                                   |
| -------------------------- | -------------------- | ----------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `multi_param_analysis`     | Analysis             | PyQt6       | stable       | Run multi-parameter sensitivity analysis with grid evaluation                                                                                                                                                                                                                                     | [docs](src/multi_param_analysis/README.md)             |
| `c3d_viewer`               | Biomechanics         | PyQt6       | stable       | View and analyze C3D motion capture files                                                                                                                                                                                                                                                         | [docs](src/c3d_viewer/README.md)                       |
| `lower_body_model`         | Biomechanics         | PyQt6       | stable       | Simulate and inspect lower-body MuJoCo kinematics and controls                                                                                                                                                                                                                                    | —                                                      |
| `rate_of_closure`          | Biomechanics         | PyQt6 + Web | stable       | Quantifies how a rotating clubhead's impact-point delivery differs from the tracked reference point (COM or geometric center): path deviation, attack-angle change, face rotation during contact, and the launch-monitor geometric-center question, with an animated 3D clubhead and rate sweeps. | [docs](src/rate_of_closure/README.md)                  |
| `data_explorer`            | Data Processing      | PyQt6       | stable       | Interactive workbench for browsing simulation datasets                                                                                                                                                                                                                                            | [docs](src/data_explorer/gui.py)                       |
| `data_processor`           | Data Processing      | PyQt6 + Web | stable       | Signal processing and time-series data analysis tool                                                                                                                                                                                                                                              | [docs](src/data_processing/data_processor/README.md)   |
| `folder_packer_pro`        | Development Tools    | PyQt6       | stable       | Professional Project Archiving and Distribution Tool                                                                                                                                                                                                                                              | [docs](src/folder_packer_pro/README.md)                |
| `folder_tool`              | Development Tools    | PyQt6       | experimental | Directory Management Utility                                                                                                                                                                                                                                                                      | [docs](src/folder_tool/README.md)                      |
| `pdf_renamer`              | Development Tools    | PyQt6       | stable       | Intelligent PDF File Renaming Tool                                                                                                                                                                                                                                                                | [docs](src/document_processing/pdf_renamer/README.md)  |
| `pid_generator`            | Engineering Drafting | PyQt6       | stable       | Generate P&ID drawings from YAML specifications (DXF + SVG output)                                                                                                                                                                                                                                | [docs](src/pid_generator/README.md)                    |
| `p1am_control_system`      | Industrial           | Web         | stable       | HMI Control System for P1AM-100 PLC                                                                                                                                                                                                                                                               | —                                                      |
| `ode_solver`               | Mathematics          | PyQt6 + Web | stable       | Solve systems of ordinary differential equations symbolically                                                                                                                                                                                                                                     | [docs](src/ode_solver/README.md)                       |
| `pendulum_simulator`       | Mathematics          | PyQt6       | experimental | Multi-link pendulum dynamics with parameter sweeps                                                                                                                                                                                                                                                | [docs](src/pendulum_simulator/README.md)               |
| `video_processor`          | Media Processing     | Web         | stable       | Video file format conversion, frame extraction, and media analysis                                                                                                                                                                                                                                | [docs](src/media_processing/video_processor/README.md) |
| `video_analyzer`           | Motion Capture       | PyQt6       | stable       | Video-based motion analysis with pose tracking                                                                                                                                                                                                                                                    | —                                                      |
| `movement_optimizer`       | Optimization         | PyQt6       | stable       | Optimize barbell biomechanics trajectories with Lagrangian dynamics, swingset, and chain models                                                                                                                                                                                                   | [docs](src/movement_optimizer/README.md)               |
| `financial_calculator`     | Process Simulation   | PyQt6 + Web | stable       | Comprehensive financial modeling for plant operations                                                                                                                                                                                                                                             | [docs](src/financial_calculator/README.md)             |
| `pressure_drop_calculator` | Process Simulation   | PyQt6 + Web | stable       | Pipe flow pressure drop analysis with multiple friction methods                                                                                                                                                                                                                                   | [docs](src/pressure_drop_calculator/README.md)         |
| `vessel_drafter`           | Process Simulation   | PyQt6       | stable       | Refractory vessel design with STEP, STL, BREP, and GLTF export                                                                                                                                                                                                                                    | [docs](src/vessel_drafter/README.md)                   |
| `humanoid_builder_gui`     | Robotics             | PyQt6       | stable       | Build parametric humanoid characters with anthropometric calculations                                                                                                                                                                                                                             | [docs](src/humanoid_builder_gui/README.md)             |
| `inertia_calculator`       | Robotics             | PyQt6       | stable       | Calculate and validate inertia tensors for rigid bodies                                                                                                                                                                                                                                           | [docs](src/inertia_calculator/README.md)               |
| `rotation_converter`       | Robotics             | PyQt6 + Web | stable       | Comprehensive rotation and rigid-body transform converter with interactive 3D visualization. Supports quaternions, Euler angles, rotation matrices, axis-angle, SE(3), twists, screw axes, and frame-aware transforms.                                                                            | [docs](src/rotation_converter/README.md)               |
| `urdf_builder_gui`         | Robotics             | PyQt6       | stable       | Generate parametric URDF models for robotics applications                                                                                                                                                                                                                                         | [docs](src/urdf_builder_gui/README.md)                 |
| `function_generator`       | Signal Processing    | PyQt6 + Web | stable       | Generate and visualize various waveforms (sine, square, triangle, etc.)                                                                                                                                                                                                                           | [docs](src/function_generator/README.md)               |
| `signal_processing_studio` | Signal Processing    | PyQt6       | stable       | Unified signal processing: waveform generation, analysis, filtering, curve fitting                                                                                                                                                                                                                | [docs](src/signal_processing_studio/README.md)         |
| `steam_engine_calculator`  | Thermodynamics       | PyQt6 + Web | stable       | Calculate thermodynamic properties of steam/water                                                                                                                                                                                                                                                 | [docs](src/steam_engine_calculator/README.md)          |
| `flow_rate_converter`      | Utilities            | PyQt6       | stable       | Convert between mass, molar, and volumetric flow rate units                                                                                                                                                                                                                                       | [docs](src/flow_rate_converter/README.md)              |

<!-- tool-catalog:end -->

Maturity: `stable` launches headless-importable code with tests; `beta` is
usable but changing; `experimental` means the tool module does not import
cleanly in a headless environment or is not yet wired end to end.

### Library and CLI packages (not launcher tiles)

These packages ship code or a command line but no launcher-registered GUI:

| Package              | What it does                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| `rrt_path_planner`   | Rapidly-exploring random tree motion planning (library + scripts)                                     |
| `project_packer`     | Directory packing and unpacking for transfer and archival (standalone script, `folder_packer_gui.py`) |
| `solar_system_model` | N-body orbital mechanics simulation and visualization (scripts)                                       |
| `shared/python/*`    | Libraries shared across the repository fleet (see `src/shared/python/README.md`)                      |

### Browser-based utilities

Served from `src/web_applications/`: a scientific `calculator`, a
`unit_converter`, and a three-dimensional `urdf_viewer`. Each runs from a static
file server with no Python runtime required, and is deliberately not a launcher
tile (`src/web_applications/gui_registration.py` records `web: False`).

## Command-line entry points

Installing the package with `pip install -e .` provides these console scripts:

| Command               | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| `urdf-gen`            | Generate URDF models from parameter files                    |
| `generate-pid`        | Render P&ID diagrams to DXF or SVG from a YAML specification |
| `video-analyzer`      | Launch the video analysis application                        |
| `rate-of-closure-web` | Serve the rate-of-closure web companion                      |
| `codemap`             | Build a structural map of a source tree                      |
| `codemap-watch`       | Rebuild that map on file change                              |
| `codemap-mcp`         | Expose the map over the Model Context Protocol               |
| `sidekick`            | Shared calculation and assistant backend                     |
| `mypy-autofix`        | Apply mechanical type-annotation repairs                     |

## Installation

### Prerequisites

- **Python**: Version **3.11+** required (3.12 recommended for best performance); Python 3.13 is not yet validated.
- Git with Git LFS installed.
- MATLAB R2020a or later, only for the tools under `matlab/`.
- Node.js, only for the browser-based utilities.

### Steps

```bash
git clone https://github.com/D-sorganization/Tools.git
cd Tools
git lfs install && git lfs pull

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install -e ".[dev]"
```

Optional dependency groups install the extras a given tool needs, for example
`pip install -e ".[gui,process,robotics]"`. The full list is in `pyproject.toml`
under `[project.optional-dependencies]`.

Contributors should also install the pre-commit hooks:

```bash
bash scripts/setup_precommit.sh
```

## Running the tools

The launcher is the canonical entry point for every graphical application:

```bash
python UnifiedToolsLauncher.py
```

It presents the tools grouped by domain, validates each tool path before
launching, and captures output and errors from the child process. Pass
`--verbose` for diagnostic logging when a tool fails to start.

Common development tasks run through the Makefile:

```bash
make help      # List available targets
make check     # Run linters and the test suite
make format    # Apply Ruff formatting
```

### Optional Rust acceleration

Several modules ship optional Rust extensions built with
[maturin](https://www.maturin.rs/). Pre-built wheels are not published today, so
the extensions must be built locally; without them the pure-Python
implementation is used automatically and logs a warning that the slower path is
active.

```bash
pip install maturin
cd rust_core/tools-core && maturin develop --features python
cd rust_core/ai_backend  && maturin develop --features python
```

See [Rust distribution](docs/development/rust_distribution.md) for crate
contents and measured performance differences, and
[AI backend setup](docs/ai_backend_setup.md) for the optional ONNX-based local
embedding feature.

## Repository layout

```text
Tools/
├── UnifiedToolsLauncher.py   Canonical GUI entry point
├── src/
│   ├── <tool_name>/          One directory per application (see catalog above)
│   ├── shared/python/        Libraries shared across the repository fleet
│   ├── web_applications/     Browser-based utilities
│   └── python/src/core/      Legacy plugin manager (tile launcher removed in #4916)
├── matlab/                   MATLAB scientific code and utilities
├── rust_core/                Optional Rust extension crates
├── docs/                     Documentation (see below)
└── tests/                    Test suite for the shared monorepo surface
```

`src/shared/python/` holds code with more than one consumer and is the only part
of the tree treated as a stable interface. Everything under an individual tool
directory belongs to that tool.

## Documentation

**Getting started**

- [Quick start](docs/quickstart.md) — first run and basic usage.
- [User manual](docs/USER_MANUAL.md) — per-tool operating instructions.
- [Launcher guide](docs/architecture/LAUNCHERS.md) — entry points and selection logic.

**Architecture and development**

- [Architecture overview](docs/ARCHITECTURE_OVERVIEW.md) — system structure and boundaries.
- [Canonical topology](docs/architecture/CANONICAL_TOPOLOGY.md) — directory policy.
- [Fleet architecture](docs/architecture/FLEET_ARCHITECTURE.md) — shared code across repositories.
- [Plugin system](docs/architecture/PLUGIN_SYSTEM.md) — automatic tool discovery.
- [Build a tool](docs/BUILD_A_TOOL.md) — adding a new application.
- [Development guidelines](docs/development/GUARDRAILS_GUIDELINES.md) — coding standards and guardrails.
- [Branching workflow](docs/development/BRANCHING_WORKFLOW_RULE.md) — required branch and PR process.

**Reference**

- [Tools inventory](docs/tools/TOOLS_INVENTORY.md) — per-tool platform coverage.
- [Enhanced tools](docs/tools/ENHANCED_TOOLS.md) — the extended folder and project tools.
- [Visualization guide](docs/architecture/VISUALIZATION_GUIDE.md) — colorblind-safe plotting standards.
- [Changelog](docs/release/CHANGELOG.md) — release history.

## Contributing

Contributions are welcome. The repository enforces the following:

1. **Branching** — work on a feature branch. Direct commits to `main` are blocked.
2. **Testing** — new behaviour requires tests. The suite runs on Python 3.11 and 3.12.
3. **Linting** — code must pass the pre-commit checks: Ruff, Black, and Mypy.
4. **Review** — all changes merge through a reviewed pull request.

Continuous integration runs a quality gate (Ruff, Black, Mypy, `pip-audit`) and
the test suite across both supported Python versions. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full process.

If a tool fails to start or a test will not collect, see the
[troubleshooting guide](docs/help/troubleshooting.md) before opening an issue.

## Security

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md), not
through public issues.

## License

Released under the MIT License. See [LICENSE](LICENSE). Individual tool
directories may carry additional notices where third-party code is vendored.
