# Tools

[![CI Standard](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml/badge.svg)](https://github.com/D-sorganization/Tools/actions/workflows/ci-standard.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
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
- **Platform**: Windows, macOS, and Linux. Python 3.11 or 3.12.
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

All graphical applications below launch from `python UnifiedToolsLauncher.py`.
The launcher groups them by the same categories used here.

### Process and mechanical engineering

| Tool                       | What it does                                                             |
| -------------------------- | ------------------------------------------------------------------------ |
| `pressure_drop_calculator` | Single- and two-phase pipe pressure drop analysis with fitting losses    |
| `flow_rate_converter`      | Conversion between mass, volumetric, and standard-condition flow units   |
| `steam_engine_calculator`  | Steam cycle and reciprocating engine performance calculations            |
| `inertia_calculator`       | Inertia tensors and mass properties for composite rigid bodies           |
| `vessel_drafter`           | Pressure vessel geometry and drafting output                             |
| `pid_generator`            | P&ID diagram generation from YAML specifications, via `programmatic_pid` |
| `financial_calculator`     | Project economics: cash flow, payback, and rate-of-return analysis       |
| `p1am_control_system`      | P1AM programmable controller configuration and SCADA support tooling     |

### Signal, data, and documents

| Tool                       | What it does                                                              |
| -------------------------- | ------------------------------------------------------------------------- |
| `signal_processing_studio` | Filtering, spectral analysis, and transform workbench                     |
| `function_generator`       | Waveform and test-signal generation on the shared `signal_toolkit` engine |
| `data_processing`          | Time-series import, cleaning, and analysis pipeline                       |
| `data_explorer`            | Interactive workbench for browsing simulation datasets                    |
| `document_processing`      | PDF metadata extraction and rule-based renaming                           |
| `media_processing`         | Audio and video processing utilities                                      |
| `video_analyzer`           | Frame-accurate video review with golf swing analysis overlays             |

### Robotics and biomechanics

| Tool                   | What it does                                                          |
| ---------------------- | --------------------------------------------------------------------- |
| `urdf_builder_gui`     | Parametric URDF authoring for arbitrary robot chains                  |
| `humanoid_builder_gui` | Anthropometric humanoid generation with URDF export                   |
| `movement_optimizer`   | Sagittal-plane trajectory optimization by Lagrangian inverse dynamics |
| `lower_body_model`     | Lower-limb multibody model and analysis API                           |
| `c3d_viewer`           | C3D motion-capture file inspection and playback                       |
| `rate_of_closure`      | Six-degree-of-freedom clubhead impact delivery analysis               |
| `rrt_path_planner`     | Rapidly-exploring random tree motion planning                         |

### Simulation and numerical work

| Tool                   | What it does                                                        |
| ---------------------- | ------------------------------------------------------------------- |
| `ode_solver`           | Symbolic and numeric solution of ODE systems                        |
| `pendulum_simulator`   | Multi-link pendulum dynamics with parameter sweeps                  |
| `optimizer_gui`        | General-purpose numerical optimization front end                    |
| `multi_param_analysis` | Multi-parameter sweeps and sensitivity studies                      |
| `solar_system_model`   | N-body orbital mechanics simulation and visualization               |
| `rotation_converter`   | Conversion between Euler angles, quaternions, and rotation matrices |

### Project and file management

| Tool                | What it does                                                           |
| ------------------- | ---------------------------------------------------------------------- |
| `folder_tool`       | Bulk file combining, organization, deduplication, and archiving        |
| `folder_packer_pro` | Project packing with AES-256 encryption and syntax-highlighted preview |
| `project_packer`    | Directory packing and unpacking for transfer and archival              |

### Browser-based utilities

Served from `src/web_applications/`: a scientific `calculator`, a
`unit_converter`, and a three-dimensional `urdf_viewer`. Each runs from a static
file server with no Python runtime required.

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

- Python 3.11 or 3.12. Python 3.13 is not yet validated.
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
│   └── python/src/core/      Plugin system and launcher infrastructure
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
