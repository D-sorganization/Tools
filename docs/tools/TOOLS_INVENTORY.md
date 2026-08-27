# Application Inventory and Platform Coverage

Every application in this repository, with the user interfaces each one
provides. The catalog in the [repository README](../../README.md#tool-catalog)
groups the same set by domain and describes what each tool does; this document
records interface coverage and identifies gaps.

## Interface surfaces

| Surface   | Stack                      | Role                         |
| --------- | -------------------------- | ---------------------------- |
| **PyQt6** | Python, PyQt6, Matplotlib  | Desktop GUI, primary surface |
| **Web**   | HTML, CSS, JavaScript      | Browser interface            |
| **Tauri** | Rust wrapper over a web UI | Packaged desktop application |

"Yes" means an implementation is present in the tree. It does not assert feature
parity between surfaces.

## Process and mechanical engineering

| Tool                       | PyQt6 | Web | Tauri | Domain                   |
| -------------------------- | :---: | :-: | :---: | ------------------------ |
| `pressure_drop_calculator` |  Yes  | Yes |  No   | Fluid mechanics          |
| `flow_rate_converter`      |  Yes  | Yes |  No   | Unit conversion          |
| `steam_engine_calculator`  |  Yes  | Yes |  No   | Thermodynamics           |
| `inertia_calculator`       |  Yes  | No  |  No   | Mechanical properties    |
| `vessel_drafter`           |  Yes  | No  |  No   | Pressure vessel drafting |
| `pid_generator`            |  Yes  | No  |  No   | P&ID generation          |
| `financial_calculator`     |  Yes  | Yes |  No   | Project economics        |
| `p1am_control_system`      |  Yes  | Yes |  No   | Industrial control       |

## Signal, data, and documents

| Tool                       | PyQt6 | Web | Tauri | Domain                    |
| -------------------------- | :---: | :-: | :---: | ------------------------- |
| `signal_processing_studio` |  Yes  | No  |  No   | Signal processing         |
| `function_generator`       |  Yes  | Yes |  Yes  | Waveform generation       |
| `data_processing`          |  Yes  | Yes |  Yes  | Time-series analysis      |
| `data_explorer`            |  Yes  | No  |  No   | Dataset browsing          |
| `document_processing`      |  Yes  | No  |  No   | PDF metadata and renaming |
| `media_processing`         |  Yes  | Yes |  No   | Audio and video           |
| `video_analyzer`           |  Yes  | No  |  No   | Video review and overlays |

## Robotics and biomechanics

| Tool                   | PyQt6 | Web | Tauri | Domain                    |
| ---------------------- | :---: | :-: | :---: | ------------------------- |
| `urdf_builder_gui`     |  Yes  | No  |  No   | URDF authoring            |
| `humanoid_builder_gui` |  Yes  | No  |  No   | Anthropometric modeling   |
| `movement_optimizer`   |  Yes  | No  |  No   | Trajectory optimization   |
| `lower_body_model`     |  Yes  | No  |  No   | Multibody limb model      |
| `c3d_viewer`           |  Yes  | No  |  No   | Motion capture inspection |
| `rate_of_closure`      |  Yes  | Yes |  No   | Impact delivery analysis  |
| `rrt_path_planner`     |  No   | No  |  No   | Motion planning, API only |

## Simulation and numerical work

| Tool                   | PyQt6 | Web | Tauri | Domain                      |
| ---------------------- | :---: | :-: | :---: | --------------------------- |
| `ode_solver`           |  Yes  | Yes |  No   | Differential equations      |
| `pendulum_simulator`   |  Yes  | Yes |  Yes  | Multibody dynamics          |
| `optimizer_gui`        |  Yes  | No  |  No   | Numerical optimization      |
| `multi_param_analysis` |  Yes  | No  |  No   | Parameter sweeps            |
| `solar_system_model`   |  No   | No  |  No   | Orbital mechanics, API only |
| `rotation_converter`   |  Yes  | Yes |  Yes  | Spatial rotation math       |
| `asteroid_jumper`      |  Yes  | No  |  No   | Simulation demonstration    |

## Project and file management

| Tool                | PyQt6 | Web | Tauri | Domain                    |
| ------------------- | :---: | :-: | :---: | ------------------------- |
| `folder_tool`       |  Yes  | No  |  No   | Bulk file operations      |
| `folder_packer_pro` |  Yes  | No  |  No   | Encrypted project packing |
| `project_packer`    |  Yes  | No  |  No   | Directory packing         |

## Browser-only utilities

Served as static files from `src/web_applications/`.

| Application      | Purpose                          |
| ---------------- | -------------------------------- |
| `calculator`     | Scientific calculator            |
| `unit_converter` | General unit conversion          |
| `urdf_viewer`    | Three-dimensional URDF rendering |

## Libraries, not standalone applications

| Module                           | Purpose                                            |
| -------------------------------- | -------------------------------------------------- |
| `shared/python/signal_toolkit`   | Signal generation, filtering, series, and calculus |
| `shared/python/safe_eval`        | Constrained expression evaluation                  |
| `shared/python/codemap`          | Source-tree structural mapping and MCP server      |
| `shared/python/programmatic_pid` | P&ID rendering engine behind `pid_generator`       |
| `shared/python/model_generation` | URDF generation engine behind the builder GUIs     |
| `sidekick`                       | Shared calculation and assistant backend           |
| `matlab/`                        | MATLAB scientific code and quality utilities       |
| `verification/`                  | Verification and audit scripts                     |

## Known gaps

**Experimental and incomplete**

- `plant_simulator` — neural-network plant simulator. Not wired into any
  launcher or tool manifest.
- `folder_tool_pro` — scaffold only, no implementation.
- `rotation_converter` — the pure-Python module is deprecated in favour of the
  `math-primitives` Rust crate. The GUI remains supported.

**No web surface**

Fifteen desktop applications have no browser implementation:
`inertia_calculator`, `vessel_drafter`, `pid_generator`,
`signal_processing_studio`, `data_explorer`, `document_processing`,
`video_analyzer`, `urdf_builder_gui`, `humanoid_builder_gui`,
`movement_optimizer`, `lower_body_model`, `c3d_viewer`, `optimizer_gui`,
`multi_param_analysis`, and `asteroid_jumper`, plus the three file-management
tools, which are inherently local.

**No graphical surface**

`rrt_path_planner` and `solar_system_model` are importable libraries with
example scripts but no maintained GUI.

## Maintaining this document

This inventory is derived from the directories present under `src/`. When adding
or removing a tool, update this table and the catalog in the repository README
in the same change. Process calculators that were previously listed here and are
no longer present were relocated to the private companion repository and are not
part of the public distribution.
