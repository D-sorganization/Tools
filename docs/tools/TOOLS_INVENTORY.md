# Tools Repository — Application Inventory & Platform Parity

> **Last Updated**: 2026-03-13
> **Repository**: D-sorganization/Tools (AffineDrift)

## Overview

This repository contains a suite of engineering, scientific, and utility
applications. Each tool targets one or more UI surfaces:

| Surface   | Stack                       | Description                   |
| --------- | --------------------------- | ----------------------------- |
| **PyQt6** | Python + PyQt6 + Matplotlib | Desktop GUI (primary surface) |
| **Web**   | HTML/JS/CSS (plain or Vite) | Browser-based UI              |
| **Tauri** | Rust + Web + Tauri          | Desktop app from web frontend |

---

## Application Inventory

### Engineering Calculators

| #   | Tool                        | PyQt6 | Web | Tauri | Category                     |
| --- | --------------------------- | :---: | :-: | :---: | ---------------------------- |
| 1   | `acid_gas_dewpoint`         |  ✅   | ✅  |  ❌   | Chemical Engineering         |
| 2   | `baghouse_calculator`       |  ✅   | ✅  |  ❌   | Environmental Engineering    |
| 3   | `electrode_advisor`         |  ✅   | ✅  |  ❌   | Welding Engineering          |
| 4   | `financial_calculator`      |  ✅   | ✅  |  ❌   | Financial Analysis           |
| 5   | `flare_calculator`          |  ✅   | ✅  |  ❌   | Process Safety               |
| 6   | `flow_rate_converter`       |  ✅   | ✅  |  ❌   | Unit Conversion              |
| 7   | `pressure_drop_calculator`  |  ✅   | ✅  |  ❌   | Fluid Mechanics              |
| 8   | `scrubber_calculator`       |  ✅   | ✅  |  ❌   | Environmental Engineering    |
| 9   | `steam_engine_calculator`   |  ✅   | ✅  |  ❌   | Thermodynamics               |
| 10  | `syngas_compression`        |  ✅   | ✅  |  ❌   | Chemical Engineering         |
| 11  | `syngas_water_calculator`   |  ✅   | ✅  |  ❌   | Chemical Engineering         |
| 12  | `thermal_profile_predictor` |  ✅   | ✅  |  ❌   | Heat Transfer                |
| 13  | `trc_vessel_designer`       |  ✅   | ✅  |  ❌   | Pressure Vessel Design       |
| 14  | `wgs_reactor`               |  ✅   | ✅  |  ❌   | Chemical Engineering         |
| 15  | `inertia_calculator`        |  ✅   | ❌  |  ❌   | Mechanical Engineering       |
| 16  | `glass_bath_fea`            |  ✅   | ❌  |  ❌   | Finite Element Analysis      |
| 17  | `multi_param_analysis`      |  ✅   | ❌  |  ❌   | Multi-Parameter Optimization |
| 18  | `psa_package`               |  ✅   | ❌  |  ❌   | Pressure Swing Adsorption    |

### Scientific & Simulation Tools

| #   | Tool                       | PyQt6 | Web | Tauri | Category                  |
| --- | -------------------------- | :---: | :-: | :---: | ------------------------- |
| 19  | `ode_solver`               |  ✅   | ✅  |  ❌   | Differential Equations    |
| 20  | `pendulum_simulator`       |  ✅   | ✅  |  ✅   | Physics Simulation        |
| 21  | `function_generator`       |  ✅   | ✅  |  ✅   | Signal Generation         |
| 22  | `rotation_converter`       |  ✅   | ✅  |  ✅   | Robotics / Spatial Math   |
| 23  | `signal_processing_studio` |  ✅   | ❌  |  ❌   | Signal Processing         |
| 24  | `optimizer_gui`            |  ✅   | ❌  |  ❌   | Mathematical Optimization |
| 25  | `gasification_equilibrium` |  ✅   | ❌  |  ❌   | Chemical Equilibrium      |
| 26  | `pid_generator`            |  ✅   | ❌  |  ❌   | P&ID Diagram Generation   |

### Biomechanics & Robotics

| #   | Tool                   | PyQt6 | Web | Tauri | Category               |
| --- | ---------------------- | :---: | :-: | :---: | ---------------------- |
| 27  | `humanoid_builder_gui` |  ✅   | ❌  |  ❌   | URDF Humanoid Building |
| 28  | `urdf_builder_gui`     |  ✅   | ❌  |  ❌   | URDF Robot Building    |
| 29  | `c3d_viewer`           |  ✅   | ❌  |  ❌   | Motion Capture Viewer  |
| 30  | `vessel_drafter`       |  ✅   | ❌  |  ❌   | Vessel CAD Drafting    |

### File & Data Processing

| #   | Tool                | PyQt6 | Web | Tauri | Category                    |
| --- | ------------------- | :---: | :-: | :---: | --------------------------- |
| 31  | `folder_tool`       |  ✅   | ❌  |  ❌   | Directory Management        |
| 32  | `folder_packer_pro` |  ✅   | ❌  |  ❌   | Project Archiving (AES-256) |
| 33  | `project_packer`    |  ✅   | ❌  |  ❌   | Folder Packing/Unpacking    |

### Web-Only Utilities

| #   | Tool                              | PyQt6 | Web | Tauri | Category              |
| --- | --------------------------------- | :---: | :-: | :---: | --------------------- |
| 34  | `web_applications/calculator`     |  ❌   | ✅  |  ❌   | Scientific Calculator |
| 35  | `web_applications/unit_converter` |  ❌   | ✅  |  ❌   | Unit Conversion       |
| 36  | `web_applications/urdf_viewer`    |  ❌   | ✅  |  ❌   | URDF 3D Viewer        |

### Libraries & Processing Modules (not standalone apps)

| #   | Module                         | Description                                    |
| --- | ------------------------------ | ---------------------------------------------- |
| 37  | `shared/python/signal_toolkit` | Signal generation, filtering, series, calculus |
| 38  | `shared/python/safe_eval`      | Safe expression evaluation                     |
| 39  | `data_processing`              | Data import/export pipeline                    |
| 40  | `document_processing`          | PDF renaming & metadata extraction             |
| 41  | `media_processing`             | Media file processing                          |
| 42  | `scientific_modeling`          | Scientific modeling utilities                  |

### Infrastructure & Build

| #   | Module                        | Description                                          |
| --- | ----------------------------- | ---------------------------------------------------- |
| 43  | `tools/` (launcher utilities) | config_loader, launch_utils, quality_utils, ui_utils |
| 44  | `matlab/`                     | MATLAB integration scripts                           |
| 45  | `python/`                     | Shared Python utilities                              |
| 46  | `hcl_reactor/`                | Notebook-only (no GUI)                               |
| 47  | `dwsim_model/`                | DWSIM integration (external model)                   |
| 48  | `verification/`               | Verification scripts                                 |
| 49  | `folder_tool_pro/`            | Empty scaffold (v3.0 shell — no source)              |

---

## Platform Parity Summary

### PyQt6 → Web Gaps (PyQt6 apps missing web surface)

The following **12 applications** have a PyQt6 GUI but **no web implementation**:

1. `inertia_calculator` — Mechanical Engineering
2. `glass_bath_fea` — Finite Element Analysis
3. `multi_param_analysis` — Multi-Parameter Optimization
4. `psa_package` — Pressure Swing Adsorption
5. `signal_processing_studio` — Signal Processing
6. `optimizer_gui` — Mathematical Optimization
7. `gasification_equilibrium` — Chemical Equilibrium
8. `pid_generator` — P&ID Diagram Generation
9. `humanoid_builder_gui` — URDF Humanoid Building
10. `urdf_builder_gui` — URDF Robot Building
11. `c3d_viewer` — Motion Capture Viewer
12. `vessel_drafter` — Vessel CAD Drafting

### PyQt6 → Web Gaps (File Processing Tools)

The following **3 file processing tools** have PyQt6 but no web surface:

13. `folder_tool` — Directory Management
14. `folder_packer_pro` — Project Archiving
15. `project_packer` — Folder Packing

### Web → Tauri Gaps

The following **14 applications** have a web frontend but **no Tauri desktop wrapper**:

1. `acid_gas_dewpoint`
2. `baghouse_calculator`
3. `electrode_advisor`
4. `financial_calculator`
5. `flare_calculator`
6. `flow_rate_converter`
7. `ode_solver`
8. `pressure_drop_calculator`
9. `scrubber_calculator`
10. `steam_engine_calculator`
11. `syngas_compression`
12. `syngas_water_calculator`
13. `thermal_profile_predictor`
14. `trc_vessel_designer`
15. `wgs_reactor`

### Web → PyQt6 Gaps

The following **3 web-only utilities** have no PyQt6 desktop version:

1. `web_applications/calculator`
2. `web_applications/unit_converter`
3. `web_applications/urdf_viewer`

---

## File Organization

```
src/
├── acid_gas_dewpoint/        # Chemical Eng calculator
├── asteroid_jumper/          # Game (standalone)
├── baghouse_calculator/      # Environmental Eng calculator
├── c3d_viewer/               # Motion capture viewer
├── data_processing/          # Data processing library
├── document_processing/      # PDF processing library
├── dwsim_model/              # DWSIM integration
├── electrode_advisor/        # Welding advisor
├── financial_calculator/     # Financial calculator
├── flare_calculator/         # Flare system calculator
├── flow_rate_converter/      # Flow rate converter
├── folder_packer_pro/        # Project archiving (AES-256)  ← RELOCATED
├── folder_tool/              # Directory management utility  ← RELOCATED
├── folder_tool_pro/          # Empty v3.0 scaffold           ← RELOCATED
├── function_generator/       # Signal/function generator
├── gasification_equilibrium/ # Chemical equilibrium
├── glass_bath_fea/           # FEA analysis
├── hcl_reactor/              # HCl reactor (notebook)
├── humanoid_builder_gui/     # URDF humanoid builder
├── inertia_calculator/       # Inertia calculator
├── matlab/                   # MATLAB scripts
├── media_processing/         # Media processing
├── multi_param_analysis/     # Multi-parameter optimization
├── ode_solver/               # ODE solver
├── optimizer_gui/            # Mathematical optimizer
├── pendulum_simulator/       # Pendulum simulator
├── pid_generator/            # P&ID generator
├── pressure_drop_calculator/ # Pressure drop calculator
├── project_packer/           # Folder packing/unpacking      ← RELOCATED
├── psa_package/              # PSA calculator
├── python/                   # Shared Python utilities
├── rotation_converter/       # Rotation converter
├── scientific_modeling/      # Scientific modeling
├── scrubber_calculator/      # Scrubber calculator
├── shared/                   # Shared libraries
├── signal_processing_studio/ # Signal processing
├── steam_engine_calculator/  # Steam engine calculator
├── syngas_compression/       # Syngas compression
├── syngas_water_calculator/  # Syngas water calculator
├── thermal_profile_predictor/# Thermal profile predictor
├── tools/                    # Launcher & quality utilities
├── trc_vessel_designer/      # TRC vessel designer
├── urdf_builder_gui/         # URDF robot builder
├── verification/             # Verification scripts
├── vessel_drafter/           # Vessel drafting
├── web_applications/         # Web-only utilities
└── wgs_reactor/              # WGS reactor
```

---

## Rust Shared Kernel (tools-core)

The `rust_core/` workspace provides the single source of truth for computation,
compiled to both **PyO3 (Python)** and **WASM (web)** via feature gates.

### Rust Crates

| Crate             | Location                                | Modules                                                    | Tests |
| ----------------- | --------------------------------------- | ---------------------------------------------------------- | ----- |
| `math-primitives` | `rust_core/math-primitives/`            | quaternion, rotation, matrix3, geometry, transform, types  | 88    |
| `tools-core`      | `rust_core/tools-core/`                 | math, atmosphere, ball_flight, **signal**, **engineering** | 58    |
| `pendulum-core`   | `src/pendulum_simulator/pendulum-core/` | physics, RK4 solver                                        | 12+   |

### Signal Processing Kernel (`tools-core::signal`)

| Function      | Formula                      | Status         |
| ------------- | ---------------------------- | -------------- |
| `sinusoid`    | y = A·sin(2πft + φ) + offset | ✅ Implemented |
| `cosine`      | y = A·cos(2πft + φ) + offset | ✅ Implemented |
| `exponential` | y = A·exp(-λt) + offset      | ✅ Implemented |
| `linear`      | y = slope·t + intercept      | ✅ Implemented |
| `step`        | Heaviside step               | ✅ Implemented |
| `square`      | Square wave with duty cycle  | ✅ Implemented |
| `triangle`    | Triangle wave                | ✅ Implemented |
| `chirp`       | Linear frequency sweep       | ✅ Implemented |
| `polynomial`  | y = Σ cₙ·tⁿ                  | ✅ Implemented |
| `pulse`       | Rectangular pulse            | ✅ Implemented |

### Engineering Calculation Kernel (`tools-core::engineering`)

| Function                             | Domain          | Status         |
| ------------------------------------ | --------------- | -------------- |
| `reynolds_number`                    | Fluid mechanics | ✅ Implemented |
| `churchill_friction_factor`          | Fluid mechanics | ✅ Implemented |
| `darcy_weisbach_pressure_drop`       | Fluid mechanics | ✅ Implemented |
| `flow_rate_from_velocity`            | Fluid mechanics | ✅ Implemented |
| `ideal_gas_density`                  | Thermodynamics  | ✅ Implemented |
| `compressibility_factor_vdw`         | Thermodynamics  | ✅ Implemented |
| `isentropic_work`                    | Thermodynamics  | ✅ Implemented |
| `convective_heat_transfer`           | Heat transfer   | ✅ Implemented |
| `radiative_heat_transfer`            | Heat transfer   | ✅ Implemented |
| `lmtd`                               | Heat transfer   | ✅ Implemented |
| Unit conversions (C/K/F, bar/psi/Pa) | Units           | ✅ Implemented |

### Tool → Rust Integration Status

| Tool                        | Uses Rust? | Via                                 | Issue |
| --------------------------- | :--------: | ----------------------------------- | ----- |
| `rotation_converter`        |     ✅     | `tools_core.math_primitives` (PyO3) | —     |
| `pendulum_simulator`        |     ✅     | `pendulum_core` (PyO3)              | —     |
| `function_generator`        |     ❌     | Planned: `tools_core.signal`        | #1354 |
| `signal_processing_studio`  |     ❌     | Planned: `tools_core.signal`        | #1354 |
| `pressure_drop_calculator`  |     ❌     | Planned: `tools_core.engineering`   | #1355 |
| `flow_rate_converter`       |     ❌     | Planned: `tools_core.engineering`   | #1355 |
| `thermal_profile_predictor` |     ❌     | Planned: `tools_core.engineering`   | #1355 |
| `syngas_compression`        |     ❌     | Planned: `tools_core.engineering`   | #1355 |
| Web frontends (WASM)        |     ❌     | Planned: `wasm-pack` build          | #1356 |
