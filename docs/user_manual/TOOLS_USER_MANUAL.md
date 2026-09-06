# Tools Monorepo — Comprehensive User Manual

**Version:** 1.0.0
**Last Updated:** February 2026
**Repository:** `D-sorganization/Tools`

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [Purpose and Scope](#11-purpose-and-scope)
   - 1.2 [Repository Architecture](#12-repository-architecture)
   - 1.3 [Technology Stack](#13-technology-stack)
   - 1.4 [Getting Started](#14-getting-started)
2. [Core Infrastructure](#2-core-infrastructure)
   - 2.1 [Unified Tools Launcher](#21-unified-tools-launcher)
   - 2.2 [Plugin System](#22-plugin-system)
   - 2.3 [GUI Launcher Framework](#23-gui-launcher-framework)
   - 2.4 [Theme System](#24-theme-system)
   - 2.5 [Shared Constants and Utilities](#25-shared-constants-and-utilities)
3. [Process Engineering Calculators](./03_process_calculators.md)
   - 3.1 [Acid Gas Dewpoint Calculator](./03_process_calculators.md#31-acid-gas-dewpoint-calculator)
   - 3.2 [Baghouse Calculator](./03_process_calculators.md#32-baghouse-calculator)
   - 3.3 [Flare Calculator](./03_process_calculators.md#33-flare-calculator)
   - 3.4 [Scrubber Calculator](./03_process_calculators.md#34-scrubber-calculator)
   - 3.5 [Pressure Drop Calculator](./03_process_calculators.md#35-pressure-drop-calculator)
   - 3.6 [Flow Rate Converter](./03_process_calculators.md#36-flow-rate-converter)
   - 3.7 [Syngas Water Calculator](./03_process_calculators.md#37-syngas-water-calculator)
   - 3.8 [Syngas Compression Calculator](./03_process_calculators.md#38-syngas-compression-calculator)
   - 3.9 [WGS Reactor Calculator](./03_process_calculators.md#39-wgs-reactor-calculator)
   - 3.10 [Thermal Profile Predictor](./03_process_calculators.md#310-thermal-profile-predictor)
   - 3.11 [ODE Solver](./03_process_calculators.md#311-ode-solver)
   - 3.12 [Electrode Advancement Calculator](./03_process_calculators.md#312-electrode-advancement-calculator)
   - 3.13 [Financial Calculator](./03_process_calculators.md#313-financial-calculator)
   - 3.14 [PSA Package](./03_process_calculators.md#314-psa-package)
   - 3.15 [Steam Engine Calculator](./03_process_calculators.md#315-steam-engine-calculator)
   - 3.16 [TRC Vessel Designer](./03_process_calculators.md#316-trc-vessel-designer)
   - 3.17 [Optimizer GUI](./03_process_calculators.md#317-optimizer-gui)
   - 3.18 [Multi-Parameter Analysis](./03_process_calculators.md#318-multi-parameter-analysis)
   - 3.19 [Inertia Calculator](./03_process_calculators.md#319-inertia-calculator)
4. [Signal Processing Toolkit](./04_signal_toolkit.md)
   - 4.1 [Signal Core Classes](./04_signal_toolkit.md#41-signal-core-classes)
   - 4.2 [Signal Generation](./04_signal_toolkit.md#42-signal-generation)
   - 4.3 [Function Fitting](./04_signal_toolkit.md#43-function-fitting)
   - 4.4 [Digital Filters](./04_signal_toolkit.md#44-digital-filters)
   - 4.5 [Calculus Operations](./04_signal_toolkit.md#45-calculus-operations)
   - 4.6 [Series Expansions](./04_signal_toolkit.md#46-series-expansions)
   - 4.7 [Noise Generation](./04_signal_toolkit.md#47-noise-generation)
   - 4.8 [Signal Limits](./04_signal_toolkit.md#48-signal-limits)
   - 4.9 [I/O Operations](./04_signal_toolkit.md#49-io-operations)
5. [Scientific Modeling Tools](./05_scientific_modeling.md)
   - 5.1 [Solar System Model](./05_scientific_modeling.md#51-solar-system-model)
   - 5.2 [RRT Path Planner](./05_scientific_modeling.md#52-rrt-path-planner)
   - 5.3 [Function Generator](./05_scientific_modeling.md#53-function-generator)
6. [Robotics and 3D Tools](./06_robotics_3d.md)
   - 6.1 [C3D Viewer](./06_robotics_3d.md#61-c3d-viewer)
   - 6.2 [Humanoid Builder GUI](./06_robotics_3d.md#62-humanoid-builder-gui)
   - 6.3 [URDF Builder GUI](./06_robotics_3d.md#63-urdf-builder-gui)
7. [Data and Document Processing](./07_data_document_processing.md)
   - 7.1 [Data Processor](./07_data_document_processing.md#71-data-processor)
   - 7.2 [PDF Renamer](./07_data_document_processing.md#72-pdf-renamer)
8. [Web Applications](./08_web_applications.md)
   - 8.1 [Calculator Web App](./08_web_applications.md#81-calculator-web-app)
   - 8.2 [Unit Converter](./08_web_applications.md#82-unit-converter)
   - 8.3 [URDF Viewer](./08_web_applications.md#83-urdf-viewer)
9. [Media Processing](./09_media_processing.md)
   - 9.1 [Video Processor](./09_media_processing.md#91-video-processor)
   - 9.2 [Audio Processor](./09_media_processing.md#92-audio-processor)
10. [Development and Utility Tools](./10_development_tools.md)
    - 10.1 [Folder Tool](./10_development_tools.md#101-folder-tool)
    - 10.2 [Folder Packer Pro](./10_development_tools.md#102-folder-packer-pro)
    - 10.3 [Project Packer](./10_development_tools.md#103-project-packer)
    - 10.4 [Quality Utilities](./10_development_tools.md#104-quality-utilities)
    - 10.5 [Dependency Utilities](./10_development_tools.md#105-dependency-utilities)
    - 10.6 [MATLAB Utilities](./10_development_tools.md#106-matlab-utilities)
    - 10.7 [Verification Tools](./10_development_tools.md#107-verification-tools)
11. [Physical Constants and Unit Conversions](./11_constants_conversions.md)
    - 11.1 [Fundamental Physical Constants](./11_constants_conversions.md#111-fundamental-physical-constants)
    - 11.2 [Standard Conditions](./11_constants_conversions.md#112-standard-conditions)
    - 11.3 [Molecular Weights](./11_constants_conversions.md#113-molecular-weights)
    - 11.4 [Conversion Functions](./11_constants_conversions.md#114-conversion-functions)
12. [Implementation Status and Gaps](./12_implementation_gaps.md)
    - 12.1 [Fully Implemented Tools](./12_implementation_gaps.md#121-fully-implemented-tools)
    - 12.2 [Partially Implemented Tools](./12_implementation_gaps.md#122-partially-implemented-tools)
    - 12.3 [Stub/Placeholder Implementations](./12_implementation_gaps.md#123-stubplaceholder-implementations)
    - 12.4 [Recommended Development Priorities](./12_implementation_gaps.md#124-recommended-development-priorities)
13. [Appendices](./13_appendices.md)
    - A. [Mathematical Reference](./13_appendices.md#appendix-a-mathematical-reference)
    - B. [API Quick Reference](./13_appendices.md#appendix-b-api-quick-reference)
    - C. [Configuration Reference](./13_appendices.md#appendix-c-configuration-reference)
    - D. [Glossary](./13_appendices.md#appendix-d-glossary)

---

## 1. Introduction

### 1.1 Purpose and Scope

The **Tools Monorepo** is a comprehensive collection of engineering, scientific, and utility tools designed for process engineering analysis, signal processing, scientific modeling, robotics, and project automation. The repository is architected as a monorepo with a plugin-based launcher system, shared libraries, and both desktop (PyQt6) and web interfaces.

This manual documents every tool in the repository, including:

- **Mathematical foundations** with LaTeX equations for all calculation engines
- **Input/output specifications** for each tool
- **Implementation status** (fully implemented, partial, or stub)
- **Usage examples** and GUI descriptions
- **Integration guidance** for incorporating tools into external projects

### 1.2 Repository Architecture

The repository follows a layered architecture:

```
Tools/
├── UnifiedToolsLauncher.py      # Primary entry point (PyQt6 GUI)
├── src/                          # All tool source code
│   ├── shared/python/            # Shared libraries
│   │   ├── upstream_drift_tools/ # Process calculator engines
│   │   ├── signal_toolkit/       # Signal processing library
│   │   ├── gui_launcher/         # GUI framework
│   │   ├── theme/                # Theming system
│   │   ├── plot_theme/           # Matplotlib plot themes
│   │   ├── model_generation/     # 3D model generation
│   │   └── humanoid_character_builder/  # Humanoid model builder
│   ├── acid_gas_dewpoint/        # Tool: Acid gas dewpoint calculator
│   ├── baghouse_calculator/      # Tool: Baghouse filter calculator
│   ├── flare_calculator/         # Tool: Flare system designer
│   ├── scrubber_calculator/      # Tool: Packed bed scrubber calculator
│   ├── pressure_drop_calculator/ # Tool: Pipe pressure drop calculator
│   ├── flow_rate_converter/      # Tool: Gas flow rate converter
│   ├── syngas_water_calculator/  # Tool: Syngas water content
│   ├── syngas_compression/       # Tool: Syngas compression analysis
│   ├── wgs_reactor/              # Tool: Water-gas shift reactor
│   ├── thermal_profile_predictor/# Tool: Thermal profile predictor
│   ├── ode_solver/               # Tool: ODE solver
│   ├── electrode_advisor/        # Tool: Electrode advancement calculator
│   ├── financial_calculator/     # Tool: Financial model calculator
│   ├── psa_package/              # Tool: Pressure swing adsorption
│   ├── steam_engine_calculator/  # Tool: Steam engine calculator
│   ├── trc_vessel_designer/      # Tool: TRC vessel designer
│   ├── optimizer_gui/            # Legacy launcher shim → movement_optimizer
│   ├── multi_param_analysis/     # Tool: Multi-parameter analysis
│   ├── inertia_calculator/       # Tool: Inertia calculator
│   ├── function_generator/       # Tool: Function generator
│   ├── c3d_viewer/               # Tool: C3D file viewer
│   ├── humanoid_builder_gui/     # Tool: Humanoid model builder
│   ├── urdf_builder_gui/         # Tool: URDF robot builder
│   ├── data_processing/          # Category: Data processing tools
│   ├── document_processing/      # Category: Document processing tools
│   ├── media_processing/         # Category: Media processing tools
│   ├── scientific_modeling/      # Category: Scientific modeling tools
│   ├── web_applications/         # Category: Web applications
│   ├── tools/                    # Utility tools
│   └── verification/             # Verification scripts
├── docs/                         # Documentation
├── python/                       # Core Python infrastructure
└── scripts/                      # Build and automation scripts
```

**Key Design Patterns:**

- **Dual Interface:** Each tool typically provides both a PyQt6 desktop GUI and a web (Flask/Streamlit) interface
- **Plugin Registration:** Tools register via `gui_registration.py` files for automatic discovery
- **Shared Engines:** Core calculation logic lives in `src/shared/python/upstream_drift_tools/process_calculators/`, making engines reusable across GUI, web, and CLI contexts
- **Graceful Degradation:** All tools handle missing optional dependencies (PyQt6, CoolProp, thermo) with fallbacks

### 1.3 Technology Stack

| Component            | Technology                             |
| -------------------- | -------------------------------------- |
| Desktop GUI          | PyQt6 >= 6.6.0                         |
| Web Interfaces       | Flask, Streamlit                       |
| Scientific Computing | NumPy, SciPy, SymPy                    |
| Plotting             | Matplotlib (with custom themes)        |
| Thermodynamics       | CoolProp (optional), thermo (optional) |
| 3D/Robotics          | Open3D, trimesh, URDF                  |
| Testing              | pytest >= 8.2.0                        |
| Linting              | Ruff, Black, MyPy                      |
| CI/CD                | GitHub Actions                         |
| Language             | Python 3.10+ (3.12 recommended)        |

### 1.4 Getting Started

```bash
# Clone and setup
git clone <repository-url>
cd Tools
pip install -r requirements.txt

# Launch the unified GUI
python UnifiedToolsLauncher.py

# Or run individual tools
python src/acid_gas_dewpoint/launch_pyqt6.py
python src/flare_calculator/launch_web.py
```

---

## 2. Core Infrastructure

### 2.1 Unified Tools Launcher

The `UnifiedToolsLauncher.py` is the primary entry point — a PyQt6-based tile launcher that discovers and presents all registered tools.

**Features:**

- Automatic tool discovery via plugin system
- Tool path validation and sanitization
- Output/error capture for launched tools
- Debug mode for troubleshooting
- Activity log for monitoring tool launches

### 2.2 Plugin System

Tools are registered through `gui_registration.py` manifest files located in each tool directory. The plugin manager scans these files to build the launcher's tool catalog.

**Registration Pattern:**

```python
# src/<tool_name>/gui_registration.py
TOOL_INFO = {
    "name": "Tool Display Name",
    "description": "Brief description",
    "category": "Process Engineering",
    "launch_file": "launch_pyqt6.py",
    "icon": "icon.png",
}
```

**20 registered tools** are currently discoverable through the launcher system.

### 2.3 GUI Launcher Framework

Located in `src/shared/python/gui_launcher/`, this framework provides:

- `launcher.py`: Base launcher with process management
- `registry.py`: Tool registration and discovery
- Common launch patterns for PyQt6 and web tools

### 2.4 Theme System

Located in `src/shared/python/theme/`:

- `colors.py`: Colorblind-safe color palettes (WCAG 2.1 AA compliant)
- `integration.py`: Theme application to PyQt6 widgets
- `plot_theme/`: Matplotlib style sheets for consistent visualization

### 2.5 Shared Constants and Utilities

**`upstream_drift_tools/process_calculators/constants.py`** provides NIST-standard physical constants:

| Constant               | Symbol   | Value                        | Unit                               |
| ---------------------- | -------- | ---------------------------- | ---------------------------------- |
| Universal Gas Constant | $R$      | $8.314462618$                | $\text{J/(mol·K)}$                 |
| Standard Gravity       | $g$      | $9.80665$                    | $\text{m/s}^2$                     |
| Avogadro Number        | $N_A$    | $6.02214076 \times 10^{23}$  | $\text{mol}^{-1}$                  |
| Boltzmann Constant     | $k_B$    | $1.380649 \times 10^{-23}$   | $\text{J/K}$                       |
| Stefan-Boltzmann       | $\sigma$ | $5.670374419 \times 10^{-8}$ | $\text{W/(m}^2\text{·K}^4\text{)}$ |

---

_Detailed tool documentation continues in the following chapters. Each chapter can be accessed as a standalone document for integration with external projects._

**Next:** [Chapter 3 — Process Engineering Calculators →](./03_process_calculators.md)
