# Tools Repository User Manual

**Version 1.0** | **Last Updated: February 2026**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Unified Tools Launcher](#3-unified-tools-launcher)
4. [Media Processing Tools](#4-media-processing-tools)
5. [Data Processing Tools](#5-data-processing-tools)
6. [Signal Processing Tools](#6-signal-processing-tools)
7. [Scientific Modeling Tools](#7-scientific-modeling-tools)
8. [Process Engineering Calculators](#8-process-engineering-calculators)
9. [Financial Tools](#9-financial-tools)
10. [Engineering and Robotics Tools](#10-engineering-and-robotics-tools)
11. [Web Applications](#11-web-applications)
12. [Development Tools](#12-development-tools)
13. [Shared Libraries](#13-shared-libraries)
14. [Configuration](#14-configuration)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Introduction

### 1.1 Overview

The **Tools Monorepo** is a comprehensive collection of utility applications for data processing, scientific computing, process engineering, robotics, and project automation. The repository houses over 50 specialized tools organized into 9 distinct categories, providing professional-grade solutions for engineers, scientists, and developers.

### 1.2 Architecture and Organization

The repository follows a modular architecture with clear separation of concerns:

```
Tools/
├── UnifiedToolsLauncher.py    # Primary GUI launcher
├── src/                        # Major tool categories
│   ├── data_processing/        # Data analysis tools
│   ├── media_processing/       # Audio/video tools
│   ├── scientific_modeling/    # Simulation tools
│   ├── web_applications/       # Browser-based apps
│   ├── shared/                 # Shared libraries
│   └── [tool_name]/            # Individual tool directories
├── python/                     # Core infrastructure
│   └── src/core/               # Plugin system
├── tools/                      # Standalone utilities
└── docs/                       # Documentation
```

### 1.3 Key Features

- **Unified Launcher**: Single entry point to access all 50+ tools
- **Plugin System**: Automatic tool discovery via manifest files
- **Cross-Platform**: Windows and Linux support
- **Modern UI**: PyQt6-based interfaces with Catppuccin Mocha dark theme
- **Shared Libraries**: Centralized calculation engines for consistency
- **Web Interfaces**: Browser-based tools with offline support (PWA)
- **MATLAB Integration**: Scientific computing with MATLAB R2020a+

---

## 2. Installation

### 2.1 Prerequisites

| Requirement | Version                  | Notes                  |
| ----------- | ------------------------ | ---------------------- |
| Python      | 3.11+ (3.12 recommended) | Core runtime           |
| Git         | Latest                   | With LFS support       |
| MATLAB      | R2020a+                  | For MATLAB-based tools |
| Node.js     | 18+                      | For web applications   |

### 2.2 Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd Tools

# Initialize Git LFS
git lfs install
git lfs pull

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### 2.3 Install Dependencies

```bash
# Install Python dependencies
pip install -r python/requirements.txt

# Key packages include:
# - PyQt6>=6.6.0 (GUI framework)
# - numpy, scipy (Scientific computing)
# - matplotlib (Plotting)
# - sympy (Symbolic math)
```

### 2.4 Install Pre-commit Hooks (Developers)

```bash
bash scripts/setup_precommit.sh
```

### 2.5 Using the Makefile

```bash
make help      # Show available targets
make install   # Install all dependencies
make check     # Run linters and tests
make format    # Format code with black and ruff
```

---

## 3. Unified Tools Launcher

### 3.1 Starting the Launcher

The **UnifiedToolsLauncher** is the primary and recommended entry point for accessing all tools:

```bash
python UnifiedToolsLauncher.py
```

### 3.2 Features

- **Tabbed Interface**: Tools organized by category
- **Plugin System**: Automatic discovery of new tools via `tool_manifest.json`
- **Output Capture**: Real-time logging of tool output and errors
- **Debug Mode**: Verbose logging for troubleshooting
- **Tool Validation**: Path verification before launch

### 3.3 Navigating Tool Categories

The launcher organizes tools into the following tabs:

| Tab                 | Description                        |
| ------------------- | ---------------------------------- |
| Process Engineering | 24+ industrial calculators         |
| Scientific Modeling | Simulation and modeling tools      |
| Signal Processing   | Function generators, filters       |
| Data Processing     | Data analysis platforms            |
| Robotics            | URDF builders, inertia calculators |
| Media Processing    | Audio/video tools                  |
| Web Applications    | Browser-based tools                |
| Development         | Folder tools, utilities            |

### 3.4 Plugin System

Tools can be registered automatically by placing a `tool_manifest.json` in their directory:

```json
{
  "name": "My Tool",
  "path": "main.py",
  "type": "python",
  "description": "Tool description",
  "category": "Development Tools"
}
```

Supported types: `python`, `matlab`, `web`, `browser`, `bat`

---

## 4. Media Processing Tools

### 4.1 Audio Processor (MATLAB)

**Location**: `src/media_processing/audio_processor/`

**Purpose**: Professional audio signal processing and multi-track mixing application.

**Features**:

- Multi-format support: WAV, MP3, FLAC, OGG, M4A
- Advanced filtering: FFT-based, Butterworth, custom FIR/IIR
- Audio effects: Reverb, delay, EQ, compression, chorus, pitch shifting
- Multi-track mixing: 8+ tracks with per-track effects chains
- Analysis tools: Spectrogram, FFT analyzer, loudness metering

**Inputs**: Audio files in supported formats

**Outputs**: Processed audio files, analysis visualizations

**Launch**:

```matlab
cd media_processing/audio_processor/matlab/audio_signal_processor
launch_audio_processor_pro
```

**Requirements**: MATLAB R2020b+, Signal Processing Toolbox, Audio Toolbox

**Status**: Fully Implemented

---

### 4.2 Video Processor Platform

**Location**: `src/media_processing/video_processor/`

**Purpose**: AI-powered video analysis platform with golf swing analysis focus.

**Features**:

- Video upload, playback, and annotation
- AI pose detection via MediaPipe
- Drawing and overlay tools
- Audio commentary recording
- 3D visualization with Three.js
- MATLAB physics modeling integration

**Inputs**: Video files, user annotations

**Outputs**: Annotated videos, pose analysis data

**Launch**:

```bash
cd media_processing/video_processor
npm install && npm run dev
```

**Requirements**: Node.js 18+, npm 9+

**Status**: Fully Implemented

---

## 5. Data Processing Tools

### 5.1 Data Processor (PyQt6)

**Location**: `src/data_processing/data_processor/`

**Purpose**: Comprehensive data analysis and processing platform with desktop GUI.

**Features**:

- Data import from multiple formats (CSV, Excel, JSON)
- Statistical analysis and visualization
- Signal processing integration
- Export to various formats

**Inputs**: Tabular data files

**Outputs**: Processed data, statistical reports, visualizations

**Launch**:

```bash
python src/data_processing/data_processor/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 5.2 Data Processor (Web)

**Location**: `src/data_processing/data_processor/web/`

**Purpose**: Browser-based data processing with Tauri desktop support.

**Features**:

- React-based modern UI
- Cross-platform desktop app via Tauri
- Real-time data visualization

**Launch**:

```bash
cd src/data_processing/data_processor/web
npm install && npm run dev
```

**Status**: Fully Implemented

---

## 6. Signal Processing Tools

### 6.1 Function Generator

**Location**: `src/function_generator/`

**Purpose**: Generate and visualize mathematical functions and waveforms.

**Features**:

- Standard waveforms: sine, square, triangle, sawtooth
- Polynomial functions
- Custom function expressions
- Real-time visualization
- Export to data files

**Inputs**: Function parameters, frequency, amplitude

**Outputs**: Waveform data, visualization plots

**Launch (PyQt6)**:

```bash
python src/function_generator/launch_pyqt6.py
```

**Launch (Web)**:

```bash
cd src/function_generator/web
npm install && npm run dev
```

**Status**: Fully Implemented (PyQt6 and Web versions)

---

### 6.2 Polynomial Generator

**Location**: Integrated with Function Generator

**Purpose**: Generate and fit polynomial functions.

**Features**:

- Polynomial coefficient input
- Root finding
- Curve fitting to data points
- Derivative and integral computation

**Status**: Fully Implemented

---

### 6.3 Signal Toolkit Widget

**Location**: `src/shared/python/signal_toolkit/`

**Purpose**: Shared library providing signal processing primitives.

**Features**:

- Digital filtering (lowpass, highpass, bandpass)
- Calculus operations (differentiation, integration)
- Noise generation and analysis
- Curve fitting algorithms
- Limit detection

**Usage**:

```python
from signal_toolkit.filters import apply_lowpass_filter
from signal_toolkit.calculus import differentiate
```

**Status**: Fully Implemented

---

## 7. Scientific Modeling Tools

### 7.1 Solar System Model

**Location**: `src/scientific_modeling/solar_system_model/`

**Purpose**: Interactive 3D visualization of the solar system with accurate orbital mechanics.

**Features**:

- Accurate planetary positions
- Orbital path visualization
- Time controls for simulation
- 3D interactive camera

**Inputs**: Date/time, visualization settings

**Outputs**: 3D visualization, orbital data

**Launch**:

```bash
python scientific_modeling/solar_system_model/run_solar_system.py
```

**Status**: Fully Implemented

---

### 7.2 RRT Path Planner

**Location**: `src/scientific_modeling/rrt_path_planner/`

**Purpose**: Rapidly-exploring Random Trees path planning for robotics applications.

**Features**:

- 3D environment with obstacles
- Dual implementation (MATLAB and Python)
- Star Wars-themed visualization
- AI pursuit system with dynamic replanning
- Cinematic camera views

**Inputs**: Environment configuration, start/goal positions

**Outputs**: Optimal path, visualization

**Launch (MATLAB)**:

```matlab
cd scientific_modeling/rrt_path_planner/matlab/src
main_improved
```

**Launch (Python)**:

```bash
cd scientific_modeling/rrt_path_planner/python/src
python star_wars_rrt.py
```

**Status**: Fully Implemented

---

### 7.3 ODE Solver

**Location**: `src/ode_solver/`

**Purpose**: Solve systems of ordinary differential equations with interactive GUI.

**Features**:

- Preset examples: Exponential decay, harmonic oscillator, Lotka-Volterra
- Custom ODE system definition
- Multiple solver methods
- Solution visualization
- Parameter sweep capability

**Inputs**: ODE system definition, parameters, initial conditions

**Outputs**: Solution curves, numerical data

**Launch**:

```bash
python src/ode_solver/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 7.4 Thermal Profile Predictor

**Location**: `src/thermal_profile_predictor/`

**Purpose**: Predict temperature profiles in heated vessels over time.

**Features**:

- Thermal mass and heat loss modeling
- Power profile options: constant, linear ramp, step
- Temperature vs time prediction
- Condensation risk assessment

**Inputs**: Thermal parameters, power profiles, time range

**Outputs**: Temperature curves, thermal analysis

**Launch**:

```bash
python src/thermal_profile_predictor/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 7.5 Multi-Parameter Analysis

**Location**: `src/multi_param_analysis/`

**Purpose**: Sensitivity analysis across multiple parameter dimensions.

**Features**:

- 2D parameter sweep
- Demo functions: Rosenbrock, Rastrigin, Sphere, Himmelblau
- Variance-based sensitivity indices
- Parallel processing support
- Contour visualization

**Inputs**: Parameter ranges, output variable selection

**Outputs**: Sensitivity analysis, optimal parameters

**Launch**:

```bash
python src/multi_param_analysis/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 7.6 Optimizer GUI (legacy shim)

**Location**: `src/optimizer_gui/`

**Purpose**: Compatibility launcher for the canonical Movement Optimizer
application. The standalone optimizer GUI that used to live here was
consolidated into `src/movement_optimizer` (Tools #3983); the drifted vendored
copy was removed and only the registration/launcher shim remains.

**Launch**:

```bash
python src/optimizer_gui/launch_pyqt6.py
```

The command starts the canonical Movement Optimizer PyQt6 application from
`src/movement_optimizer`.

---

## 8. Process Engineering Calculators

The repository includes 24 specialized process engineering calculators for industrial applications. All calculators feature:

- PyQt6 GUI with Catppuccin Mocha dark theme
- Web interface option (React/Tauri)
- Shared calculation engines from `upstream_drift_tools`

---

### 8.1 Acid Gas Dewpoint Calculator

**Location**: `src/acid_gas_dewpoint/`

**Purpose**: Calculate dewpoints for acid gases (HF, HCl, H2S) in syngas systems.

**Features**:

- Preset compositions: Typical Syngas, Coal Gasification, Biomass
- Calculation methods: Antoine, Extended Antoine
- Safety analysis with condensation risk assessment
- Dewpoint comparison charts

**Inputs**: Temperature, pressure, gas composition (H2O, HF, HCl, H2S)

**Outputs**: Individual and overall dewpoints, safety margins, warnings

**Launch**:

```bash
python src/acid_gas_dewpoint/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.2 Baghouse Calculator

**Location**: `src/baghouse_calculator/`

**Purpose**: Design and performance analysis for baghouse filter systems.

**Inputs**:

- Gas stream: Flow rate (kg/s), inlet temperature, pressure
- Solids: Carbon and ash input rates
- Removal efficiencies: Carbon, ash
- Equipment: Heat loss, drum volume, bag filter area

**Outputs**:

- Solids removal rates
- Drum fill time (hours/days)
- Air-to-cloth ratio
- Outlet temperature

**Launch**:

```bash
python src/baghouse_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.3 Flare Calculator

**Location**: `src/flare_calculator/`

**Purpose**: Size flare systems and determine safety zones.

**Inputs**:

- Total flow rate (kg/hr)
- Gas composition (H2, CO, CH4, CO2, N2, H2O, H2S)
- Temperature (K) and pressure (bar)

**Outputs**:

- Flare dimensions: height, diameter
- Exit velocity, heat release
- Radiation safety zones: lethal, damage, safe, comfort
- Gas mixture properties

**Launch**:

```bash
python src/flare_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.4 Scrubber Calculator

**Location**: `src/scrubber_calculator/`

**Purpose**: Design wet scrubber systems for gas cleaning.

**Inputs**: Gas flow, contaminant concentrations, scrubbing liquid properties

**Outputs**: Removal efficiency, liquid requirements, pressure drop

**Launch**:

```bash
python src/scrubber_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.5 Pressure Drop Calculator

**Location**: `src/pressure_drop_calculator/`

**Purpose**: Calculate pressure drops in piping systems.

**Features**:

- Pipe sizes: 0.5" to 24" nominal
- Schedules: 5, 10, 20, 40, 80, STD, XS, XXS
- Friction methods: Colebrook, Swamee-Jain, Churchill, Haaland
- Materials: Carbon Steel, Stainless, Copper, PVC, HDPE, Concrete

**Inputs**:

- Pipe parameters: size, schedule, length, material, elevation
- Flow conditions: rate, pressure, temperature
- Gas composition (8 components)

**Outputs**:

- Total pressure drop (Pa)
- Friction factor, Reynolds number
- Flow velocity, Mach number
- Erosional velocity warnings

**Launch**:

```bash
python src/pressure_drop_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.6 Syngas Water Calculator

**Location**: `src/syngas_water_calculator/`

**Purpose**: Calculate water content and dew point in syngas systems.

**Inputs**:

- Temperature (C) and pressure (bar)
- Gas composition preset
- Calculation method: Auto, Antoine, Buck, IAPWS-IF97, Magnus

**Outputs**:

- Water content in multiple units (mg/Nm3, ppmv, g/m3, lb/MMscf)
- Vapor pressure, dew point
- Condensation risk assessment
- Recommended minimum temperature

**Launch**:

```bash
python src/syngas_water_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.7 Syngas Compression Calculator

**Location**: `src/syngas_compression/`

**Purpose**: Size and analyze syngas compression systems.

**Inputs**: Inlet conditions, outlet pressure requirements, gas composition

**Outputs**: Compression ratios, power requirements, stage design

**Launch**:

```bash
python src/syngas_compression/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.8 WGS Reactor Calculator

**Location**: `src/wgs_reactor/`

**Purpose**: Water-gas shift reactor design and analysis.

**Inputs**: Feed composition, reaction conditions, catalyst parameters

**Outputs**: Conversion, product composition, heat duty

**Launch**:

```bash
python src/wgs_reactor/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.9 Electrode Advisor

**Location**: `src/electrode_advisor/`

**Purpose**: 3-phase electrical system analysis for electrode heating applications.

**Features**:

- 3-phase electrical measurements input
- Electrode depth configuration
- Physical parameter settings (bath diameter, tip diameter, temperature)
- Uses shared `ThreePhaseElectricalModelEnhanced` engine

**Inputs**:

- Phase currents and voltages (3 phases)
- Electrode depths (3 electrodes)
- Bath geometry and temperature

**Outputs**:

- Total power (kW)
- Phase resistances and powers
- System status indicators

**Launch**:

```bash
python src/electrode_advisor/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.10 TRC Vessel Designer

**Location**: `src/trc_vessel_designer/`

**Purpose**: Design thermal reaction chamber vessels with refractory lining.

**Features**:

- Vessel geometry: cylinder + cone configuration
- Refractory presets: Standard (3-layer), High Temperature (4-layer), Economy (2-layer)
- Operating conditions integration
- Uses shared `TRCGeometryEngine`

**Inputs**:

- Cylinder dimensions (height, diameter)
- Cone parameters (height, bottom diameter)
- Refractory configuration
- Operating temperature, pressure, flow rate

**Outputs**:

- Net internal volume
- Total refractory mass
- Layer-by-layer breakdown
- Residence time calculation
- Outside surface area

**Launch**:

```bash
python src/trc_vessel_designer/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.11 PSA Package

**Location**: `src/psa_package/`

**Purpose**: Pressure Swing Adsorption system design and analysis.

**Inputs**: Feed composition, product purity requirements, cycle parameters

**Outputs**: Bed sizing, cycle timing, product recovery

**Launch**:

```bash
python src/psa_package/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.12 Steam Engine Calculator

**Location**: `src/steam_engine_calculator/`

**Purpose**: Calculate steam thermodynamic properties.

**Features**:

- Calculation modes: T&P, Saturated (from T), Saturated (from P)
- Multiple calculation engines: CoolProp, Cantera, Simplified
- Comprehensive property output

**Inputs**:

- Temperature (K or C)
- Pressure (Pa, kPa, bar, MPa)

**Outputs**:

- Phase state and quality
- Thermodynamic: density, enthalpy, entropy, internal energy, Cp, Cv
- Transport: speed of sound, thermal conductivity, viscosity
- Derived: compressibility factor, Prandtl number, Cp/Cv ratio

**Launch**:

```bash
python src/steam_engine_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.13 Flow Rate Converter

**Location**: `src/flow_rate_converter/`

**Purpose**: Convert between mass, molar, and volumetric flow rate units.

**Features**:

- Mass flow: kg/s, kg/h, kg/min, g/s, g/h, lb/s, lb/h, lb/min, ton/h
- Molar flow: mol/s, mol/h, kmol/s, kmol/h, lbmol/s, lbmol/h
- Volumetric: m3/s, m3/h, L/s, L/min, ft3/s, CFM, GPM

**Inputs**: Source value, from unit, to unit

**Outputs**: Converted value

**Launch**:

```bash
python src/flow_rate_converter/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 8.14-8.24 Additional Process Calculators

The following additional calculators follow the same pattern:

| Calculator                    | Purpose                               |
| ----------------------------- | ------------------------------------- |
| **Heat Exchanger Calculator** | LMTD, effectiveness-NTU calculations  |
| **Pump Sizing Calculator**    | Pump head, power, NPSH calculations   |
| **Tank Volume Calculator**    | Storage tank capacity calculations    |
| **Relief Valve Sizer**        | Safety relief valve sizing            |
| **Cooling Tower Calculator**  | Cooling tower performance             |
| **Distillation Column**       | Stage calculations, reflux ratio      |
| **Reactor Sizing**            | CSTR, PFR, batch reactor design       |
| **Catalyst Bed Calculator**   | Catalyst volume, pressure drop        |
| **Combustion Calculator**     | Air requirements, exhaust composition |
| **Mass Balance Tool**         | Process mass balance calculations     |
| **Energy Balance Tool**       | Process energy balance calculations   |

---

## 9. Financial Tools

### 9.1 Financial Calculator

**Location**: `src/financial_calculator/`

**Purpose**: Financial modeling for industrial plant operations.

**Features**:

- Plant operations: capacity, operating days, utilization
- Revenue modeling with product pricing
- Variable costs: feedstock, labor, utilities, maintenance
- Fixed costs: fixed labor, insurance
- Capital and financing: debt ratio, interest rate, depreciation
- 10-year projections

**Inputs**:

- Plant capacity (TPD), operating days, utilization
- Product price, variable costs per ton
- Fixed annual costs
- Capital investment, financing parameters

**Outputs**:

- Annual feedstock processing
- Total revenue and costs
- Net income, EBITDA
- Return on equity, payback period
- 10-year financial projections table

**Launch**:

```bash
python src/financial_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

## 10. Engineering and Robotics Tools

### 10.1 Inertia Calculator

**Location**: `src/inertia_calculator/`

**Purpose**: Calculate and validate inertia tensors for rigid bodies.

**Features**:

- Primitive shapes: box, cylinder, sphere, capsule
- Custom inertia tensor input
- Physical validity checks
- Principal axis computation
- URDF export compatibility

**Inputs**: Shape parameters, mass, dimensions

**Outputs**: Inertia tensor (Ixx, Iyy, Izz, Ixy, Ixz, Iyz), principal axes

**Launch**:

```bash
python src/inertia_calculator/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 10.2 URDF Builder

**Location**: `src/urdf_builder_gui/`

**Purpose**: Generate parametric URDF models for robotics applications.

**Features**:

- Link and joint definition
- Visual and collision geometry
- Inertia property generation
- Preview and validation
- Export to URDF XML

**Inputs**: Robot structure definition, link parameters

**Outputs**: URDF XML file

**Launch**:

```bash
python src/urdf_builder_gui/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 10.3 Humanoid Builder

**Location**: `src/humanoid_builder_gui/`

**Purpose**: Build parametric humanoid character models with anthropometric calculations.

**Features**:

- Anthropometric presets
- Body segment customization
- Inertia calculation
- URDF export for simulation

**Inputs**: Height, weight, body proportions

**Outputs**: Humanoid URDF model, segment properties

**Launch**:

```bash
python src/humanoid_builder_gui/launch_pyqt6.py
```

**Status**: Fully Implemented

---

### 10.4 C3D Viewer

**Location**: `src/c3d_viewer/`

**Purpose**: View and analyze C3D motion capture files.

**Features**:

- C3D file import
- 3D marker visualization
- Frame-by-frame playback
- Marker trajectory analysis

**Inputs**: C3D motion capture files

**Outputs**: Visualization, marker data export

**Launch**:

```bash
python src/c3d_viewer/launch_pyqt6.py
```

**Status**: Fully Implemented

---

## 11. Web Applications

### 11.1 Aurora CAS Calculator

**Location**: `src/web_applications/calculator/`

**Purpose**: Web-based Computer Algebra System (CAS) calculator.

**Features**:

- Symbolic math: factor, expand, simplify, solve
- Calculus: derivatives, integrals, limits, Taylor series
- Linear algebra: matrix operations, decompositions
- Robotics support: screw theory, SE(3) operations
- Touch-friendly interface with mode-specific soft keys

**Launch**:

```bash
cd src/web_applications/calculator
flask --app webapp run
# Access at http://localhost:5000
```

**Status**: Fully Implemented

---

### 11.2 Unit Converter (PWA)

**Location**: `src/web_applications/unit_converter/`

**Purpose**: NIST-compliant unit converter with offline support.

**Features**:

- 16+ categories: Length, Mass, Volume, Temperature, Pressure, Energy, etc.
- 100+ units with NIST-standard conversion factors
- Offline support (Progressive Web App)
- iOS-optimized interface
- Custom unit support

**Launch**:

```bash
cd src/web_applications/unit_converter/unit-converter-app
python -m http.server 8000
# Access at http://localhost:8000
```

Or open `index.html` directly in a browser.

**Status**: Fully Implemented

---

### 11.3 URDF Viewer

**Location**: `src/web_applications/urdf_viewer/`

**Purpose**: Web-based 3D viewer for URDF robot models.

**Features**:

- Three.js 3D rendering
- URDF file upload
- Interactive model inspection
- FastAPI backend

**Launch**:

```bash
cd src/web_applications/urdf_viewer
uvicorn app:app --reload
# Access at http://localhost:8000
```

**Status**: Fully Implemented

---

## 12. Development Tools

### 12.1 Folder Packer Pro

**Location**: `src/folder_packer_pro/`

**Purpose**: Project archiving and distribution tool.

**Features**:

- Selective file/folder packaging
- Exclusion patterns (node_modules, **pycache**, etc.)
- Multiple output formats
- Integrity verification

**Status**: Fully Implemented

---

### 12.2 PDF Renamer

**Location**: `src/document_processing/`

**Purpose**: Batch rename PDF files based on content or metadata.

**Features**:

- Metadata extraction
- Content-based naming
- Pattern-based renaming
- Preview before apply

**Status**: Fully Implemented

---

## 13. Shared Libraries

### 13.1 Signal Toolkit

**Location**: `src/shared/python/signal_toolkit/`

**Purpose**: Shared signal processing primitives.

**Modules**:

- `core.py`: Base signal operations
- `filters.py`: Digital filter implementations
- `calculus.py`: Differentiation and integration
- `fitting.py`: Curve fitting algorithms
- `noise.py`: Noise generation and analysis
- `limits.py`: Limit detection
- `io.py`: Signal I/O utilities

**Usage**:

```python
from signal_toolkit.filters import apply_lowpass_filter
from signal_toolkit.calculus import differentiate, integrate
```

---

### 13.2 Upstream Drift Tools

**Location**: `src/shared/python/upstream_drift_tools/`

**Purpose**: Centralized calculation engines for process engineering.

**Packages**:

- `calculators/`: Core calculation engines
  - `electrical/`: 3-phase electrical models
  - `mechanical/`: Geometry and stress calculations
  - `thermo/`: Steam and thermodynamic properties
  - `conversion/`: Unit conversion utilities
- `process_calculators/`: Industrial process tools
- `lab/`: Laboratory and biomechanics tools
- `ui/`: Shared UI components

---

### 13.3 GUI Theme System

**Location**: Integrated in each tool

**Theme**: Catppuccin Mocha dark theme

**Colors**:

```python
COLORS = {
    "base": "#1e1e2e",      # Background
    "surface0": "#313244",   # Input fields
    "surface1": "#45475a",   # Borders
    "text": "#cdd6f4",       # Primary text
    "blue": "#89b4fa",       # Accent/buttons
    "green": "#a6e3a1",      # Success
    "red": "#f38ba8",        # Error
    "yellow": "#f9e2af",     # Warning
}
```

---

### 13.4 Plot Theme

**Location**: Shared matplotlib configuration

**Features**:

- Dark background matching GUI theme
- Colorblind-safe palettes
- Consistent styling across all tools

---

## 14. Configuration

### 14.1 Environment Setup

Create a `.env` file in the repository root for custom configuration:

```bash
# Python path additions
PYTHONPATH=src:python

# MATLAB path (if not in system PATH)
MATLAB_PATH=/usr/local/MATLAB/R2023a/bin

# Debug mode
DEBUG=true
```

### 14.2 Tool Manifests

Each tool can define a `tool_manifest.json`:

```json
{
  "name": "Tool Display Name",
  "path": "main.py",
  "type": "python",
  "description": "Tool description",
  "category": "Category Name"
}
```

### 14.3 Plugin Discovery

The plugin system scans these directories for manifests:

- `src/` (all subdirectories)
- `tools/` (all subdirectories)
- `python/` (legacy tools)

---

## 15. Troubleshooting

### 15.1 Common Issues

#### Python Version Errors

**Problem**: `ImportError: cannot import name 'StrEnum' from 'enum'`

**Cause**: Running Python earlier than 3.11, which lacks required runtime features.

**Solution**: Upgrade to Python 3.11 or 3.12 (recommended)

#### Launcher Won't Start

**Problem**: `UnifiedToolsLauncher.py` fails to launch

**Solutions**:

1. Install dependencies: `pip install -r requirements.txt`
2. Check Python version: `python --version` (must be 3.11+)
3. Install PyQt6: `pip install PyQt6>=6.6.0`
4. Run with verbose: `python UnifiedToolsLauncher.py --verbose`

#### MATLAB Tools Not Working

**Problem**: MATLAB-based tools fail to launch

**Solutions**:

1. Install MATLAB R2020a or later
2. Add MATLAB to system PATH
3. Verify: `matlab -batch "disp('OK')"`

#### Tests Not Running

**Problem**: `pytest` fails with collection errors

**Solutions**:

1. Ensure virtual environment is active
2. Install test dependencies: `pip install pytest>=8.2.0`
3. Run from repository root

### 15.2 CI/CD Status

The repository uses GitHub Actions for continuous integration:

- **Quality Gate**: Ruff, Black, Mypy, pip-audit
- **Multi-Version Testing**: Python 3.11 and 3.12
- **Tauri Builds**: Desktop application packaging

Check CI status at: `https://github.com/dieterolson/Tools/actions`

### 15.3 Getting Help

- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues
- **Quick Start**: `QUICKSTART.md` in repository root

---

## Appendix A: Tool Quick Reference

| Tool                 | Location                      | Launch Command                   |
| -------------------- | ----------------------------- | -------------------------------- |
| Unified Launcher     | Root                          | `python UnifiedToolsLauncher.py` |
| Acid Gas Dewpoint    | src/acid_gas_dewpoint         | `python launch_pyqt6.py`         |
| Baghouse Calculator  | src/baghouse_calculator       | `python launch_pyqt6.py`         |
| Flare Calculator     | src/flare_calculator          | `python launch_pyqt6.py`         |
| Pressure Drop        | src/pressure_drop_calculator  | `python launch_pyqt6.py`         |
| Steam Engine         | src/steam_engine_calculator   | `python launch_pyqt6.py`         |
| ODE Solver           | src/ode_solver                | `python launch_pyqt6.py`         |
| Thermal Predictor    | src/thermal_profile_predictor | `python launch_pyqt6.py`         |
| Optimizer            | src/optimizer_gui             | `python launch_pyqt6.py`         |
| Inertia Calculator   | src/inertia_calculator        | `python launch_pyqt6.py`         |
| URDF Builder         | src/urdf_builder_gui          | `python launch_pyqt6.py`         |
| Financial Calculator | src/financial_calculator      | `python launch_pyqt6.py`         |

---

## Appendix B: Keyboard Shortcuts

| Shortcut       | Action                |
| -------------- | --------------------- |
| Ctrl+L         | Launch selected tool  |
| Ctrl+Tab       | Next category tab     |
| Ctrl+Shift+Tab | Previous category tab |
| F5             | Refresh tool list     |
| Ctrl+Q         | Quit launcher         |

---

## License

This repository is licensed under the MIT License. See individual tool directories for specific licensing terms where applicable.

---

_Generated for Tools Monorepo v1.0_
