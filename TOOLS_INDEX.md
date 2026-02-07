# Tools Repository - Complete Project Index

> **Last updated**: 2026-02-07
> **Total**: 63+ distinct tools, libraries, and applications

This index catalogs every tool and project in the repository to ensure nothing is lost or forgotten.

---

## Process Engineering Calculators (15 tools)

| # | Tool | Path | Type | Web UI |
|---|------|------|------|--------|
| 1 | Electrode Advisor | `src/electrode_advisor/` | PyQt6 | React |
| 2 | TRC Vessel Designer | `src/trc_vessel_designer/` | PyQt6 | React |
| 3 | Syngas Compression | `src/syngas_compression/` | PyQt6 | React |
| 4 | Syngas Water Calculator | `src/syngas_water_calculator/` | PyQt6 | - |
| 5 | Pressure Drop Calculator | `src/pressure_drop_calculator/` | PyQt6 | React |
| 6 | Acid Gas Dewpoint | `src/acid_gas_dewpoint/` | PyQt6 | React |
| 7 | Scrubber Calculator | `src/scrubber_calculator/` | PyQt6 | React |
| 8 | WGS Reactor | `src/wgs_reactor/` | PyQt6 | React |
| 9 | Flare Calculator | `src/flare_calculator/` | PyQt6 | React |
| 10 | Baghouse Calculator | `src/baghouse_calculator/` | PyQt6 | React |
| 11 | PSA Package | `src/psa_package/` | PyQt6 | React |
| 12 | Steam Engine Calculator | `src/steam_engine_calculator/` | PyQt6 | - |
| 13 | Flow Rate Converter | `src/flow_rate_converter/` | PyQt6 | - |
| 14 | Financial Calculator | `src/financial_calculator/` | PyQt6 | React |
| 15 | Glass Bath FEA | `src/glass_bath_fea/` | PyQt6 | - |

---

## Scientific Modeling & Simulation (7 tools)

### Solar System Model
- **Path**: `src/scientific_modeling/solar_system_model/`
- **Entry**: `launch_solar_system.py`
- **Type**: Python 3D Visualization (PyGame/OpenGL)
- **Description**: Interactive 3D solar system simulation with accurate Keplerian mechanics, trajectory planning, multiple camera modes, date picker, historical events, and educational overlays.

### RRT Path Planner
- **Path**: `src/scientific_modeling/rrt_path_planner/`
- **Type**: Dual implementation - MATLAB GUI + Python OpenGL
- **Description**: Star Wars-themed Rapidly-exploring Random Tree path planner with 3D environment, dynamic obstacle avoidance, AI pursuit system, and cinematic visualization.
  - MATLAB: `matlab/src/gui/starWarsPathPlannerGUI.m`
  - Python: `python/src/star_wars_rrt.py`

### ODE Solver
- **Path**: `src/ode_solver/`
- **Type**: PyQt6 GUI
- **Description**: Differential equation solver with multiple integration methods and presets.

### Thermal Profile Predictor
- **Path**: `src/thermal_profile_predictor/`
- **Type**: PyQt6 GUI
- **Description**: Temperature distribution analysis for thermal systems.

### Multi-Parameter Analysis
- **Path**: `src/multi_param_analysis/`
- **Type**: PyQt6 GUI
- **Description**: Parameter sweep and sensitivity analysis tool.

### Optimizer GUI
- **Path**: `src/optimizer_gui/`
- **Type**: PyQt6 GUI
- **Description**: Optimization interface for constrained and unconstrained optimization.

### MATLAB Core Module
- **Path**: `src/matlab/`
- **Entry**: `run_all.m`
- **Type**: MATLAB
- **Description**: Core MATLAB scientific computing module for Golf Biomechanics Simulator.

---

## Data Processing & Signal Analysis (4 tools)

### Data Processor
- **Path**: `src/data_processing/data_processor/`
- **Type**: PyQt6 GUI + React Web
- **Description**: Time series CSV/Parquet analyzer and converter with signal processing and filtering. Core analysis engine with 40+ modules.

### Function Generator
- **Path**: `src/function_generator/`
- **Type**: PyQt6 GUI + React Web
- **Description**: Waveform generator for signal synthesis with 13+ signal types.

### Signal Toolkit (Library)
- **Path**: `src/shared/python/signal_toolkit/`
- **Type**: Python Library + Standalone widget
- **Description**: Comprehensive signal processing library with signal generation (13 types), curve fitting, digital filters (Butterworth, Chebyshev, Bessel), calculus operations, noise generation, and I/O support (CSV, JSON, MAT, NPZ).

### Polynomial Generator
- **Path**: `src/shared/python/signal_toolkit/polynomial_generator.py`
- **Type**: Interactive Python script
- **Description**: Interactive polynomial and curve generation utility.

---

## Media Processing (2 major projects)

### Audio Processor
- **Path**: `src/media_processing/audio_processor/`
- **Entry**: `matlab/audio_signal_processor/launch_audio_processor_pro.m`
- **Type**: MATLAB Application
- **Description**: Professional audio signal processing suite with multi-format loading (WAV, MP3, FLAC, OGG, M4A), FFT-based filters, time-domain filters, effects library (reverb, delay, chorus, phaser, flanger), multi-track mixer, convolution reverb, and spectrogram analysis.

### Video Processor
- **Path**: `src/media_processing/video_processor/`
- **Entry**: `apps/web/launch_platform.py`
- **Type**: Next.js Web Platform
- **Description**: AI-powered golf swing video analysis platform for coaches with video upload, analysis, and student sharing capabilities.

---

## Robotics & Biomechanics (4 tools)

### Inertia Calculator
- **Path**: `src/inertia_calculator/`
- **Type**: PyQt6 GUI
- **Description**: Mass and inertia tensor calculations for primitive geometric shapes (box, cylinder, sphere, capsule, ellipsoid) compatible with URDF format.

### URDF Builder GUI
- **Path**: `src/urdf_builder_gui/`
- **Type**: PyQt6 GUI
- **Description**: Parametric URDF builder for generating robot models with template-based design, gender-based scaling, and configurable joint parameters.

### Humanoid Builder GUI
- **Path**: `src/humanoid_builder_gui/`
- **Type**: PyQt6 GUI
- **Description**: Humanoid character builder using de Leva (1996) anthropometric data with build types (ectomorph, mesomorph, endomorph), gender-specific models, BMI calculation, and URDF export.

### C3D Viewer
- **Path**: `src/c3d_viewer/`
- **Type**: PyQt6 GUI
- **Description**: C3D motion capture file viewer for biomechanics and gait analysis with marker visualization, trajectory analysis, analog channel support, force plate analysis, and multi-format export (CSV, JSON, NPZ).

---

## Document Processing (1 tool)

### PDF Renamer
- **Path**: `src/document_processing/pdf_renamer/`
- **Entry**: `launch_gui.py`
- **Type**: PyQt6 GUI
- **Description**: AI-powered PDF renamer with layered extraction (metadata, regex patterns), OpenAI API fallback, dual processing modes (batch/API-only), manual review workflow, parallel processing, duplicate detection, and transaction logging.

---

## Folder & File Management (4 tools)

### Folder Tool ("Folder Fix")
- **Path**: `src/tools/folder_tools/folder_tool/`
- **Entry**: `launch_folder_tool.py`
- **Type**: Tkinter GUI
- **Description**: Comprehensive folder processor with modes: Combine & Copy, Flatten & Tidy, Copy & Prune, Deduplicate, Analyze & Report. Features file filtering, organization by type/date, bulk archive extraction (.zip, .rar, .7z), preview mode, and automatic backup.

### Folder Tool Pro
- **Path**: `src/tools/folder_tools/folder_tool_pro/`
- **Type**: Tkinter GUI
- **Description**: Professional version of Folder Tool with enhanced features.

### Folder Packer Pro
- **Path**: `src/tools/folder_tools/folder_packer_pro/`
- **Entry**: `folder_packer_pro.py`
- **Type**: Tkinter GUI
- **Description**: Professional project packing tool with pack/unpack to encrypted archives, AES-256 encryption, multiple compression levels, Git integration, syntax highlighting, smart file filtering, batch operations, and manifest export.

### Project Packer
- **Path**: `src/tools/folder_tools/project_packer/`
- **Entry**: `folder_packer_gui.py`
- **Type**: Tkinter GUI
- **Description**: Folder packer/unpacker for programming files with structure preservation, smart exclusions, file type filtering, and batch operations. Supports 30+ file types.

---

## Web Applications (3 apps)

| App | Path | Type |
|-----|------|------|
| Scientific Calculator | `src/web_applications/calculator/` | Flask |
| Unit Converter | `src/web_applications/unit_converter/` | HTML/JS |
| URDF Viewer | `src/web_applications/urdf_viewer/` | FastAPI + React + Three.js |

---

## MATLAB Development Tools (2 tools)

### MATLAB Code Analyzer GUI
- **Path**: `src/tools/matlab_code_analyzer_gui/`
- **Entry**: `launchCodeAnalyzer.m`
- **Type**: MATLAB GUI
- **Description**: Interactive GUI for MATLAB Code Analyzer (MLint) with file/folder selection, configurable options, multiple output formats (CSV, Excel, JSON, Markdown), and results summary.

### MATLAB Utilities
- **Path**: `src/tools/matlab_utilities/`
- **Type**: MATLAB/Python Scripts
- **Description**: MATLAB quality checking and testing utilities.

---

## Shared Libraries (7 packages)

| Library | Path | Description |
|---------|------|-------------|
| Theme System | `src/shared/python/theme/` | Fleet-wide UI theme system (13 built-in themes) |
| Plot Theme | `src/shared/python/plot_theme/` | Shared matplotlib/plot theming |
| Signal Toolkit | `src/shared/python/signal_toolkit/` | Signal processing, filtering, I/O |
| Model Generation | `src/shared/python/model_generation/` | URDF/MJCF building, editing, conversion |
| Humanoid Character Builder | `src/shared/python/humanoid_character_builder/` | Parametric URDF humanoid generation |
| GUI Launcher | `src/shared/python/gui_launcher/` | Shared launcher components |
| Upstream Drift Tools | `src/shared/python/upstream_drift_tools/` | Fleet-wide shared logic (thermo, conversion, robotics) |

---

## Launchers & Infrastructure

| Component | Path | Status |
|-----------|------|--------|
| Unified Launcher (PRIMARY) | `UnifiedToolsLauncher.py` | Active |
| Tile Launcher | `src/python/src/tile_launcher/` | Active |
| Plugin System | `src/python/src/core/` | Active |
| Python Utilities | `src/python/src/utils/` | Active |
| CLI Launcher (deprecated) | `launch_tools_main.py` | Deprecated |

---

## Development Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `analyze_completist_data.py` | Data analysis utility |
| `baseline_assessments.py` | Baseline assessment generation |
| `convert_print_to_logging.py` | Code modernization (print -> logging) |
| `create_issues_from_assessment.py` | GitHub issue automation |
| `enhanced_batch_fix_dry.py` | Batch code fixing |
| `generate_assessment_summary.py` | Assessment reporting |
| `generate_assessments.py` | Assessment generation |
| `pragmatic_programmer_review.py` | Code review utility |
| `quality-check.py` | Code quality verification |
| `setup_hooks.py` | Git hooks setup |
| `setup_precommit.sh` | Pre-commit configuration |
| `snapshot.sh` | Repository snapshot utility |
| `validate_themes.py` | Theme validation |

---

## Verification Tools (`src/verification/`)

| Tool | Purpose |
|------|---------|
| `verify_a11y.py` | Accessibility verification |
| `verify_palette.py` | Color palette verification |
| `verify_palette_final.py` | Final palette validation |

---

## Legacy Directories (kept for backwards compatibility)

These mirror `src/` structure and contain older copies:

| Legacy Path | Canonical Path |
|-------------|---------------|
| `data_processing/` | `src/data_processing/` |
| `scientific_modeling/` | `src/scientific_modeling/` |
| `media_processing/` | `src/media_processing/` |
| `document_processing/` | `src/document_processing/` |
| `web_applications/` | `src/web_applications/` |
| `python/` | `src/python/` |
| `development_tools/folder_tools/` | `src/tools/folder_tools/` |

> Canonical versions are always in `src/`. Legacy paths will be consolidated per issue #566.
