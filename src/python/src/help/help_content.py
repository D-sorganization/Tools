# ruff: noqa: E501
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""Help content definitions for tool categories.

This module provides:
- Category-to-topic mappings for the help system
- Inline help text for all tool categories
- Tooltip definitions for common inputs
"""

from __future__ import annotations

# ============================================================================
# CATEGORY TO TOPIC MAPPINGS
# ============================================================================
# Maps launcher categories to help topic files

CATEGORY_TOPIC_MAPPINGS: dict[str, str] = {
    "Process Engineering": "process_calculators",
    "Scientific Modeling": "scientific_tools",
    "Signal Processing": "signal_toolkit",
    "Data Processing": "data_processing",
    "Robotics": "engineering_robotics",
    "Media Processing": "media_processing",
    "Web Applications": "web_applications",
    "Development": "development_tools",
    "Financial": "financial_tools",
}


# ============================================================================
# CATEGORY HELP CONTENT
# ============================================================================
# Inline help content for each tool category

CATEGORY_HELP_CONTENT: dict[str, str] = {
    # -------------------------------------------------------------------------
    # Media Processing
    # -------------------------------------------------------------------------
    "media_processing": """# Media Processing Tools

The Media Processing category contains tools for working with audio and video files.

## Audio Processor (MATLAB)

Professional audio signal processing and multi-track mixing application.

**Features:**
- Multi-format support: WAV, MP3, FLAC, OGG, M4A
- Advanced filtering: FFT-based, Butterworth, custom FIR/IIR
- Audio effects: Reverb, delay, EQ, compression, chorus, pitch shifting
- Multi-track mixing: 8+ tracks with per-track effects chains
- Analysis tools: Spectrogram, FFT analyzer, loudness metering

**Requirements:** MATLAB R2020b+, Signal Processing Toolbox, Audio Toolbox

## Video Processor Platform

AI-powered video analysis platform with golf swing analysis focus.

**Features:**
- Video upload, playback, and annotation
- AI pose detection via MediaPipe
- Drawing and overlay tools
- Audio commentary recording
- 3D visualization with Three.js
- MATLAB physics modeling integration

**Requirements:** Node.js 18+, npm 9+

## Tips

- For audio processing, ensure your MATLAB installation includes the required toolboxes
- The video processor requires a modern browser for full functionality
- Large video files may require significant processing time
""",
    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    "data_processing": """# Data Processing Tools

Tools for analyzing, transforming, and visualizing tabular data.

## Data Processor (PyQt6)

Comprehensive data analysis platform with a desktop GUI.

**Features:**
- Import data from CSV, Excel, JSON formats
- Statistical analysis and summary statistics
- Data visualization with matplotlib
- Signal processing integration
- Export to multiple formats

**Inputs:**
- Tabular data files (CSV, Excel, JSON)
- Column selection for analysis
- Filter and transformation parameters

**Outputs:**
- Processed datasets
- Statistical reports
- Visualization plots
- Exported data files

## Data Processor (Web)

Browser-based data processing with Tauri desktop support.

**Features:**
- React-based modern UI
- Cross-platform desktop app via Tauri
- Real-time data visualization
- Drag-and-drop file import

## Tips

- For large datasets (>100MB), consider using the desktop version
- Use the filter engine for efficient data subsetting
- Statistical functions support both parametric and non-parametric methods
""",
    # -------------------------------------------------------------------------
    # Signal Processing
    # -------------------------------------------------------------------------
    "signal_toolkit": """# Signal Processing Tools

Tools for generating, analyzing, and processing signals and waveforms.

## Function Generator

Generate and visualize mathematical functions and waveforms.

**Waveform Types:**
- Standard: Sine, square, triangle, sawtooth
- Mathematical: Polynomial, exponential, logarithmic
- Custom: User-defined expressions

**Parameters:**
- Frequency (Hz)
- Amplitude
- Phase offset
- DC offset
- Duty cycle (for square waves)

## Polynomial Generator

Generate and fit polynomial functions to data.

**Features:**
- Polynomial coefficient input
- Root finding (real and complex)
- Curve fitting to data points
- Derivative and integral computation
- Taylor series expansion

## Signal Toolkit Widget

Shared library providing signal processing primitives.

**Modules:**
- `filters.py`: Digital filters (lowpass, highpass, bandpass, bandstop)
- `calculus.py`: Differentiation and integration
- `noise.py`: Noise generation (white, pink, brown)
- `fitting.py`: Curve fitting algorithms
- `limits.py`: Limit detection and validation

**Filter Types:**
| Filter | Use Case |
|--------|----------|
| Butterworth | Maximally flat passband |
| Chebyshev | Sharp cutoff, ripple allowed |
| Bessel | Linear phase response |
| FIR | Custom frequency response |

## Tips

- Use the preview feature to visualize signals before exporting
- For real-time applications, consider filter order vs. latency tradeoff
- The calculus module uses numerical differentiation - be aware of noise amplification
""",
    # -------------------------------------------------------------------------
    # Scientific Modeling
    # -------------------------------------------------------------------------
    "scientific_tools": """# Scientific Modeling Tools

Simulation and modeling tools for scientific and engineering applications.

## Solar System Model

Interactive 3D visualization of the solar system with accurate orbital mechanics.

**Features:**
- Accurate planetary positions based on ephemeris data
- Orbital path visualization
- Time controls (play, pause, speed adjustment)
- 3D interactive camera controls
- Moon systems for major planets

**Controls:**
- Mouse drag: Rotate view
- Scroll: Zoom in/out
- Click planet: Center view and show info

## RRT Path Planner

Rapidly-exploring Random Trees path planning for robotics.

**Features:**
- 3D environment with obstacles
- Dual implementation (MATLAB and Python)
- Star Wars-themed visualization
- AI pursuit system with dynamic replanning

**Parameters:**
- Start and goal positions
- Obstacle configuration
- Step size and max iterations
- Goal bias probability

## ODE Solver

Solve systems of ordinary differential equations interactively.

**Preset Examples:**
- Exponential decay
- Harmonic oscillator
- Lotka-Volterra (predator-prey)
- Van der Pol oscillator

**Solver Methods:**
- RK45 (default, adaptive)
- RK23 (lower order adaptive)
- DOP853 (high accuracy)
- Radau (stiff systems)
- BDF (stiff systems)

## Thermal Profile Predictor

Predict temperature profiles in heated vessels over time.

**Inputs:**
- Thermal mass (J/K)
- Heat loss coefficient (W/K)
- Power profile (constant, ramp, step)
- Initial temperature
- Time range

**Outputs:**
- Temperature vs. time curves
- Steady-state temperature
- Condensation risk assessment

## Multi-Parameter Analysis

Sensitivity analysis across multiple parameter dimensions.

**Demo Functions:**
- Rosenbrock (optimization benchmark)
- Rastrigin (multimodal)
- Sphere (simple convex)
- Himmelblau (multiple minima)

## Optimizer GUI (legacy shim)

`src/optimizer_gui` is now a compatibility launcher only. The standalone
optimizer GUI was consolidated into the Movement Optimizer app
(`src/movement_optimizer`); launching the old path opens that application.

## Tips

- For stiff ODE systems, use Radau or BDF solvers
- The thermal predictor assumes lumped capacitance model
- Multi-parameter analysis benefits from parallel processing on multi-core systems
""",
    # -------------------------------------------------------------------------
    # Process Engineering (24 Calculators)
    # -------------------------------------------------------------------------
    "process_calculators": """# Process Engineering Calculators

The repository includes 24 specialized process engineering calculators for industrial applications. All calculators feature:
- PyQt6 GUI with theme support
- Web interface option (React/Tauri)
- Shared calculation engines from `upstream_drift_tools`

## Thermodynamic Calculators

### Acid Gas Dewpoint Calculator
Calculate dewpoints for acid gases (HF, HCl, H2S) in syngas systems.

**Inputs:** Temperature, pressure, gas composition (H2O, HF, HCl, H2S)
**Outputs:** Individual and overall dewpoints, safety margins, warnings

### Syngas Water Calculator
Calculate water content and dew point in syngas systems.

**Methods:** Antoine, Buck, IAPWS-IF97, Magnus
**Output Units:** mg/Nm3, ppmv, g/m3, lb/MMscf

### Steam Engine Calculator
Calculate steam thermodynamic properties.

**Modes:** T&P, Saturated (from T), Saturated (from P)
**Properties:** Density, enthalpy, entropy, internal energy, Cp, Cv

## Equipment Sizing Calculators

### Baghouse Calculator
Design and performance analysis for baghouse filter systems.

**Inputs:** Gas flow rate, solids input rates, removal efficiencies
**Outputs:** Solids removal rates, drum fill time, air-to-cloth ratio

### Flare Calculator
Size flare systems and determine safety zones.

**Outputs:** Flare dimensions, exit velocity, heat release, radiation zones

### Scrubber Calculator
Design wet scrubber systems for gas cleaning.

### Pressure Drop Calculator
Calculate pressure drops in piping systems.

**Pipe Sizes:** 0.5" to 24" nominal
**Schedules:** 5, 10, 20, 40, 80, STD, XS, XXS
**Methods:** Colebrook, Swamee-Jain, Churchill, Haaland

### TRC Vessel Designer
Design thermal reaction chamber vessels with refractory lining.

**Outputs:** Net internal volume, refractory mass, residence time

## Process Unit Calculators

### WGS Reactor Calculator
Water-gas shift reactor design and analysis.

### PSA Package
Pressure Swing Adsorption system design.

### Syngas Compression Calculator
Size syngas compression systems.

## Electrical and Control

### Electrode Advisor
3-phase electrical system analysis for electrode heating.

**Inputs:** Phase currents/voltages, electrode depths, bath geometry
**Outputs:** Total power, phase resistances, system status

## Unit Conversion

### Flow Rate Converter
Convert between mass, molar, and volumetric flow rates.

**Mass:** kg/s, kg/h, lb/h, ton/h
**Molar:** mol/s, kmol/h, lbmol/h
**Volumetric:** m3/s, L/min, CFM, GPM

## Additional Calculators

| Calculator | Purpose |
|------------|---------|
| Heat Exchanger | LMTD, effectiveness-NTU |
| Pump Sizing | Head, power, NPSH |
| Tank Volume | Storage capacity |
| Relief Valve Sizer | Safety valve sizing |
| Cooling Tower | Performance analysis |
| Distillation Column | Stage calculations |
| Reactor Sizing | CSTR, PFR, batch design |
| Catalyst Bed | Volume, pressure drop |
| Combustion | Air requirements, exhaust |
| Mass Balance | Process mass balance |
| Energy Balance | Process energy balance |

## Common Inputs

Most calculators accept gas compositions in the following format:

| Component | Symbol | Typical Range |
|-----------|--------|---------------|
| Hydrogen | H2 | 20-60% |
| Carbon Monoxide | CO | 10-40% |
| Methane | CH4 | 0-15% |
| Carbon Dioxide | CO2 | 5-30% |
| Nitrogen | N2 | 0-10% |
| Water | H2O | 5-40% |
| Hydrogen Sulfide | H2S | 0-2% |

## Tips

- All calculators validate input ranges before calculation
- Use the preset compositions for quick estimates
- Export results to CSV for documentation
- The shared engines ensure consistency across tools
""",
    # -------------------------------------------------------------------------
    # Financial Tools
    # -------------------------------------------------------------------------
    "financial_tools": """# Financial Tools

Financial modeling tools for industrial plant operations.

## Financial Calculator

Comprehensive financial modeling for plant operations.

### Plant Operations
- **Capacity (TPD):** Tons per day nameplate capacity
- **Operating Days:** Days per year (typically 330-350)
- **Utilization:** Percentage of capacity utilized

### Revenue Modeling
- Product pricing per unit
- Production volumes
- Multiple product streams

### Variable Costs
| Cost Type | Description |
|-----------|-------------|
| Feedstock | Raw material costs |
| Labor | Operating labor |
| Utilities | Power, water, steam |
| Maintenance | Routine maintenance |

### Fixed Costs
- Fixed labor (supervision, admin)
- Insurance
- Property taxes
- Depreciation

### Capital and Financing
- Total capital investment
- Debt/equity ratio
- Interest rate
- Depreciation method (straight-line, MACRS)

### Outputs
- Annual feedstock processing
- Total revenue and costs
- Net income, EBITDA
- Return on equity
- Payback period
- 10-year financial projections

## Tips

- Use sensitivity analysis to assess key variable impacts
- The model uses pre-tax calculations; adjust for tax effects
- Export projections to Excel for further analysis
""",
    # -------------------------------------------------------------------------
    # Engineering and Robotics
    # -------------------------------------------------------------------------
    "engineering_robotics": """# Engineering and Robotics Tools

Tools for mechanical engineering and robotics applications.

## Inertia Calculator

Calculate and validate inertia tensors for rigid bodies.

**Primitive Shapes:**
- Box (rectangular prism)
- Cylinder
- Sphere
- Capsule (cylinder + hemispheres)

**Inputs:**
- Shape type
- Dimensions (per shape)
- Mass
- Orientation (optional)

**Outputs:**
- Inertia tensor (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
- Principal axes
- Physical validity check
- URDF-compatible format

## URDF Builder

Generate parametric URDF models for robotics applications.

**Features:**
- Visual link and joint definition
- Geometry primitives (box, cylinder, sphere, mesh)
- Collision geometry
- Inertia property generation
- Joint types: revolute, prismatic, continuous, fixed
- Preview and validation
- Export to URDF XML

**Workflow:**
1. Define links with visual/collision geometry
2. Connect links with joints
3. Add mass and inertia properties
4. Preview the robot model
5. Export to URDF file

## Humanoid Builder

Build parametric humanoid character models with anthropometric calculations.

**Features:**
- Anthropometric presets (adult male, female, child)
- Body segment customization
- Automatic mass distribution
- Inertia calculation per segment
- URDF export for simulation

**Body Segments:**
- Head, neck, torso
- Upper arm, forearm, hand
- Thigh, shank, foot

## C3D Viewer

View and analyze C3D motion capture files.

**Features:**
- C3D file import
- 3D marker visualization
- Frame-by-frame playback
- Marker trajectory analysis
- Marker labeling and grouping
- Export marker data to CSV

**Controls:**
- Space: Play/pause
- Arrow keys: Frame step
- Mouse drag: Rotate view
- Scroll: Zoom

## Tips

- URDF files should use SI units (meters, kilograms, radians)
- Inertia tensors must be symmetric and positive definite
- For humanoid models, verify center of mass location
- C3D files from different systems may have different coordinate conventions
""",
    # -------------------------------------------------------------------------
    # Web Applications
    # -------------------------------------------------------------------------
    "web_applications": """# Web Applications

Browser-based tools with optional offline support.

## Aurora CAS Calculator

Web-based Computer Algebra System (CAS) calculator.

**Symbolic Math:**
- factor, expand, simplify
- solve (algebraic equations)
- polynomial operations

**Calculus:**
- Derivatives (diff)
- Integrals (integrate)
- Limits
- Taylor series

**Linear Algebra:**
- Matrix operations
- Determinants
- Eigenvalues/eigenvectors
- Matrix decompositions

**Robotics Support:**
- Screw theory operations
- SE(3) transformations
- Twist and wrench calculations

**Interface:**
- Touch-friendly soft keyboard
- Mode-specific key layouts
- Expression history
- Result export

## Unit Converter (PWA)

NIST-compliant unit converter with offline support.

**Categories:**
| Category | Example Units |
|----------|---------------|
| Length | m, ft, in, km, mi |
| Mass | kg, lb, oz, ton |
| Volume | L, gal, m3, ft3 |
| Temperature | C, F, K |
| Pressure | Pa, bar, psi, atm |
| Energy | J, kWh, BTU, cal |
| Power | W, hp, BTU/h |
| Flow | m3/s, GPM, CFM |
| Velocity | m/s, ft/s, mph, km/h |
| Force | N, lbf, kgf |
| Torque | Nm, ft-lbf |
| Density | kg/m3, lb/ft3 |

**Features:**
- Offline support (Progressive Web App)
- NIST-standard conversion factors
- iOS-optimized interface
- Custom unit support

## URDF Viewer

Web-based 3D viewer for URDF robot models.

**Features:**
- Three.js 3D rendering
- URDF file upload
- Interactive model inspection
- Joint state manipulation
- Screenshot export

## Tips

- PWA apps can be installed to home screen for app-like experience
- The CAS calculator requires a network connection for complex operations
- URDF viewer supports meshes in DAE and STL formats
""",
    # -------------------------------------------------------------------------
    # Development Tools
    # -------------------------------------------------------------------------
    "development_tools": """# Development Tools

Utilities for project management and automation.

## Folder Packer Pro

Project archiving and distribution tool.

**Features:**
- Selective file/folder packaging
- Exclusion patterns (node_modules, __pycache__, .git)
- Multiple output formats (ZIP, TAR, 7z)
- Integrity verification (checksums)
- Size estimation before packing

**Exclusion Presets:**
- Python: __pycache__, *.pyc, venv, .eggs
- Node.js: node_modules, .npm, dist
- IDE: .idea, .vscode, *.swp
- VCS: .git, .svn, .hg

## Folder Fix Pro

Automated folder structure cleanup and organization.

**Features:**
- Empty folder detection and removal
- Duplicate file detection (by hash)
- Structure validation against templates
- Batch renaming with patterns
- Dry-run mode for preview

**Operations:**
1. Scan directory tree
2. Identify issues (empty dirs, duplicates)
3. Preview proposed changes
4. Apply changes (with undo support)

## PDF Renamer

Batch rename PDF files based on content or metadata.

**Extraction Methods:**
- Title from metadata
- First heading from content
- Date extraction
- Custom patterns (regex)

**Features:**
- Preview before apply
- Collision handling (append number)
- Transaction log for undo
- API mode for automation

## Tips

- Always use dry-run mode first to preview changes
- Folder Packer respects .gitignore patterns
- PDF Renamer works best with text-based PDFs (not scanned images)
- Keep the transaction log for potential undo operations
""",
}


# ============================================================================
# TOOLTIP DEFINITIONS
# ============================================================================
# Common tooltips for input fields across tools

TOOLTIPS: dict[str, str] = {
    # General
    "temperature": "Temperature in specified units. Common ranges: -40 to 1500 C for process applications.",
    "pressure": "Pressure in specified units. Absolute pressure unless noted as gauge.",
    "flow_rate": "Volumetric or mass flow rate. Ensure consistent units with other inputs.",
    # Gas composition
    "gas_composition": "Mole fractions or percentages. Components should sum to 100% (or 1.0 for fractions).",
    "h2_content": "Hydrogen content. Typical syngas: 20-60%",
    "co_content": "Carbon monoxide content. Typical syngas: 10-40%",
    "h2o_content": "Water vapor content. Important for dewpoint calculations.",
    # Equipment
    "pipe_diameter": "Nominal pipe size. Select from standard sizes.",
    "pipe_schedule": "Pipe wall thickness designation. Higher numbers = thicker walls.",
    "pipe_length": "Total equivalent length including fittings.",
    "pipe_roughness": "Internal surface roughness. Affects friction factor.",
    # Process
    "efficiency": "Fractional efficiency (0-1) or percentage (0-100).",
    "residence_time": "Time material spends in the process unit.",
    "conversion": "Fractional conversion of limiting reactant.",
    # Financial
    "capex": "Capital expenditure - total project cost.",
    "opex": "Operating expenditure - annual operating costs.",
    "depreciation": "Annual depreciation charge for tax purposes.",
    "discount_rate": "Rate used for NPV calculations. Typically 8-15% for industrial projects.",
    # Signal processing
    "sampling_rate": "Samples per second (Hz). Must satisfy Nyquist criterion.",
    "cutoff_frequency": "Filter cutoff frequency. Must be less than half the sampling rate.",
    "filter_order": "Number of poles in the filter. Higher = sharper cutoff but more latency.",
    # Robotics
    "inertia_tensor": "3x3 symmetric positive definite matrix. Diagonal elements are moments of inertia.",
    "joint_limits": "Min and max joint angles in radians (or meters for prismatic).",
    "link_mass": "Mass of the link in kilograms.",
}


# ============================================================================
# GETTING STARTED CONTENT
# ============================================================================

GETTING_STARTED_CONTENT = """# Getting Started with the Unified Tools Launcher

Welcome to the Unified Tools Launcher! This guide will help you get started
with the tool collection.

## Launching the Application

```bash
python UnifiedToolsLauncher.py
```

## Interface Overview

The launcher is organized into tabs by category:

| Tab | Description |
|-----|-------------|
| Process Engineering | Industrial calculators |
| Scientific Modeling | Simulation tools |
| Signal Processing | Waveform generators, filters |
| Data Processing | Data analysis platforms |
| Robotics | URDF builders, inertia calculators |
| Media Processing | Audio/video tools |
| Web Applications | Browser-based tools |
| Development | Project utilities |

## Launching a Tool

1. Select the category tab
2. Find the tool card
3. Click **Launch Tool**

## Getting Help

- Press **F1** to open the User Manual
- Click the **?** button next to a tool for context-sensitive help
- Use the **Help** menu for additional options

## Debug Mode

Enable **Debug Mode** in the header to see detailed logging output
in the activity log area at the bottom of the window.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F1 | Open User Manual |
| Ctrl+L | Launch selected tool |
| Ctrl+Tab | Next category tab |
| Ctrl+Shift+Tab | Previous category tab |
| Ctrl+Q | Quit |

## Next Steps

- Browse the tool categories to explore available tools
- Read the User Manual for detailed documentation
- Check tool descriptions for specific requirements (MATLAB, Node.js, etc.)
"""


def initialize_help_manager() -> None:
    """Initialize the help manager with all content.

    Call this function during application startup to register
    all help content with the HelpManager singleton.
    """
    from .help_system import get_help_manager

    manager = get_help_manager()

    # Register category mappings
    for category, topic_id in CATEGORY_TOPIC_MAPPINGS.items():
        manager.register_category_mapping(category, topic_id)

    # Register inline content
    for topic_id, content in CATEGORY_HELP_CONTENT.items():
        manager.register_topic(topic_id, content)

    # Register getting started
    manager.register_topic("getting_started", GETTING_STARTED_CONTENT)

    # Register tooltips
    tooltip_mgr = manager.tooltip_manager
    for key, text in TOOLTIPS.items():
        tooltip_mgr.register_tooltip(key, text)


__all__ = [
    "CATEGORY_HELP_CONTENT",
    "CATEGORY_TOPIC_MAPPINGS",
    "GETTING_STARTED_CONTENT",
    "TOOLTIPS",
    "initialize_help_manager",
]
