# Parametric URDF Builder

A PyQt6-based GUI application for generating parametric URDF (Unified Robot Description Format) models for robotics applications.

## Architecture

The tool follows a clean modular architecture with **5 GUI-independent core modules** and a thin UI shell:

```
urdf_builder_gui/
├── python/urdf_builder_gui/         ← Canonical module source
│   ├── contracts.py                 ← DbC require/ensure
│   ├── anthropometric_model.py      ← Constants, config, physics
│   ├── urdf_generator.py            ← URDF XML generation + validation
│   ├── preview_generator.py         ← Human-readable model previews
│   ├── theme.py                     ← Catppuccin Mocha palette + stylesheet
│   └── ui/pyqt6/main_window.py      ← Thin PyQt6 UI shell
├── tests/
│   └── test_urdf_builder_gui.py     ← 75+ TDD tests
├── gui_registration.py              ← Fleet launcher registration
├── launch_pyqt6.py                  ← Standalone launcher
└── README.md
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **DRY** | Shared `HEIGHT_RATIOS`, `MASS_RATIOS` constants; no duplicate logic |
| **DbC** | `require()`/`ensure()` contracts on all public functions |
| **LoD** | `_get_config()` gateway prevents generator from touching GUI widgets |
| **TDD** | 75+ tests covering all modules, boundaries, and integration |
| **Orthogonality** | Core modules have zero GUI dependencies |
| **Reusability** | Web viewer API reuses same core modules as PyQt6 GUI |

## Key Features

- **Parametric Model Generation**: Define robots using height, mass, and proportion factors
- **6 Templates**: Full Humanoid, Upper Body Only, Lower Body Only, Torso + Arms, Torso + Legs, Custom
- **Gender-Based Scaling**: Adjust body proportions using anthropometric gender factors
- **Geometry Options**: Box collision primitives (with collision geometry toggle)
- **Joint Configuration**: Configurable damping, friction, and limit parameters
- **Physics-Based Inertia**: Computed from actual box model (`I = m(h²+d²)/12`)
- **Live Preview**: View model structure and estimated segment sizes
- **URDF Validation**: Structural validation before export (XML, links, joints)
- **Direct Export**: Save URDF files with proper XML formatting
- **Web API**: FastAPI endpoints for headless URDF generation

## Installation

### Dependencies

```bash
pip install PyQt6
```

### Running

```bash
# Via fleet launcher
python launch_pyqt6.py

# Direct module execution
python -m urdf_builder_gui.ui.pyqt6.main_window
```

## Usage

### Body Parameters Tab

1. Enter a robot name (used as the URDF robot element name)
2. Set the total height in meters (0.5 – 3.0 m)
3. Set the total mass in kilograms (20 – 200 kg)
4. Adjust the gender factor slider (affects shoulder/hip width ratios)
5. Select a model template

### Proportions Tab

1. Adjust individual body segment proportions using sliders (50% – 150%)
2. Click "Reset to Defaults" to restore 100% scaling

### Options Tab

1. **Geometry Options**: Select default visual/collision geometry type
2. **Joint Options**: Set default damping and friction coefficients
3. **Inertia Calculation**: Choose primitive, mesh-based, or scaled inertia mode

### Actions

1. Click **Preview Structure** to see a summary (segment sizes, options, template segments)
2. Click **Generate URDF** to create the XML
3. Click **Export URDF File** to save to disk (validates URDF structure first)

## Input Parameters

### Basic Parameters

| Parameter     | Unit | Range     | Description                      |
| ------------- | ---- | --------- | -------------------------------- |
| Robot Name    | -    | text      | Valid XML NCName identifier      |
| Height        | m    | 0.5 - 3.0 | Total standing height            |
| Mass          | kg   | 20 - 200  | Total body mass                  |
| Gender Factor | %    | 0 - 100   | Female (0) to Male (100) scaling |

### Body Proportions

| Parameter      | Range      | Description                |
| -------------- | ---------- | -------------------------- |
| Shoulder Width | 50% - 150% | Biacromial breadth scaling |
| Hip Width      | 50% - 150% | Bi-iliac breadth scaling   |
| Arm Length     | 50% - 150% | Upper + lower arm scaling  |
| Leg Length     | 50% - 150% | Thigh + shin scaling       |
| Torso Length   | 50% - 150% | Lumbar + thorax scaling    |
| Head Size      | 50% - 150% | Head diameter scaling      |

### Joint Configuration

| Parameter        | Unit      | Range      | Description                  |
| ---------------- | --------- | ---------- | ---------------------------- |
| Default Damping  | N·m·s/rad | 0 - 100    | Viscous damping coefficient  |
| Default Friction | N·m       | 0 - 100    | Coulomb friction coefficient |
| Density          | kg/m³     | 500 - 2000 | Default material density     |

## Mathematical Models

### Segment Length Estimation

Segment lengths are calculated from total height using de Leva (1996) anthropometric ratios
(defined in `anthropometric_model.HEIGHT_RATIOS`):

| Segment    | Ratio  |
|------------|--------|
| Pelvis     | 0.078  |
| Torso      | 0.278  |
| Head       | 0.139  |
| Thigh      | 0.245  |
| Shin       | 0.246  |
| Upper Arm  | 0.186  |
| Forearm    | 0.146  |

### Mass Distribution

Segment masses distributed per `MASS_RATIOS` (de Leva 1996):

| Segment    | Ratio | Note      |
|------------|-------|-----------|
| Pelvis     | 0.112 |           |
| Torso      | 0.355 | Combined  |
| Head       | 0.069 |           |
| Thigh      | 0.142 | Per leg   |
| Shin       | 0.043 | Per leg   |
| Upper Arm  | 0.027 | Per arm   |
| Forearm    | 0.016 | Per arm   |

### Inertia Calculation

For primitive geometry mode, inertia is computed assuming uniform density box model:

```
I_box_xx = (1/12) * m * (h² + d²)
I_box_yy = (1/12) * m * (w² + d²)
I_box_zz = (1/12) * m * (w² + h²)
```

Cylinder and sphere inertia formulae also available in `anthropometric_model.py`.

## Web API

The URDF Viewer web application (`src/web_applications/urdf_viewer/`) exposes API
endpoints that reuse the same core modules:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate URDF XML from parameters |
| `/api/preview` | POST | Generate human-readable preview |
| `/api/templates` | GET | List available templates |
| `/api/models` | GET | List uploaded models |
| `/api/upload` | POST | Upload a URDF file |
| `/api/models/{name}` | GET | Download a model file |

## Testing

```bash
cd src/urdf_builder_gui
python -m pytest tests/test_urdf_builder_gui.py -v
```

Test coverage includes: contracts, theme, anthropometric model, URDF generator,
URDF validator, preview generator, URDFConfig, integration round-trips,
and file sync verification.

## References

- URDF Specification: http://wiki.ros.org/urdf/XML
- de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.

## Implementation State

- PyQt6 launcher: ✅ Implemented
- Web API: ✅ Implemented (FastAPI)
- Tests: ✅ 75+ passing
- README last reviewed: 2026-03-14
