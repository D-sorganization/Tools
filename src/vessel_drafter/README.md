# Vessel Drafter

Refractory vessel design tool with STEP, STL, BREP, and GLTF export.

## Current Features

- Purpose: Parametric vessel geometry creation and engineering drawing export
- Category: Process Simulation
- Engine: `vessel_drafter` local package (in `python/vessel_drafter/`)
- Surface support: PyQt6=implemented, Web=disabled
- Test visibility: 4 test files + conftest under tests/

## Quick Start

### PyQt6 GUI
```bash
python src/vessel_drafter/launch_pyqt6.py
```

## Implementation State

- PyQt6 launcher: Implemented (parametric design GUI)
- CLI launcher: Not yet implemented
- Web surface: Disabled
- README last reviewed: 2026-03-11

## Dependencies

- `build123d` (CAD kernel)
- `numpy`, `matplotlib` (analysis and preview)
- `PyQt6` (desktop GUI)

## Implementation Gaps

- Web surface not yet implemented.
- CLI surface not yet implemented.
