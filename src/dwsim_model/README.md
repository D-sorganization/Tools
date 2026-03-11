# DWSIM Gasification Model

DWSIM-backed gasification process simulation with Tkinter desktop GUI and full CLI.

## Current Features

- Purpose: Gasification process simulation backed by DWSIM runtime (pythonnet)
- Category: Process Simulation
- Engine: `dwsim_model` shared library (in `src/shared/python/dwsim_model/`)
- Surface support: Tkinter=implemented, CLI=implemented, PyQt6=planned, Web=planned
- Test visibility: 15 test files under tests/

## Quick Start

### Tkinter GUI
```bash
python src/dwsim_model/launch_gui.py
```

### CLI
```bash
python -m dwsim_model run --scenario baseline
python -m dwsim_model sweep --param feeds.Gasifier_Biomass_Feed.mass_flow_kg_s --min 2.0 --max 6.0 --steps 9
python -m dwsim_model validate --config config/master_config.yaml
python -m dwsim_model export --output model.dwxml
python -m dwsim_model summary
```

## Implementation State

- Tkinter launcher: Implemented (tabbed design GUI with sidebar)
- CLI launcher: Implemented (run, sweep, validate, export, summary subcommands)
- PyQt6 launcher: Not yet implemented (Tkinter GUI needs porting)
- Web surface: Not yet implemented
- README last reviewed: 2026-03-11

## Dependencies

- `pythonnet >= 3.0.0` (DWSIM runtime bridge)
- `pydantic >= 2.0.0` (config validation)
- `PyYAML >= 6.0` (shared dependency)

## Implementation Gaps

- PyQt6 GUI port not yet implemented (currently Tkinter).
- Web surface not yet implemented.
- Requires DWSIM to be installed on the host system for simulation execution.
