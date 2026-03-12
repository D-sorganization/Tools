# Programmatic Drafting

Programmatic STEP drafting for D-sorganization projects.

This repo is the CAD automation home for inspectable geometry exports that can be
generated in CI and opened in FreeCAD locally. It currently tracks the default
electrode advisor layout from the web tool plus standalone drafting studies.

## Why this repo exists

- Keeps CAD automation separate from application repos.
- Uses code as the source of truth for geometry and exports reproducible STEP files.
- Makes STEP generation CI-friendly instead of depending on a desktop FreeCAD install.

## Current projects

- `electrode_advisor_default_layout`
  - Source of truth: `Tools/src/electrode_advisor/web/src/components/GlassBath3DViewer.tsx`
  - Default bath: rectangular, `3.0 m x 2.0 m x 2.5 m`
  - Default glass level: `1.5 m`
  - Default electrodes: `3`
  - Default electrode length: `1500 mm`
  - Default worn length: `150 mm`
  - Default operating current: `2500 A`
- `cylindrical_bath_layout`
  - Bath: `50 in` inner diameter, `14 in` depth
  - Refractory annulus: `6 in` radial thickness
  - Electrodes: `3`, radial, `120 deg` apart
  - Electrode diameter: `2 in`
  - Electrode insertion into inner circle: `14 in`
  - Electrode extension outside inner circle: `36 in`
  - Vertical placement assumption: mid-depth centerline
- `vessel_drafter_default`
  - Bath: `50 in` inner diameter, `14 in` glass depth
  - Plenum: `14 in` above the glass bath
  - Layer stackup: `6 in` hot face, `4.5 in` IFB, `1 in` duraboard, `0.5 in` steel
  - Dished heads: offset elliptical head profiles with constant-thickness shell layers
  - Colors: orange glass, reddish-orange hot face, tan IFB, white duraboard, grey steel
  - Ports: multiple sidewall and lid ports with editable clocking, size, and placement
  - GUI: PyQt6 vessel editor with live 2D previews, a rotatable 3D tab, layer
    visibility toggles, a vertical section cut, and STEP export
  - Reporting: computed material volumes in `ft^3`, surface areas in `ft^2`,
    masses, densities, conductivity, and thermal expansion metadata in the GUI
    and JSON manifest

Generated artifacts:

- `generated/electrode_advisor_default/electrode_advisor_default_layout.step`
- `generated/electrode_advisor_default/electrode_advisor_default_layout.json`
- `generated/cylindrical_bath_layout/cylindrical_bath_layout.step`
- `generated/cylindrical_bath_layout/cylindrical_bath_layout.json`
- `generated/vessel_drafter_default/vessel_drafter_default.step`
- `generated/vessel_drafter_default/vessel_drafter_default.json`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m programmatic_drafting.cli export-electrode-advisor-default
python -m programmatic_drafting.cli export-cylindrical-bath-layout
python -m programmatic_drafting.cli export-vessel-drafter-default
python -m programmatic_drafting.cli launch-vessel-drafter-gui
```

Open the generated STEP file in FreeCAD for inspection.
You can also launch the dedicated GUI with `vessel-drafter`.

## Repo Outline

- `src/programmatic_drafting/models/`
  - Geometry inputs and source-of-truth defaults copied from product code.
- `src/programmatic_drafting/projects/`
  - Concrete drafting projects, profile helpers, and shape builders.
- `src/programmatic_drafting/exporters/`
  - STEP export utilities and manifest writers.
- `src/programmatic_drafting/preview/`
  - Preview data builders shared by the GUI.
- `src/programmatic_drafting/gui/`
  - PyQt6 user interfaces, renderers, and port-editing widgets.
- `tests/`
  - Contract tests for default values, layout math, and STEP export.
- `docs/projects/`
  - Project-specific drafting notes and geometry assumptions.
- `.github/workflows/`
  - Fleet-style CI, PR hygiene, and weekly CAD validation.

## CAD Strategy

- Primary kernel: `build123d`
- Export format: STEP
- Inspection target: FreeCAD
- Coordinate convention in CAD: `Z` up
- Coordinate convention in the source viewer: `Y` up

The drafting code applies a documented axis transform so the model opens in a
natural CAD orientation while still matching the tool layout.
