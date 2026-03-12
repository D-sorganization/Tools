# Electrode Advisor Default Layout

This project captures the current default 3D layout of the electrode advisor
tool as an inspectable STEP model.

## Source of truth

Defaults were copied from:

- `Tools/src/electrode_advisor/web/src/components/GlassBath3DViewer.tsx`
- `Tools/src/electrode_advisor/web/src/components/ElectrodeAdvisorCalculator.tsx`

## Imported defaults

- Bath shape: rectangular
- Bath width: `3.0 m`
- Bath depth: `2.0 m`
- Bath height: `2.5 m`
- Glass level: `1.5 m`
- Electrode count: `3`
- Electrode top offset above bath: `0.1 m`
- Electrode diameter: `150 mm`
- Electrode current length: `1500 mm`
- Electrode worn length: `150 mm`
- Operating current: `2500 A`

## CAD assumptions

The source viewer is not a manufacturing model, so this repo adds a few
explicit drafting assumptions to make the geometry inspectable in CAD:

- Bath shell thickness: `25 mm`
- Glass clearance from shell: `10 mm`
- Electrode holder height: `100 mm`
- Tip wear band height: `20 mm`

These assumptions are isolated in code so they can be changed without altering
the UI-derived layout defaults.

## Axis transform

The source viewer uses `Y` as the vertical axis. This repo exports CAD solids
with `Z` up:

- viewer `x` -> CAD `x`
- viewer `z` -> CAD `y`
- viewer `y` -> CAD `z`, shifted so the bath floor sits at `z = 0`
