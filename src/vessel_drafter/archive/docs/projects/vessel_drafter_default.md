# Vessel Drafter Default

This project packages the requested molten-glass vessel as both a STEP export and
an interactive PyQt6 drafting tool.

## Default Geometry

- Inner bath diameter: `50 in`
- Glass depth: `14 in`
- Plenum height: `14 in`
- Hot face refractory: `6 in`
- IFB: `4.5 in`
- Duraboard: `1 in`
- Steel shell: `0.5 in`
- Electrodes: `3`, radial, `120 deg` apart
- Electrode diameter: `2 in`
- Electrode insertion into inner circle: `14 in`
- Electrode extension outside inner circle: `36 in`

## Modeling Notes

- The top and bottom heads use offset elliptical profiles so the refractory and
  shell layers remain constant thickness through the head sections.
- Only the plenum above the glass bath is modeled as the internal void.
- The bottom head is layered solid support beneath the flat-bottom glass bath.
- STEP solids are exported with color metadata for glass, refractory, IFB,
  duraboard, steel, and electrodes.
- Sidewall ports are radial cuts defined by clock angle, diameter, and height
  above the glass surface.
- Lid ports are vertical cuts defined by clock angle, diameter, and distance
  from the vessel center.

## GUI Scope

- Editable vessel, layer, and electrode dimensions.
- Add/remove sidewall ports and lid ports from the editor.
- Live cross-section preview for shell stackup, plenum, and projected port
  heights.
- Live top-view preview for radial electrode placement and port placement.
- Separate 3D preview tab with mouse rotation, per-layer visibility toggles,
  and a user-rotated vertical section cut.
- Material summary table with computed component volumes in `ft^3`, surface
  areas in `ft^2`, refractory totals, and default density and thermal-property
  assumptions.
- STEP + JSON export from the current form state.
