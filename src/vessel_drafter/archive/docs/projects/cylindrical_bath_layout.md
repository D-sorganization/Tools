# Cylindrical Bath Layout

This project captures a cylindrical bath and refractory envelope for rapid
inspection in FreeCAD.

## Requested Geometry

- Inner bath diameter: `50 in`
- Bath depth: `14 in`
- Refractory thickness: `6 in`
- Electrode count: `3`
- Electrode spacing: `120 deg`
- Electrode diameter: `2 in`
- Electrode insertion past inner wall: `14 in`
- Electrode extension outside inner wall: `36 in`

## Assumptions

- CAD units are millimeters, with the source request preserved in inches.
- The bath axis is aligned with `+Z`.
- Electrodes are centered vertically at mid-depth because no alternate height was
  specified.
- The conflicting note about `15 in` sticks is ignored in favor of the explicit
  `14 in` insertion and `36 in` extension values.
