# Launch-Monitor Convention Registry

## Purpose

The registry is the provenance boundary between simulation-native values and
values calculated on a convention intended to be comparable with a launch
monitor. It does not claim to reproduce proprietary device algorithms or turn
a modeled quantity into a measurement.

The initial catalog covers club speed, club path, attack angle, face angle,
dynamic loft, face-to-path, three-dimensional spin loft, and launch direction.
Each appears under three explicit convention IDs:

- `app_native`
- `trackman_comparable`
- `foresight_comparable`

Product and company names identify public calculation conventions for
interoperability and comparison. They do not imply affiliation, certification,
endorsement, or identical output from a commercial device.

## Required Metadata

Every parameter definition includes:

- a stable convention-qualified identifier and display label;
- primary-source URL and retrieval date;
- physical reference point and event-time policy;
- coordinate-frame identifier, geometry contract, and signed-direction rule;
- canonical unit, availability requirement, and quantity status.

The quantity status distinguishes values derived or modeled by this project
from values intended to be comparable with a measured device field. UI and
exports must preserve that status.

## Comparison Contract

Two values may be subtracted directly only when their parameter, reference
point, event time, frame, geometry, sign rule, unit, and availability contracts
match. A mismatch is a typed result, not a warning hidden in prose.

Changing a rigid body's reference point uses

```text
v_point = v_reference + omega x r_reference_to_point
```

Changing coordinate frames requires a finite, proper orthonormal rotation.
The transform helpers reject reflections, scaling, shear, and nonfinite input.
No event-time interpolation is invented by the registry; a caller must supply
a separately validated time transformation before comparing different events.

## Provenance Notes

The TrackMan-comparable policies use TrackMan's public club-data and parameter
definition pages. Those sources distinguish geometric-center club speed/path/
attack-angle policies from face values evaluated at the impact location and
maximum-compression event.

The Foresight-comparable policies use Foresight Sports' public club-head and
ball-launch definition pages. Where a public source does not provide enough
detail to establish direct equivalence, the catalog retains a distinct datum or
event and the comparison contract reports the mismatch. In particular, the
general ball-launch definition describes horizontal azimuth but does not define
one handedness-independent numeric sign. Its launch-direction sign is therefore
`unspecified`, and direct signed comparison fails closed instead of assuming
TrackMan's absolute positive-right convention.

Primary sources:

- [TrackMan club-data definitions](https://www.trackman.com/blog/golf/club-data-definitions)
- [TrackMan parameter definitions](https://www.trackman.com/blog/golf/40-trackman-parameters)
- [Foresight club-head definitions](https://help.foresightsports.com/hc/en-us/articles/47214673873811-Club-Head-Data-Measurements-Definitions)
- [Foresight ball-launch definitions](https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results)

## Serialization and Parity

`launch-monitor-conventions/v1` is strict: missing or extra definition fields
are rejected. The only supported legacy migration is the explicit v0 `vendor`
field rename to `convention_id`; unknown versions fail closed.

Python and TypeScript serialize the full catalog with sorted object keys and a
stable definition order. Both clients verify the same SHA-256 checksum from the
shared fixture, so parity covers every field rather than representative rows
alone.
