# Launch-Monitor Convention Registry

## Purpose

The registry is the provenance boundary between simulation-native values and
values calculated on a convention intended to be comparable with a launch
monitor. It does not claim to reproduce proprietary device algorithms or turn
a modeled quantity into a measurement.

The catalog covers 28 reconciled parameters across 5 logical groups:
- **Club Delivery**: `club_speed`, `club_path`, `attack_angle`, `dynamic_lie`, `closure_rate`, `swing_direction`, `low_point`
- **Face Orientation & Impact**: `face_angle`, `dynamic_loft`, `face_to_path`, `spin_loft`, `impact_offset`, `impact_height`
- **Ball Launch**: `ball_speed`, `launch_angle`, `launch_direction`, `smash_factor`
- **Ball Spin**: `total_spin`, `spin_axis`, `back_spin`, `side_spin`
- **Ball Flight**: `apex_height`, `carry_distance`, `total_distance`, `carry_offline`, `curve`, `flight_time`, `landing_angle`

Each parameter appears under three explicit convention IDs:
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

Changing a rigid body's reference point uses:

```text
v_point = v_reference + omega x r_reference_to_point
```

Changing coordinate frames requires a finite, proper orthonormal rotation.
The transform helpers reject reflections, scaling, shear, and nonfinite input.
No event-time interpolation is invented by the registry; a caller must supply
a separately validated time transformation before comparing different events.

## Full Parameter Coverage Matrix

| Parameter ID | Label | Group | Unit | TrackMan Policy | Foresight Policy | Comparability Verdict |
|---|---|---|---|---|---|---|
| `club_speed` | Club Speed | Club Delivery | m/s | Geometric center, pre-contact, measured | Face center, pre-contact, measured | Incompatible (`reference_point`) |
| `club_path` | Club Path | Club Delivery | deg | Geometric center, max compression, measured | Face center, impact, measured | Incompatible (`reference_point`, `event_time`) |
| `attack_angle` | Attack Angle | Club Delivery | deg | Geometric center, max compression, measured | Face center, impact, measured | Incompatible (`reference_point`, `event_time`) |
| `dynamic_lie` | Dynamic Lie | Club Delivery | deg | Face center, max compression, measured | Face center, impact, measured | Incompatible (`event_time`) |
| `closure_rate` | Closure Rate | Club Delivery | deg/s | Impact location, max compression, derived | Face center, impact, derived | Incompatible (`reference_point`, `event_time`) |
| `swing_direction` | Swing Direction | Club Delivery | deg | Geometric center, pre-contact, measured | Face center, impact, unavailable | Incompatible (`availability`, `reference_point`, `event_time`) |
| `low_point` | Low Point | Club Delivery | m | Geometric center, max compression, measured | Face center, impact, unavailable | Incompatible (`availability`, `reference_point`) |
| `face_angle` | Face Angle | Face Orientation | deg | Impact location, max compression, measured | Impact location, impact, measured | Incompatible (`event_time`) |
| `dynamic_loft` | Dynamic Loft | Face Orientation | deg | Impact location, max compression, measured | Impact location, impact, measured | Incompatible (`event_time`) |
| `face_to_path` | Face to Path | Face Orientation | deg | Mixed delivery, max compression, derived | Mixed delivery, impact, derived | Incompatible (`event_time`) |
| `spin_loft` | Spin Loft | Face Orientation | deg | Mixed delivery, max compression, derived | Mixed delivery, impact, derived | Incompatible (`event_time`) |
| `impact_offset` | Impact Offset | Face Orientation | m | Impact location, max compression, measured | Impact location, impact, measured | Incompatible (`event_time`) |
| `impact_height` | Impact Height | Face Orientation | m | Impact location, max compression, measured | Impact location, impact, measured | Incompatible (`event_time`) |
| `ball_speed` | Ball Speed | Ball Launch | m/s | Ball center, post-separation, measured | Ball center, post-separation, measured | **Directly Comparable** |
| `launch_angle` | Launch Angle | Ball Launch | deg | Ball center, post-separation, measured | Ball center, post-separation, measured | **Directly Comparable** |
| `launch_direction` | Launch Direction | Ball Launch | deg | Ball center, post-separation, measured (+ right) | Ball center, post-separation, measured (unspecified sign) | Incompatible (`sign_rule`) |
| `smash_factor` | Smash Factor | Ball Launch | ratio | Mixed delivery, post-separation, derived | Mixed delivery, post-separation, derived | **Directly Comparable** |
| `total_spin` | Total Spin | Ball Spin | rpm | Ball center, post-separation, measured | Ball center, post-separation, measured | **Directly Comparable** |
| `spin_axis` | Spin Axis | Ball Spin | deg | Ball center, post-separation, measured | Ball center, post-separation, measured | **Directly Comparable** |
| `back_spin` | Back Spin | Ball Spin | rpm | Ball center, post-separation, measured | Ball center, post-separation, measured | **Directly Comparable** |
| `side_spin` | Side Spin | Ball Spin | rpm | Ball center, post-separation, measured | Ball center, post-separation, measured | **Directly Comparable** |
| `apex_height` | Apex Height | Ball Flight | m | Ball center, apex, modeled | Ball center, apex, modeled | **Directly Comparable** |
| `carry_distance` | Carry Distance | Ball Flight | m | Ball center, landing, modeled | Ball center, landing, modeled | **Directly Comparable** |
| `total_distance` | Total Distance | Ball Flight | m | Ball center, landing, modeled | Ball center, landing, modeled | **Directly Comparable** |
| `carry_offline` | Carry Offline | Ball Flight | m | Ball center, landing, modeled | Ball center, landing, modeled | **Directly Comparable** |
| `curve` | Curve | Ball Flight | m | Ball center, landing, modeled | Ball center, landing, unavailable | Incompatible (`availability`) |
| `flight_time` | Flight Time | Ball Flight | s | Ball center, flight duration, modeled | Ball center, flight duration, modeled | **Directly Comparable** |
| `landing_angle` | Landing Angle | Ball Flight | deg | Ball center, landing, modeled | Ball center, landing, modeled | **Directly Comparable** |

## Ambiguity Register and Limitations

Where public sources leave definitions under-specified, the catalog enforces fail-closed comparability rules:

1. **Foresight Launch Direction Sign**: Foresight's public documentation describes launch direction in qualitative push/pull and left/right terms without defining a universal signed scalar independent of player handedness. TrackMan defines positive-right unconditionally. The Foresight sign rule is therefore set to `unspecified`, preventing direct subtraction deltas.
2. **TrackMan vs Foresight Club Speed Reference**: TrackMan radar resolves club head motion at the geometric center of the club head. Foresight photometric systems measure fiducial markers to report velocity at the face center. On off-center strikes, gear-effect and head yaw/pitch create a physical velocity difference of ~1.5 to 2.5 mph between these locations. Direct subtraction without rigid-body offset transformation is prohibited.
3. **Event Timing Mismatch**: TrackMan reports club face orientation and path at maximum compression (approximately 0.25 ms into impact). Foresight reports delivery at first contact/impact entry frame. Dynamic face closure and shaft deflection during the impact interval mean these quantities differ systematically.
4. **Trajectory Modeling Provenance**: Both TrackMan and Foresight report carry, apex, and landing parameters as modeled/calculated trajectories, but TrackMan utilizes continuous Doppler tracking downrange while Foresight models trajectory from measured launch conditions with aerodynamic drag/lift coefficients.
5. **Unavailable Public Quantities**: Foresight does not publish separate quantitative metrics for `curve` (lateral offset from initial launch azimuth) or `swing_direction` (base of 3D swing plane). These parameters are designated `unavailable` under Foresight-Comparable.

## Side-by-Side Comparison Workspace

The comparison workspace (PyQt `LaunchMonitorComparisonWorkspace` and React `LaunchMonitorComparisonWorkspace.tsx`) enables side-by-side analysis of TrackMan-Comparable and Foresight-Comparable values:
- **Signed Delta Contract**: If two quantities are comparable, the signed difference (`TrackMan - Foresight`) is calculated and displayed with appropriate units. If non-equivalent, the table renders the typed comparability mismatch reason, preventing fabricated deltas.
- **Search and Category Filtering**: Filter rows by logical parameter groups (`Club Delivery`, `Face Orientation`, `Ball Launch`, `Ball Spin`, `Ball Flight`) or search by parameter name, ID, and definition.
- **Export & Serialization**: One-click deterministic JSON and CSV exports formatted for reporting and downstream verification.
- **Accessibility**: Full keyboard navigation, focus indicators, tooltip discoverability, and screen-reader accessibility labels across PyQt and React.

## Primary Sources

- [TrackMan club-data definitions](https://www.trackman.com/blog/golf/club-data-definitions)
- [TrackMan parameter definitions](https://www.trackman.com/blog/golf/40-trackman-parameters)
- [Foresight club-head definitions](https://help.foresightsports.com/hc/en-us/articles/47214673873811-Club-Head-Data-Measurements-Definitions)
- [Foresight ball-launch definitions](https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results)

## Serialization and Parity

`launch-monitor-conventions/v1` is strict: missing or extra definition fields are rejected. The only supported legacy migration is the explicit v0 `vendor` field rename to `convention_id`; unknown versions fail closed.

Python and TypeScript serialize the full catalog with sorted object keys and a stable definition order. Both clients verify the same SHA-256 checksum from the shared fixture, so parity covers every field across all 28 parameters and 3 convention families.
