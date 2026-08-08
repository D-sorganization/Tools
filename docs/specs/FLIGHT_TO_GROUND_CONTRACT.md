# Flight-to-Ground Contract v1

## Scope

The `swing_sim.ground` package owns the reusable boundary between a ball-flight
integrator and a qualified bounce, skid, and roll model. It does not implement
ground physics. The v1 boundary is deliberately strict so a later solver cannot
silently substitute flat terrain, launch spin, or carry distance when the
required physical state is unavailable.

## Coordinate and unit contract

Every request and result declares `unit_system: "SI"` and the only accepted v1
frame is `target_frame:x_downrange,y_up,z_right`. Positions are metres, linear
velocities are metres per second, angular velocities are signed radians per
second, masses are kilograms, time is seconds, and pressure is pascals.

All finite floating-point fields normalize to the shared canonical numeric
policy: 11 decimal places, half-away rounding, fixed-point tokens, integer
spelling for integer-valued magnitudes, and normalized negative zero. JSON
Schema integer values such as `64.0` normalize to integer records rather than
creating a schema/runtime disagreement. Integer fields are bounded to the
cross-runtime safe range, surrogate code points and duplicate object keys are
rejected, noncanonical leading/trailing whitespace is not trimmed, and physical
plus cross-field relational bounds are checked before numeric normalization.
Strictly positive fields require at least the canonical `1e-11` quantum.

Frame conversion is an adapter responsibility. The canonical target-frame
origin is the ball centre at the launch or tee event and must be represented as
exactly `(0, 0, 0)`. The contract never guesses or infers that origin. The
existing flight frame maps vectors into the target frame as `(x, z, -y)`;
positions additionally require the caller to translate the launch/tee origin
to zero. Ground solvers preserve that common origin in every trajectory and
event position.

## Physical contact handoff

A request supplies the last separated state and first penetrating state. For a
planar surface, the signed sphere gap is

`gap = dot(ball_center - plane_origin, surface_normal) - ball_radius`.

The states must have increasing times and straddle zero gap. Their relative
normal velocities must both be strictly incoming. Moving v1 planes may translate
only tangentially; normal surface motion is rejected because v1 has no plane
reference epoch from which to calculate a time-dependent gap. This makes the
contact interpolation method a solver choice while preventing a ball-centre
crossing of an arbitrary launch plane from being mislabeled as physical sphere
contact.

The complete signed angular-velocity vector is mandatory. Current flight
surfaces that only expose scalar spin or initial launch spin are not ready to
construct this request; they must report typed unavailability in their adapter
work rather than fabricate terminal angular state.

## Surface profile

The profile identifies its provider and version and preserves the full material
inputs required by the planned UpstreamDrift one-way adapter: restitution,
static and kinetic friction, rolling resistance, firmness, hardness, grass
height, compressibility, compression damping, turf density, moisture, surface
velocity, plane height, and plane normal. Fraction fields are bounded to
`[0, 1]`; dimensional fields carry unit suffixes and fail on non-finite values.

## Result semantics

`GroundSimulationResult` separates the following quantities:

- carry distance to first physical contact;
- airborne distance accumulated during bounces;
- skid distance;
- pure-roll distance;
- accumulated surface path length;
- final downrange and offline coordinates; and
- launch-to-final horizontal total distance.

Carry is the horizontal displacement from the required launch/tee origin to
first physical contact. Total distance is the horizontal displacement from that
same origin to the final point; it is not accumulated path length. `bounce_count`
counts only ledger events typed `bounce` after `first_contact`. The initial
contact is excluded even when its impulse launches the first airborne bounce.

These values are not aliases. Failed or unavailable results cannot contain a
trajectory, events, or summary. Complete and partial results require an ordered
trajectory and typed termination. The one-way `to_ground_model_result` adapter
projects only a complete qualified result into the older metric DTO.

Every contact/bounce event preserves signed linear and angular velocity vectors
before and after the discontinuity. First-contact and terminal event outputs are
bound to their trajectory points. An unavailable result carries one or more
unique typed field/reason/provenance records, including explicit terminal-spin,
physical-contact-bracket, and surface-profile identities; other statuses reject
those records.

## Wire compatibility and migration policy

The canonical versions are `flight-to-ground-request/v1` and
`flight-to-ground-result/v1`. Serialization is compact, key-sorted JSON, and
every nested parser rejects missing or unknown fields. v1 is the first version;
there is no predecessor that can be migrated without inventing a frame or
physical state. Unsupported versions therefore fail closed. Future migrations
must be explicit, deterministic, loss-audited transforms into these canonical
records rather than permissive parser branches.

## Flight transfer implementation

Issue #4269 supplies the first strict flight-side adapters in Python,
TypeScript, Rust, PyO3, and WASM. The built-in flight models preserve a signed
three-dimensional angular-velocity vector at every emitted sample, integrate to
the configured launch-relative physical sphere/plane gap, and transfer the last
separated plus first penetrating states into the v1 request. A zero-gap
interpolated contact is a valid first-penetrating state under the v1 `gap <= 0`
rule. Rust additionally preserves its raw post-crossing sample in transfer-event
evidence; Python and TypeScript expose the exact contact state instead. The
trajectory contract does not promise that every runtime retains a raw
post-crossing sample.

Native flight integration requires explicit launch-relative plane geometry.
The launch or tee ball centre remains the exact target-frame origin; terrain
height, surface normal, ball radius, and vertical tee height therefore affect
the terminal gap without canceling one another. Tee height is measured vertically
from the ground plane to the ball bottom, not along a sloped surface normal.
Adapters reject missing origin evidence, non-increasing samples, absent terminal
angular state, grazing contact, and trajectories without a qualified physical
contact bracket using typed unavailable outcomes.

The Rust-owned PyO3 and WASM entry points parse the complete strict v1 request;
they do not replace material, calibration, provenance, provider identity, or
ball data with reduced plane defaults. The older ground-only Rust flight API is
retained as a compatibility path only where it can preserve the requested
physics; unsupported tee transfer fails closed.

This transfer still does not implement bounce, skid, or roll. UpstreamDrift
terrain remains a one-way UpstreamDrift-to-Tools adapter concern; Tools must
never import UpstreamDrift.
