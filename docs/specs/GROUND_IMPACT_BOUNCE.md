# Ground Impact and Repeated-Bounce Specification

Status: locally implemented feature-stack slice for Tools issue #4270; not a
protected or main-branch release.

## Scope and authority

This specification defines the reusable Python reference model for a rigid
sphere striking a planar surface and completing zero or more airborne hops.
The solver consumes the strict `GroundSimulationRequest` created by #4268 and
#4269. It reuses the existing `GroundContactState`, `GroundTrajectoryPoint`,
and `GroundEvent` records, but returns `RepeatedBounceResult`: a deliberately
typed prefix rather than a fabricated final `GroundSimulationResult`.

Issue #4271 remains authoritative for skid, pure roll, rest, total distance,
and construction of the final v1 result. This slice also excludes terrain
deformation, grass interaction, rolling resistance, UI work, TypeScript
physics, Rust/PyO3/WASM kernels, and cross-application adapters.

## Units, frame, and state

All calculations use SI units in the request's target frame:
`x` downrange, `y` up, and `z` right. Position is metres, linear velocity is
metres per second, angular velocity is radians per second, impulse is
newton-seconds, energy is joules, and time is seconds. Surface normals are
unit and upward. Surface velocity must be tangential to the plane.

For ball radius `R`, mass `m`, rotational-inertia factor `k`, unit normal `n`,
and surface velocity `u_s`:

```text
I = k m R^2
r = -R n
c = v + omega cross r - u_s
c_n = c dot n
c_t = c - c_n n
```

An admissible impact requires `c_n < -1e-12 m/s`. A state within the tolerance
is rejected as `grazing`; a positive value is rejected as `outgoing`.

## Normal and tangential impulse

With normal restitution `e`:

```text
J_n = -(1 + e) m c_n
J_t,stick = -m k/(k + 1) c_t
```

The impact sticks when `|J_t,stick| <= mu_s J_n + 1e-12 N s`, including the
exact Coulomb boundary. Otherwise kinetic sliding is used:

```text
J_t = -mu_k J_n unit(c_t)
J = J_n n + J_t
v_after = v_before + J/m
omega_after = omega_before + (r cross J_t)/I
```

Postconditions verify restitution, zero residual slip for sticking, no slip
reversal for sliding, the friction cone, finite output, and exact time/frame
continuity. Rotation is fully coupled; neither initial spin nor the tangential
impulse is projected away.

## Passive energy and moving-boundary work

The immutable impact ledger records translational plus rotational kinetic
energy before and after impact, moving-boundary work, and dissipation:

```text
K = 0.5 m |v|^2 + 0.5 I |omega|^2
W_surface = J dot u_s
D = K_before + W_surface - K_after
```

`D` must be nonnegative within `1e-10 J + 1e-10` relative tolerance. A small
negative round-off value is normalized to zero; a larger violation fails
closed. A fixed surface therefore cannot add energy. A moving surface may add
energy only through its explicitly reported boundary work.

## First contact and repeated-hop state machine

The physical first contact is interpolated from the request's separated and
penetrating bracket using signed sphere-plane gap, then projected by the
round-off residual along `n`. Emitted times remain absolute. In contrast,
`max_time_s` is an elapsed ground-run duration beginning at that interpolated
first contact. `max_events` includes `FIRST_CONTACT` and every subsequent
`BOUNCE` event.

Airborne motion requires exact versioned standard gravity
`(0, -9.80665, 0) m/s^2`; custom or horizontal gravity is rejected at the
configuration boundary, keeping target-frame x-z acceleration exactly zero.
The solver uses
constant spin, and the exact positive root of
`gap(t) = v_n t + 0.5 (g dot n) t^2`. Output sampling is anchored at first
contact. Event points replace time-coincident grid points, so trajectory times
are strictly increasing and an event never creates a duplicate sample.

An incoming speed at or below `capture_speed_m_s` uses effective restitution
zero on that final micro-impact. The exact physical-contact output is emitted
once as terminal `GroundPhase.SKID`, exposed as `handoff_state`, and terminates
`SETTLED_TO_SKID`. This avoids a zero-time impact loop and gives #4271 an
unambiguous starting state.

Cancellation is checked before contact work and at every event boundary.
Bounded typed terminations are `SETTLED_TO_SKID`, `CANCELLED`, `TIME_LIMIT`,
`EVENT_LIMIT`, `NO_RECONTACT`, and `NUMERICAL_FAILURE`. Invalid request or
configuration records raise before execution. Numerical failures retain only
the already validated prefix; they never invent a summary, roll distance, or
terminal rest state.

## Bounce-distance evidence

Every completed hop and time-limited partial hop produces one immutable
`BounceAirSegment` with exact start/end time, start/end position, completion
flag, and horizontal distance. For this target frame, horizontal airborne arc
length is the accumulated x-z displacement:

```text
bounce_air_distance_m = sum(hypot(x_end - x_start, z_end - z_start))
```

Because standard gravity has no x/z component, each segment is straight in
horizontal projection and the displacement equals horizontal arc length. This
is reproducible segment evidence for #4271; it is not total shot distance.

## Material limitations

The rigid restitution law uses `normal_restitution`, `static_friction`, and
`kinetic_friction`. It intentionally does not use `firmness_pa`,
`hardness_fraction`, `grass_height_m`, `compressibility_fraction`,
`compression_damping_fraction`, `turf_density_kg_m3`, `moisture_fraction`, or
`rolling_resistance`. Every result discloses that limitation. Those fields
must not be silently presented as calibrated terrain response.

## Qualification evidence

The shared fixture
`src/rate_of_closure/web/src/model/__fixtures__/ground_impact_bounce_golden_v1.json`
locks a clean analytic sticking impact and geometric repeated-hop sequence.
Python tests cover elastic, inelastic, frictionless, sticking, exact-cone,
sliding, pure-spin, moving-boundary, and tilted-normal impacts; passivity and
friction properties; bracket and output refinement; repeated-hop analytic
times; capture; ordering; time/event limits; cancellation; no-recontact; and
numerical failure. Release still requires a normal protected carrier, hosted
checks, independent review, and downstream consumer integration.

## Repeated-bounce evidence wire boundary

`RepeatedBounceResult` is transferable as
`ground-repeated-bounce-result/v1`. The envelope adds only `schema_version`
and `unit_system` to the complete prefix record; it does not add a summary or
claim a final ground result. The exact v1 fields are `request_id`,
`surface_id`, `frame`, `model_id`, `model_version`,
`request_fingerprint_sha256`, `trajectory`, `events`, `impacts`,
`airborne_segments`, `handoff_state`, `termination`, `warnings`,
`unit_system`, and `schema_version`.

Python is the serialization authority. Its parser reuses the canonical
contact-state, trajectory-point, and event constructors, plus the immutable
impact, energy, air-segment, termination, and prefix-result constructors. The
React parser mirrors those same invariants for imported evidence only and
does not execute or approximate bounce physics. Both runtimes reject missing
or additional keys at every object level, duplicate JSON keys, non-finite or
cross-runtime-unsafe numbers, unsupported frames/units/versions, malformed
fingerprints, and inconsistent event/impact, segment, or handoff evidence.
Settled prefixes require a non-null handoff matching the terminal skid point;
termination time matches the final trajectory point and elapsed time is measured
from first contact. Energy evidence must satisfy
`D = K_before + W_boundary - K_after` within the model's documented
`1e-10 J + 1e-10` relative tolerance. Cross-record scalar and vector evidence
uses explicit `1e-10` absolute and relative tolerances in both runtimes.
Every event also requires one uniquely time-aligned trajectory point matching
its impact post-state: first contact uses `impact`, later contacts use `bounce`,
and a terminal zero-restitution contact uses `skid`. Eventful records cannot
omit trajectory evidence.

Input is bounded by UTF-8 byte length to 1 MiB. Output uses the shared
11-decimal-place canonical numeric JSON policy with lexicographically sorted
object keys. The shared golden fixture
`ground_repeated_bounce_wire_golden_v1.json` has SHA-256
`d8e7400632215220d3c5b1ccd7c57040f6023ebd72470b380b48b8f8fa99b9f9`.
This boundary deliberately performs no file persistence, ground-request
construction, regional execution, interpolation, or playback.
