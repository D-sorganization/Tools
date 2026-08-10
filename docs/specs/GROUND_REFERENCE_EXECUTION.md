# Ground Reference Execution

## Scope

`ground-reference-execution/v1` defines the immutable controls for one-shot
execution of the qualified planar ground model. The Python implementation
remains the scientific reference authority: it runs the existing
repeated-impact solver, passes its exact settled contact state into the existing
skid/roll solver, and uses the existing result composer. The compiled
`tools-core` implementation independently executes the same qualified
contact/bounce/skid/roll/rest sequence for its explicitly narrower static-plane,
resolver-free scope and emits the existing canonical
`flight-to-ground-result/v1` wire contract.

The public call is:

```python
run_ground_reference(request, execution=None) -> GroundSimulationResult
```

Compiled callers use native `run_ground_reference_v1`, strict JSON
`run_ground_reference_v1_json`, PyO3
`run_flight_to_ground_reference_v1`, or wasm-bindgen
`runFlightToGroundReferenceV1`. All compiled surfaces share one parser,
normalized request authority, runtime, typed error record, and canonical result
encoder; binding layers do not reimplement physics.

`GroundReferenceExecution` is immutable and contains the exact
`BounceModelSettings`, exact `SkidRollSettings`, an optional exact
`SurfaceResolver`, and one optional cancellation callback. The same callback
instance is supplied to both phases. The callback is cooperative: it is checked
at the bounded points already defined by each phase solver, not from another
thread inside an integration step.

The compiled execution record carries the versioned bounce and skid/roll
settings but no serialized callback. A non-null material or terrain resolver
fails closed because compiled v1 qualifies one immutable plane only.

## Python reference sequence

1. Validate an exact `GroundSimulationRequest` and exact execution controls.
2. Calculate the canonical request SHA-256 fingerprint.
3. Run `simulate_repeated_bounce` once.
4. Continue only from `settled_to_skid` and its validated physical handoff.
5. Run `simulate_skid_roll` once with the selected settings, resolver, and the
   same cancellation callback.
6. Continue only for a suffix termination the existing composer can encode.
7. Delegate final trajectory, event, summary, warning, status, and termination
   construction to `compose_ground_result`.

## Compiled parity sequence

1. Strictly parse and validate the request and execution wires, rejecting
   duplicate keys, unsupported identities, resolvers, and serialized callbacks.
2. Normalize the typed request once and use that exact record for fingerprint,
   preflight, and physics.
3. Prove wire-time representability and independent output, integration-step,
   event, and total-trajectory budgets before callbacks or physics.
4. Interpolate contact, resolve the first impact, and execute bounded repeated
   bounce or immediate capture.
5. Continue the exact handoff through skid, pure roll, rest, or an honest
   censored termination.
6. Compose, recursively validate, and canonically serialize the complete v1
   result; derived unsafe values return typed owning-phase errors.

The Python executor owns orchestration only. Impact impulses, airborne
propagation, friction, skid-to-roll transition, surface resolution, distance
accounting, warnings, and JSON encoding remain owned by their existing Python
modules. The compiled runtime owns separate Rust phase implementations for its
narrower domain; exact golden bytes, seeded Python parity, native tilted-plane
invariants, and real PyO3/WASM tests constrain that implementation.

Surface passivity is enforced independently on the physical mechanical balance
of every unquantized constant-motion segment. Earlier legitimate dissipation
therefore cannot mask a later energy-creating segment. The canonical
11-decimal endpoint remains the deterministic state used by the next segment
and the public result, but its exclusion from physical work is bounded by an
accumulated fixed-component quantization budget. No-slip projection is admitted
only while contact slip and the resulting velocity/spin corrections remain
inside their versioned thresholds. A final endpoint outside the accumulated
budget fails closed. Adversarial Python and Rust tests cover masking, unforced
acceleration, and unexplained endpoint energy.

Rolling resistance uses a bounded closing step when its frozen direction would
cross zero relative speed. If the versioned resistance magnitude can balance
the slope drive, a zero-relative-speed state is held and may continue to move
with a translating surface. This state is not mislabeled as absolute rest;
the balancing contact force remains in the work ledger. Any residual already
inside the velocity tolerance is projected to exact surface co-motion through
the same slip, velocity, spin, and energy bounds before holding.
The contact-slip gate continues to use `slip_tolerance_m_s`; the center and
transverse-spin correction gates use the independent `velocity_tolerance_m_s`
and `velocity_tolerance_m_s / radius`. If that projection creates a stationary
rest state, the solver terminates as `rest` in the same step. At the exact
surface handoff time it advances one zero-motion interval before emitting the
rest point because the public trajectory contract requires strictly increasing
timestamps.

## Compiled resource and cancellation contract

The synchronous compiled boundary applies independent, fail-closed budgets
before callbacks or physics and repeats capacity checks at every dynamic append:

- at most 200,001 scheduled endpoint-inclusive output points;
- at most 1,000,000 caller-authorized surface-loop steps;
- at most 10,000 declared events; and
- at most 210,003 total trajectory points, including unscheduled contact,
  phase-transition, event, and terminal evidence.

Output samples and integration steps are different dimensions. A sparse output
interval cannot authorize excessive integration work, and a small integration
step allowance does not reject a valid denser output schedule. `max_steps`
remains a runtime exhaustion cap, not a promise that the full requested horizon
will be reached; ordinary exhaustion remains typed `step_limit`. Values above
the trusted compiled ceiling fail as `integration_step_limit`, and oversized
event or trajectory work fails as `event_count_limit` or
`trajectory_point_limit`.

The absolute output grid is generated by a bounded integer index in elapsed
time. Projection to the canonical wire epoch must remain strictly increasing;
unrepresentable grids fail as `time_resolution`. Derived unsafe numeric values
fail from their owning phase as `numeric_range`; they must never panic or trap.

Native cancellation is a cooperative closure checked before execution and
inside bounce, surface, and output loops. PyO3 releases the GIL between polls
and reacquires it only to invoke the callback, preserving callback exceptions
and strict boolean results. The current WASM call is synchronous and
cooperative; asynchronous WASM cancellation is not part of v1.

## Terminal-state policy

| Phase outcome | Public behavior | Rationale |
| --- | --- | --- |
| Bounce `settled_to_skid` | Run skid/roll | Exact physical handoff exists. |
| Bounce `cancelled` | Raise `GroundReferenceCancelled` | Cancellation is operational, not a simulated result. |
| Bounce `time_limit`, `event_limit`, `no_recontact`, `numerical_failure` | Raise `GroundReferenceExecutionError` | No qualified skid handoff exists and v1 composition would require invention. |
| Skid/roll `rest`, `left_surface`, `time_limit`, `event_limit` | Compose `GroundSimulationResult` | These outcomes have an existing explicit v1 mapping. |
| Skid/roll `cancelled` | Raise `GroundReferenceCancelled` | Cancellation must not masquerade as physics. |
| Skid/roll `step_limit`, `unsupported_surface`, `numerical_failure` | Raise `GroundReferenceExecutionError` | The current public result schema has no honest mapping. |
| Composition rejection | Raise `GroundReferenceExecutionError` with `composition_error` | The caller receives typed fail-closed evidence without exposing a fabricated result. |

Both error types retain the phase, the native stable reason code, and the exact
request fingerprint. `GroundReferenceCancelled` is a subclass of
`GroundReferenceExecutionError`, allowing callers to handle all non-results
together or cancellation separately. The original composition exception is
retained as the Python exception cause.

`SkidRollTerminationReason.NUMERICAL_FAILURE` is a reserved native terminal
code that the current `simulate_skid_roll` implementation does not emit.
Current invalid numerical states raise natively and are deliberately not
normalized by this coordinator. If a future compatible suffix implementation
returns the reserved typed outcome, the executor will fail closed with
`GroundReferenceExecutionError` as shown in the table.

## Determinism and golden evidence

For an immutable request and execution record, the reference executor is
deterministic. The fixture
`ground_reference_pipeline_golden_v1.json` contains one complete resting run,
its exact request and result objects, and canonical JSON SHA-256 digests. Tests
execute the full bounce-to-skid-to-roll-to-rest path twice, require identical
canonical bytes, and reproduce the committed fixture. The fixture uses a
100 ms output interval to keep the cross-runtime artifact compact; this does not
change the skid/roll solver's versioned 1 ms internal integration step.

The digest proves byte identity only. It is not a signature, calibration
certificate, or evidence that the illustrative surface parameters match a
specific course.

## Scientific conformance corpus

`ground_reference_conformance_v1.json` is the separate scientific evidence
boundary. It references the canonical request template, replaces only declared
existing leaves, and defines six independently derived cases: shallow
bounce/capture, flat skid-to-roll, pure roll along +x, the proper active
-90-degree rotation about +y into +z, pure roll relative to a translating
plane, and zero-resistance pure roll on the immutable incline
`n=[0,sqrt(0.99),0.1]`. The seventh case reflects that incline through the xy
plane to form a seventh case at `n=[0,sqrt(0.99),-0.1]`; position and velocity follow the polar-vector
reflection while angular velocity follows the axial-vector transformation.

Each case declares its scientific basis, applicable runtimes, observable,
unit, and applicable bounded tolerance. Whitelisted checks cover exact event
sequences and terminal reasons, scalar/vector proximity, Newton restitution,
the no-slip contact constraint, and passive impact energy. No expression is
evaluated from fixture text. The Python reference, native Rust executor,
installed PyO3 wheel, and rebuilt Node/WASM package all consume this same
artifact. A runtime agreeing with another runtime is insufficient: each must
also satisfy the declared analytic oracle.

Both inclines additionally require every non-bounce center to remain one radius
from the declared plane and compare their four-second path, position, velocity,
and spin to the reflected closed-form constant-acceleration solutions.

A separate deterministic property harness uses local PRNG seed `4275` to build
20 requests from the same canonical template. It covers nonzero x-normal
components and both z-tilt signs while varying bounded radius, mass, inertia,
surface height and tangential velocity, restitution, friction, rolling
resistance, launch tangent, and spin. Python and a freshly installed PyO3
extension must emit identical canonical JSON for every request. This seeded
sample complements rather than replaces the independently analytic corpus.

The prior five-case artifact is bound to implementation commit
`9df3928a1ef32d81db2e568884ca24d8c576d49a` with raw-file SHA-256
`f7fda73e45c5c64951a9934ba126cd9edbde7f7f85843a69612f86b8ec518310`.
The six-case artifact is bound to reviewed implementation commit
`5d333a4448d6484f8c98e78c9878cb83b40aa522` with raw-file SHA-256
`502dae7cacb346e55a0624b5758efce1baf123065a45571cd3aaf2ee0045bb76`.
The local seven-case successor has raw-file SHA-256
`c1c363a8ee79b12ab2b7d9c69677e71ab8ab30ba5288c275fff8ddcd4e683465`;
its implementation commit is bound in the follow-up evidence commit. None of
these digests establishes exhaustive tilted/property coverage, performance,
calibration, changing terrain, user interfaces, or downstream release. PR
#4322 remains the exact parent carrier; hosted approval and integration remain
separate gates for this unpublished continuation.

This corpus does not replace the canonical full-result fixture. Scientific
tolerances establish physical conformance; exact canonical bytes and SHA-256
remain serialization evidence. The current analytic cases and finite seeded
sample do not qualify production surface parameters, exhaustive frames,
regional terrain, performance, or statistical uncertainty.

## Design-by-contract boundary

- Requests and nested execution controls require exact canonical types.
- Invalid configuration raises before orchestration.
- The executor never catches and reclassifies caller contract violations.
- Callback exceptions and non-boolean behavior retain the native phase
  solver's current behavior; the coordinator does not broadly catch
  `ValueError`, because doing so would conflate invalid configuration, resolver
  mismatch, callback errors, and numerical failure.
- Only native phase results are inspected; phase details are not recomputed.
- Only the existing composer may construct a successful public result.
- A non-representable native terminal state always produces a typed error and
  never a partial substitute.
- The flight-first and ground-first import orders remain supported.

## Explicit limitations

The Python reference supports the existing qualified immutable planar surface
and its optional finite tangent-axis boundary. Its default unbounded resolver
derives a deterministic tangent by projecting the least-aligned Cartesian axis
into the plane, so every valid normal has a tangent frame. Explicit finite axes
and bounds remain caller-owned. Compiled parity is intentionally narrower: one
immutable plane, standard gravity, the exact v1 model identities, and no
material/terrain resolver. Cross-runtime conformance includes the two mirrored
analytic inclines; the broader seeded property sample currently qualifies
Python and installed PyO3 only.

Neither implementation adds changing normals, regional materials, terrain
deformation, torsional-spin damping, roll-to-skid transitions, production
material presets, ensemble execution, inverse solving, UI controls, or
UpstreamDrift consumers. Cancellation, lack of recontact, unsafe numerics,
unsupported terrain, and step exhaustion remain typed non-success outcomes.

The executor advances ground-model issues #4273 and #4275 but does not complete
the ground-model epic #4267. User-facing execution and rendering remain issue
#4274 work; downstream UpstreamDrift integration remains issue #4276 work.
