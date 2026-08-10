# Ground Reference Execution

## Scope

`ground-reference-execution/v1` is the canonical one-shot Python boundary for
the qualified planar ground model. It runs the existing repeated-impact solver,
passes its exact settled contact state into the existing skid/roll solver, and
uses the existing result composer. It does not copy phase physics, relabel an
internal solver outcome, or fabricate a terminal state that
`ground-result/v1` cannot represent.

The public call is:

```python
run_ground_reference(request, execution=None) -> GroundSimulationResult
```

`GroundReferenceExecution` is immutable and contains the exact
`BounceModelSettings`, exact `SkidRollSettings`, an optional exact
`SurfaceResolver`, and one optional cancellation callback. The same callback
instance is supplied to both phases. The callback is cooperative: it is checked
at the bounded points already defined by each phase solver, not from another
thread inside an integration step.

## Execution sequence

1. Validate an exact `GroundSimulationRequest` and exact execution controls.
2. Calculate the canonical request SHA-256 fingerprint.
3. Run `simulate_repeated_bounce` once.
4. Continue only from `settled_to_skid` and its validated physical handoff.
5. Run `simulate_skid_roll` once with the selected settings, resolver, and the
   same cancellation callback.
6. Continue only for a suffix termination the existing composer can encode.
7. Delegate final trajectory, event, summary, warning, status, and termination
   construction to `compose_ground_result`.

The executor owns orchestration only. Impact impulses, airborne propagation,
friction, skid-to-roll transition, surface resolution, distance accounting,
warnings, and JSON encoding remain owned by their existing modules.

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

This v1 executor is Python-only and supports the existing qualified immutable
planar surface plus its optional finite tangent-axis boundary. It does not add
changing normals, material regions, terrain deformation, torsional-spin
damping, roll-to-skid transitions, production material presets, ensemble
execution, inverse solving, UI controls, Rust/WASM parity, or UpstreamDrift
consumers. It does not turn cancellation, lack of recontact, numerical failure,
unsupported terrain, or numerical step exhaustion into a result.

The executor advances ground-model issues #4273 and #4275 but does not complete
the ground-model epic #4267. User-facing execution and rendering remain issue
#4274 work; downstream UpstreamDrift integration remains issue #4276 work.
