# Ball-Flight Result Contract

## Purpose and Scope

`ball-flight-metrics/v1` is the reusable contract between a trajectory
integrator, result displays, exports, and desired-flight solvers. It describes
model output; it does not claim that modeled values were measured by a launch
monitor or that this project reproduces a proprietary device algorithm.

The Python implementation is in `swing_sim.flight.result_contract` and
`swing_sim.flight.result_metrics`. The TypeScript mirror is in
`ballFlightMetricContract.ts` and `ballFlightMetrics.ts`. A shared golden
fixture pins the complete catalog and analytic result hashes across clients.

## Coordinate and Sign Convention

All vectors use the right-handed target frame
`target_frame:x_downrange,y_up,z_right`:

- `x` points from the launch position toward the target line;
- `y` points up;
- `z` points right when looking downrange;
- Launch Direction and lateral/offline values are positive right;
- vertical launch angle is positive up; and
- landing angle is reported as a positive downward angle below horizontal.

Adapters from the legacy flight frame (`x` forward, `y` left, `z` up) must use
the existing frame transformation before calling this contract. Raw initial,
spin, landing-position, and landing-velocity vectors are retained so scalar
angles can be recomputed under a future display convention.

## Metric Semantics

Every row has a stable ID, label, definition, canonical unit, value-status
class, frame, sign rule, reference event, geometry formula, availability rule,
solver eligibility flag, provenance, and three-cell convention coverage row.

Important distinctions are contractual:

- Carry Distance is horizontal distance to the first descending ground
  crossing. It excludes bounce and roll.
- Carry Offline is the signed lateral coordinate at that crossing. It is not
  Curve.
- Curve is the largest signed lateral departure from the vertical plane
  established by the initial horizontal velocity.
- Apex Height is the maximum available trajectory sample height. The contract
  does not silently fit an unvalidated apex between samples.
- Total Distance, Roll Distance, Bounce Count, and Final Offline are numeric
  only when an identified qualified ground model supplies them. Otherwise each
  is `unavailable` with reason `ground_model_required`.
- Target Residual is the three-dimensional miss at first ground contact;
  downrange and lateral components remain separately available.

The first descending ground crossing is linearly interpolated between the two
bracketing trajectory states, including time, position, and velocity. Launch at
ground level is not misidentified as landing: the trajectory must first contain
an airborne point. A missing crossing and an insufficient trajectory are
separate typed unavailable states.

## Status, Availability, and Solver Selection

The value-status vocabulary is `input`, `directly_simulated`, `derived`,
`model_dependent`, `estimated`, `optimized`, `unsupported`, and `unavailable`.
The current derivation produces input, derived, model-dependent, and
unavailable values; the larger vocabulary prevents later estimation or solver
output from being mislabeled as direct simulation.

An unavailable value contains no numeric substitute. Its reason is one of:

- `insufficient_trajectory`;
- `no_ground_crossing`;
- `zero_horizontal_speed`;
- `zero_spin`;
- `target_not_configured`; or
- `ground_model_required`.

Solver interfaces may offer only definitions whose `solver_objective` flag is
true and whose current result/model pipeline can produce the metric. The flag
means that the quantity is meaningful as an objective, not that it is always
available.

## Run Manifest and Export

Every result carries the model ID and version, integration status, termination
reason, frame ID, environment fields, wind fields, and uncertainty status.
Python and TypeScript serialize sorted keys and stable metric order. This makes
exports suitable for hashing and regression evidence without relying on object
insertion order.

## Convention Coverage and Legal Boundary

The coverage matrix has explicit `app_native`, `trackman_comparable`, and
`foresight_comparable` cells. `definition_aligned` means only that a modeled
value is organized under a public parameter definition; its reason remains
`modeled_not_measured`. Raw vectors, target residuals, and other fields for
which the public material does not establish a direct device result are
`not_comparable` with `public_definition_not_established`.

Product and company names are used solely to describe public interoperability
conventions. They do not imply affiliation, certification, endorsement, or
identical results.

Primary public definition sources:

- [TrackMan parameter definitions](https://www.trackman.com/blog/golf/40-trackman-parameters)
- [Foresight ball launch and flight definitions](https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results)

## Integration Boundaries

This foundation does not yet wire the catalog into every React/PyQt result row,
the HTTP API, or Rust/WASM. Those consumers should adapt to this contract rather
than introduce another metric dictionary. Rust/WASM parity requires a separate
adapter because its current flight result does not carry the complete manifest,
target residuals, typed unavailable states, or qualified ground outputs.
