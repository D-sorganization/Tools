# Desired Ball-Flight Inverse Solver

## Purpose and Boundary

`inverse-flight-request/v1` specifies desired or optimized canonical ball-flight
metrics independently of a particular impact or trajectory model.
`inverse-flight-result/v1` returns bounded-search evidence rather than a claim
that one unique swing or impact condition exists. Python consumers import from
`swing_sim.flight`; TypeScript consumers use `inverseFlightContract.ts` and
`inverseFlightSolver.ts`.

The solver accepts an injected deterministic forward evaluator. This keeps the
contract reusable for direct launch-condition inputs, impact-derived inputs,
wind scenarios, qualified ground models, and future capability constraints.
The evaluator owns physics and reports one of `complete`, `no_impact`, `failed`,
or `nonconverged`. An incomplete evaluation never supplies substitute metrics.

## Request Contract

Each decision variable has a stable ID, display unit, finite lower/upper bounds,
and an initial value within those bounds. The solver never evaluates outside
this box. Each objective references a scalar metric marked `solver_objective`
in `ball-flight-metrics/v1` and must use that metric's exact canonical unit.

Objective modes are:

- `target`: approach `target_value`; the candidate satisfies the desired value
  when its absolute normalized residual is at most one;
- `maximize`: prefer a larger value; and
- `minimize`: prefer a smaller value.

`tolerance` normalizes each metric before weighting and, for `target`, defines
the satisfaction band. Optional lower and upper bounds are hard feasibility
constraints for every mode. `weight` is positive and controls tradeoffs between
unlike metrics after tolerance normalization. Objective metric IDs and decision
variable IDs must be unique.

## Search and Ranking

The v1 implementation evaluates the declared initial point followed by unique
points from a deterministic Halton sequence. The configured evaluation count is
a hard upper bound. All returned parameter points therefore remain within the
declared closed box and repeatable forward evaluators produce byte-identical
results.

Candidates are ranked by:

1. feasible before infeasible;
2. total normalized hard-bound violation;
3. weighted objective score; and
4. evaluation index as a deterministic tie break.

Target score uses the absolute normalized residual. Maximize uses the negative
normalized value and minimize uses the positive normalized value, so lower is
always preferred. Hard-bound violations receive a fixed dominant penalty and
are also ordered independently before the score. Every residual retains actual
value, unit, mode, target when applicable, normalized residual, constraint
violation, and forward-model provenance.

## Terminal Status

- `solved`: the highest-ranked returned candidate satisfies every target
  tolerance and hard bound;
- `infeasible`: a target is statically outside its own declared hard bounds, so
  no forward evaluations are attempted;
- `no_impact`: every attempted parameter point reports no club-ball impact; and
- `nonconverged`: no complete feasible solution was established within the
  evaluation budget, including forward failures and integrator nonconvergence.

A finite bounded sample cannot prove that a dynamically feasible solution does
not exist. Therefore an unsuccessful search is `nonconverged`, not
`infeasible`. Diagnostic counts expose attempted, complete, no-impact, and
failed/nonconverged evaluations without conflating those cases.

## Determinism and Provenance

Both runtimes use sorted JSON keys, stable array order, and eleven-decimal wire
rounding. The shared fixture
`inverse_flight_solver_golden_v1.json` pins an analytic solve by SHA-256. Result
provenance identifies the canonical metric schema, sampler, solver ID, and
solver version. Metric residual provenance comes from the injected evaluator.

## Deferred Integration

This slice intentionally does not include PyQt6 or React controls, a concrete
impact-to-flight adapter, gradients, local refinement, uncertainty propagation,
capability envelopes, or persistent solve libraries. The spatial-target
contract from issue #4192 is not required by this foundation; target residual
metrics can be supplied by a later forward adapter without merging or rewriting
that sibling branch.
