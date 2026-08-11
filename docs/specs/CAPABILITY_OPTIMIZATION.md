# Player-Capability Optimization Contract

## Scope

The `player-capability-profile/v1` contract records a player's per-club delivery envelope without treating modeled values as measured facts. Each parameter declares:

- a hard safety interval used by optimization and ensemble clipping;
- a narrower evidence interval used to flag extrapolated recommendations;
- a baseline, systematic bias, unit, and standard deviation.

Each club supplies either a correlation matrix, combined with the parameter standard deviations, or a dimensional covariance matrix. Matrices are finite, symmetric, positive semidefinite, and ordered exactly like the club's parameter list. Club and profile confidence values are bounded to `[0, 1]`; provenance strings identify the fit session or source but are not interpreted as proof of calibration quality.

## Optimization Boundary

`optimize_capability` and `optimizeCapability` do not implement ball-flight physics. A caller injects a deterministic evaluator that returns canonical `ball-flight-metrics/v1` values. The optimizer consumes `carry_distance` and `carry_offline`, so the same flight model, target frame, wind configuration, and ground assumptions used elsewhere remain authoritative.

The bounded search alternates across the requested discrete clubs. Within each club it evaluates the declared baseline and deterministic low-discrepancy continuous candidates over all delivery parameters. Every candidate receives the same seeded low-discrepancy ensemble transformed by the declared covariance. Bias is applied before clipping to the hard safety envelope.

## Qualified Waterloo/Penner Evaluator

`make_capability_flight_evaluator` and `makeCapabilityFlightEvaluator` bind a
validated player profile and optimization request to the actual Waterloo/Penner
forward model. The established variable IDs are `ball_speed` (`m/s`),
`launch_angle` (`deg`), and `launch_direction` (`deg`, target-frame positive
right). Existing profiles may use explicit configuration defaults for total
spin and canonical target-frame spin-axis tilt, but each default is keyed by
club and requires a nonempty evidence/provenance string. There is no global
driver-spin fallback. A club may instead declare `total_spin` (`rpm`) and
`spin_axis_tilt` (`deg`) together as variable capability parameters.
Positive tilt means fade/right curvature in the target frame. Metric
provenance records whether spin was sampled or supplied by a named fixed
club default.

Unknown variables, wrong units, non-finite values, undeclared clubs, and
samples outside a club's hard bounds fail before integration. The full safe
interval must also fit the physical flight domain: ball speed is strictly
positive, total spin is nonnegative, launch angle and spin-axis tilt lie in
`[-90, 90]` degrees, and launch direction lies in `[-180, 180]` degrees.
The shared trajectory interval contract is `[0.001, 0.1]` seconds in exact
0.001-second increments; unsupported settings fail identically before either
runtime runs.

Every trajectory position, velocity, and spin vector is transformed into the
canonical target frame without adding a tee-height display offset. The request
target is supplied to `ball-flight-result/v1` derivation, so a completed run
returns all available scalar launch, landing, and target-residual metrics—not
only the carry/offline pair required by the optimizer. A completed result
requires a physical descending ground crossing with available carry and
offline values. Reaching the time horizon first is `nonconverged` with no
partial metrics. Expected Python floating-point overflow failures are
`failed` with a stable non-leaking reason. Contract and programming errors
surface instead of being silently counted as ordinary failed trials. This
post-impact launch evaluator cannot fabricate `no_impact`; contact-aware
evaluators own that status.

The logical coefficient model is versioned as
`waterloo-penner-coefficients/v1`. Python uses adaptive SciPy RK45 and React
uses fixed-step RK4; each runtime records its actual integrator in provenance.
Metric sets, frames, signs, and typed statuses are parity contracts, while
numerical comparisons use the same published tolerance bands as the flight
explorer rather than claiming bitwise integrator equivalence.

## Objectives And Diagnostics

The v1 request supports:

- `maximize_carry`;
- `minimize_expected_miss` from the configured target center;
- `maximize_target_hold` using the shared green/fairway target geometry;
- `minimize_variability`, ranked by RMS two-dimensional landing dispersion about the ensemble mean rather than target miss;
- `minimize_downside`, ranked by the sum of worst-tail miss-distance CVaR and worst-tail carry shortfall relative to the target center;
- `distance_control_pareto`, comparing absolute mean-distance error and landing dispersion and explicitly marking nondominated alternatives.

Every returned alternative includes mean carry, expected miss, RMS landing dispersion, target-hold probability, miss-distance CVaR, downside carry, ensemble counts, no-impact and failure fractions, confidence, limiting constraints, extrapolation, and Pareto membership. Downside carry is the positive shortfall between the target-center distance and the mean of the lowest `(1 - cvar_alpha)` carry tail; miss CVaR is the mean of the corresponding highest miss-distance tail. Ranking is deterministic for a deterministic evaluator. Candidates below the configured minimum success fraction receive a dominating penalty but remain visible when the alternatives budget permits, preserving failure evidence.

## End-User Workflow

PyQt6 and React expose the optimizer through a primary `Shot Optimizer` module.
Both clients author and strictly validate
`capability-optimization-workflow/v1`, containing the complete profile,
request, sourced per-club fixed-spin configuration, integrator settings, and
deterministic search basis. The default document is explicitly representative
and user-authored; it is not presented as measured player data.

The persisted v1 wire contract is strict at every nested primitive. Numeric
fields accept only finite JSON numbers (and integer fields require an integral
number); text fields accept only nonempty JSON strings. Numeric strings,
booleans used as numbers, fractional integer values, and numeric identifiers or
provenance values are rejected in both runtimes. Python and TypeScript execute
one shared versioned accept/reject fixture to prevent parser drift.

Optimization runs outside the UI thread. Progress is based on attempted model
evaluations, cancellation publishes no partial optimization result, and input
changes invalidate captured output. Every attempted sample is retained in
`scalar-ensemble/v1` with complete, no-impact, or failed cohort identity. The
clients present ranked alternatives, selectable scalar axes, paired-finite and
unavailable counts, managed zoom/autofit, a bounded paged raw table,
spreadsheet-safe lossless CSV, and stable JSON. Duplicate evaluator and target
diagnostic labels are stage-qualified in selectors without changing their
contract keys.

### Whole-Workspace Input Persistence

Explorer-session v5 embeds the exact `capability-optimization-workflow/v1`
document as `model_session.data.capability_request`. The nested document is the
sole cross-runtime schema for this input specification; the workspace does not
duplicate individual optimizer fields. It contains only the profile, request,
target, evaluator configuration, search budgets, and deterministic seed needed
to reproduce a request.

Whole-workspace parsing completes before either UI mutates. PyQt6 applies the
validated request inside the window's rollback boundary, while React applies
the projected editable inputs from the validated document. Both invalidate any
previous computed output. Explorer-session v1-v4 migration requires an
explicit current capability fallback and never invents an optimizer request.
Ranked alternatives, observation ensembles, progress, cancellation/runtime
objects, and inferred player identity are excluded. The current interactive
workflow declares still air and has no editable wind input; workspace restore
must not fabricate wind-aware optimization or execution parity.

## Interpretation And Limitations

- Results are conditional model recommendations, not measured launch-monitor results or guarantees.
- Confidence is the product of profile confidence, club confidence, successful-trial fraction, and a declared extrapolation discount. It is a transparent engineering indicator, not a calibrated probability of real-world success.
- The deterministic low-discrepancy ensemble is reproducible and correlation-aware; it is not a replacement for convergence studies or posterior sampling.
- Clipping enforces hard delivery bounds but can distort tail covariance near a bound. The result reports the limiting safety or evidence boundary so callers can identify that condition.
- Target hold is landing containment only. Roll, turf, hazards, weather uncertainty, and strategic utility require evaluators or higher-level policies that explicitly model them.
- V1 searches continuous delivery parameters uniformly over declared safety bounds. Adaptive optimization and player-specific priors remain deferred.
- The matched PyQt6/React workflow is implemented. Issue #4197 still requires protected CI, ordered review/merge, and downstream application parity before closure.

## Cross-Runtime Parity

Python and TypeScript share `capability_optimizer_golden_v1.json`. It pins the
profile, request, selected club, mean carry, target-hold probability,
variability score, and downside-tail score for an analytic evaluator. They
also share `capability_flight_evaluator_parity_v1.json`, which bands all 16
available scalar metrics for one pinned Waterloo/Penner launch. Runtime tests
cover both spin-tilt signs, per-club default provenance, all physical domains,
coarse-but-supported sampling, typed edge states, all six objectives, and
strict contract validation. Cross-producer regressions also pin the same
gyro-projected tilt calculation in canonical result derivation, impact
diagnostics, and variation output.
