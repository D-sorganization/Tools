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

## Objectives And Diagnostics

The v1 request supports:

- `maximize_carry`;
- `minimize_expected_miss` from the configured target center;
- `maximize_target_hold` using the shared green/fairway target geometry;
- `minimize_variability`, ranked by RMS two-dimensional landing dispersion about the ensemble mean rather than target miss;
- `minimize_downside`, ranked by the sum of worst-tail miss-distance CVaR and worst-tail carry shortfall relative to the target center;
- `distance_control_pareto`, comparing absolute mean-distance error and landing dispersion and explicitly marking nondominated alternatives.

Every returned alternative includes mean carry, expected miss, RMS landing dispersion, target-hold probability, miss-distance CVaR, downside carry, ensemble counts, no-impact and failure fractions, confidence, limiting constraints, extrapolation, and Pareto membership. Downside carry is the positive shortfall between the target-center distance and the mean of the lowest `(1 - cvar_alpha)` carry tail; miss CVaR is the mean of the corresponding highest miss-distance tail. Ranking is deterministic for a deterministic evaluator. Candidates below the configured minimum success fraction receive a dominating penalty but remain visible when the alternatives budget permits, preserving failure evidence.

## Interpretation And Limitations

- Results are conditional model recommendations, not measured launch-monitor results or guarantees.
- Confidence is the product of profile confidence, club confidence, successful-trial fraction, and a declared extrapolation discount. It is a transparent engineering indicator, not a calibrated probability of real-world success.
- The deterministic low-discrepancy ensemble is reproducible and correlation-aware; it is not a replacement for convergence studies or posterior sampling.
- Clipping enforces hard delivery bounds but can distort tail covariance near a bound. The result reports the limiting safety or evidence boundary so callers can identify that condition.
- Target hold is landing containment only. Roll, turf, hazards, weather uncertainty, and strategic utility require evaluators or higher-level policies that explicitly model them.
- V1 searches continuous delivery parameters uniformly over declared safety bounds. Adaptive optimization, player-specific priors, UI authoring, persistence, and live launch-monitor fitting are deferred extensions.

## Cross-Runtime Parity

Python and TypeScript share `capability_optimizer_golden_v1.json`. It pins the profile, request, selected club, mean carry, target-hold probability, variability score, and downside-tail score for an analytic evaluator. Runtime tests exercise all six objectives and strict contract validation.
