# Conditional Chip-Shot Forgiveness Analysis

## Purpose and Claim Boundary

This feature compares wedge-chip outcomes for one declared variation plan,
club candidate, target, turf profile, solver, and seed. A result may be called
"more forgiving" only within that declared population and objective. It is not
a universal player recommendation, a manufacturer claim, or a calibrated
prediction of a named playing surface.

Illustrative and uncalibrated turf profiles never authorize turf-supported
rankings. The UI displays that restriction alongside every result. The current
turf model is a reduced effective-mass diagnostic and does not replay the
retained swing under the computed turf wrench.

## Frames, Units, and Population

- App frame: `x` toward target, `y` up, `z` right; right-handed.
- Ground frame: the selected planar datum and unit normal.
- Internal units: SI. The worked target defaults to 30 yd = 27.432 m.
- Population: the retained joint Monte-Carlo ensemble from one complete v2
  `VariationPlan`. Rows are independent seeded draws; variables within a row
  may be independent or use a declared correlation/covariance group.
- Inference boundary: the displayed Wilson and bootstrap intervals apply only
  to those independent Monte-Carlo rows. One-at-a-time sensitivity runs,
  paired interventions, deterministic grids, and repeated observations are
  separate descriptive analyses and are never passed to this summarizer.
- Reproducibility metadata: plan schema, seed, stable noise IDs, candidate ID,
  objective ID, turf profile/calibration state, solver ID, sampling-design and
  inference-method IDs, frame, and explicit limitations.

## Mutually Exclusive Trial Cohorts

Every configured trial occupies exactly one cohort:

1. ball first;
2. ball only;
3. ground first;
4. simultaneous or grazing;
5. ground only with ball missed;
6. neither ball nor ground contact;
7. numerical/model failure.

All probabilities, expected loss, constraint rates, convergence prefixes, and
tail risk use the configured trial count as their denominator. A miss or
failure is never dropped, converted to a zero, or given a fabricated landing.
Optional physical metrics retain `null`/`None`, plus explicit support and
unavailable counts.

## Physical Metrics

Successful retained runs can contribute:

- carry, lateral landing, apex, and landing angle;
- leading-edge clearance at ball contact, minimum pre-ball clearance,
  ball-to-ground time margin, and low-point clearance;
- delivered bounce, path-projected effective bounce, reference AoA, and
  bounce-utilization margin;
- peak reduced turf penetration and normal impulse;
- shaft rotation rate, remove-shaft AoA counterfactual, Shapley AoA share,
  vertical-velocity share, face-normal rate, leading-edge 3D rate, and
  leading-edge rate relative to the arc where available.

The Python adapter uses the canonical nine-point wedge/ground geometry and
reduced turf model. The browser uses the same retained sweep concepts and a
parity-tested TypeScript port of the passive reduced firm-fairway diagnostic.
Both identify the illustrative calibration state.

## Decision Statistics

For each cohort count `k` in `n` all trials, the UI reports the two-sided 95%
Wilson interval. The declared nonnegative loss combines normalized carry and
lateral errors and visible penalties for ground-first/grazing/miss/failure,
missing required outcomes, and unsupported turf states. Turf penetration may
enter a ranking loss only when the selected turf profile is calibrated; the
current illustrative profile reports penetration diagnostically but excludes
it from the objective.

The study reports:

- expected all-trial loss with a deterministic seeded bootstrap 95% interval;
- upper-tail CVaR of the worst declared loss fraction;
- clean-contact probability and constraint-violation rate;
- prefix mean and standard-error convergence checkpoints;
- availability-aware P5/median/P95 metric distributions;
- nondominated Pareto candidates across expected loss, CVaR, and clean-contact
  probability without hiding tradeoffs in an arbitrary scalar weight.

## Interface and Export Contract

PyQt performs wedge post-processing on its variation worker thread and exposes
a dedicated Chip Forgiveness result tab with linked metric scatter/marginal
views. React exposes an explicit "Analyze Wedge Chip Forgiveness" option that
selects a ground-mode representative 56-degree wedge. Both clients allow the
carry target to be changed while defaulting to the 30-yard worked example.

Strict JSON retains the plan, every sampled input, complete simulation
configurations, wedge/ground/turf/loss contracts, all trial records including
turf status, diagnostics, metrics, summary confidence/tail/convergence
evidence, and limitations. CSV contains one row per configured trial with
stable candidate/objective/turf/sampling identifiers and unavailable metrics
left blank.

The browser is capped at 500 trials but currently performs the retained swing
and forgiveness analysis synchronously. Large-run worker execution,
phase-aware cancellation, and benchmarked memory budgets remain required
before this draft slice can be called release-complete.

## Verification

Automated tests pin Wilson intervals, deterministic bootstrap replay, known
CVaR values, failure retention, constraint penalties, Pareto dominance,
retained-run analysis without re-execution, cancellation/progress, strict
JSON/CSV export, PyQt worker/UI integration, browser interaction, and the
reduced turf parity case against the shared Python implementation.
