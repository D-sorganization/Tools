# Reproducible Ensemble Variation and Sensitivity Analysis

This guide describes the open, reusable variation authority used by the Rate
of Closure workbench and downstream consumers. It explains what the software
computes, how to reproduce it, and which conclusions the outputs cannot
support. The implementation is useful for model interrogation and falsification;
it is not a substitute for governed participant measurements.

## Scope and Evidence Boundary

The variation package perturbs registered inputs, retains typed trial outcomes,
summarizes trajectory dispersion, identifies low-variability intervals, and
computes local, rank-based, and Morris screening measures. Its outputs are
**model-scenario evidence**. A deterministic replay establishes implementation
and data-contract consistency; it does not establish human validity, identify
muscle action, or prove that a simulated policy is physiologically feasible.

The package does not justify universal coaching advice. A high simulated
clubhead speed, a quiet geometric region, or a small sensitivity estimate is
conditional on the model, input domain, objective, event definition, solver,
and missing-data rule. Comparisons must retain those conditions and plausible
adverse cases.

No-impact is retained as a typed scientific outcome rather than deleted or
converted into a fabricated impact record. Numerical failures remain separate
from no-impact trials. Analyses that need an impact variable report it as
unavailable for trials that did not produce one.

## Mechanical and Statistical Interpretation

Variation results describe changes in declared observables under declared
input perturbations. They do not, by themselves, identify a mechanical cause.
In particular:

- trajectory dispersion is not energy transfer;
- input-output association is not joint work or momentum flow;
- a negative torque is not necessarily negative power;
- a low-variability region is not necessarily passive, stable, or optimal;
- a parameter importance measure depends on its sampled range and design; and
- correlation is not causation.

Interpret an output only after fixing its coordinate frame, units, point ID,
time or phase coordinate, event rule, valid-sample denominator, and cohort.
Compare speed, face/path, strike, balance, loading, effort proxies, and
robustness as separate objectives. A single scalar ranking must not silently
replace a multi-objective question.

## Data and Schema Contracts

The shared Python API is
`shared.python.swing_sim.variation`. The principal contracts are:

| Contract                                 | Purpose                                                                                               | Important Boundary                                                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `VariationPlan` and `NoiseSpec`          | Registered variables, distributions, bounds, groups, seed, run count, and optional locus              | Unknown variables, duplicate streams, unsafe integers, nonfinite values, and incompatible loci fail at construction. |
| `PerturbationGroup`                      | Correlation or covariance for disjoint jointly normal streams                                         | Matrices must be finite, symmetric, dimensionally consistent, and positive semidefinite.                             |
| Version-3 execution document             | Plan, resolved bases and units, registry digest, RNG identity, executor compatibility, and provenance | A matching plan digest does not claim identical solvers or floating-point behavior.                                  |
| `VariationDataset` and typed trial rows  | Sampled inputs, outputs, statuses, and available traces                                               | Hit, no-impact, and numerical-failure cohorts remain distinct.                                                       |
| JSON, CSV, and HDF5 readers/writers      | Review, interchange, and lossless durable data                                                        | CSV is a review table and cannot authorize replay without its canonical JSON/HDF5 evidence.                          |
| `EnsemblePositionTraces`                 | Point IDs, frame, common grid, positions, and validity masks                                          | Interpolation and missing-data rules are part of the estimand.                                                       |
| `MorrisDesign`, observations, and report | Global screening for nonlinear or interacting inputs                                                  | Elementary effects are scaled to declared normalized factor ranges; unavailable outputs remain typed.                |

The complete persistence and replay rules are in
[`docs/specs/VARIATION_PLAN_PERSISTENCE.md`](../specs/VARIATION_PLAN_PERSISTENCE.md).
The requirement-level evidence ledger is
[`docs/audits/rate_of_closure_epic_4142_evidence.v1.json`](../audits/rate_of_closure_epic_4142_evidence.v1.json).

## Methods and Assumptions

### Seeded Sampling

Independent normal, uniform, and triangular inputs and validated joint-normal
groups use stable per-stream seed derivation. Sampling is deterministic across
chunk sizes and worker counts, and an explicitly identified one-at-a-time
stream remains subset-stable. Determinism does not make the chosen distribution
or parameter range empirically correct.

### Dispersion and Quiet Regions

At each registered point and sample, the geometry authority can calculate
valid count, centroid, covariance, eigensystem, RMS radius, principal spread,
and confidence ellipsoid availability. Quiet intervals are contiguous regions
meeting an explicit metric, threshold, and minimum-duration rule. Rank
deficiency and inadequate sample counts produce typed unavailable states.

Quiet regions answer a geometric question under one alignment and cohort. They
do not demonstrate attractive dynamics, self-correction, low biological effort,
or good impact outcomes. Those require separate state-return, work/load, and
task-result evidence.

### Local and Rank Attribution

One-at-a-time effects compare complete declared factor streams. Spearman
analysis uses pairwise finite observations and reports the actual denominator.
Absolute scatter and noise responsiveness are different quantities and should
be shown together rather than conflated.

### Morris Screening

Morris screening evaluates elementary effects along a registered trajectory
design and reports mean absolute effect, signed mean, spread, uncertainty,
availability, and sample adequacy. Its values are conditional on factor bounds,
grid levels, trajectories, clamp behavior, and the chosen output. A large
spread can indicate nonlinearity, interaction, discontinuity, mixed typed
outcomes, or numerical trouble; the statistic alone does not distinguish them.

### Durable Execution

Large studies use bounded chunks, atomic manifests, per-chunk checksums,
verified-prefix resume, progress, and cancellation. An interrupted archive
authorizes analysis only over its verified contiguous prefix. Completion is
not inferred from a directory or a stale status field.

## Quick Start

From the repository root, expose the source tree for a clean checkout that has
not been installed as a package:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
```

Then create a small deterministic delivery plan and verify that serial and
parallel sampling produce the same inputs:

```python
import numpy as np

from shared.python.swing_sim.variation import NoiseSpec, VariationPlan, run_variation

plan = VariationPlan(
    mode="delivery",
    noise=(
        NoiseSpec(
            "swing_sim.impact.delivery.clubhead_speed_mps",
            distribution="normal",
            scale=1.0,
            spec_id="speed-stream",
        ),
        NoiseSpec(
            "swing_sim.impact.delivery.face_angle_deg",
            distribution="normal",
            scale=1.5,
            spec_id="face-stream",
        ),
    ),
    n_runs=32,
    seed=20260825,
)

serial = run_variation(plan, n_workers=1)
parallel = run_variation(plan, n_workers=4)
np.testing.assert_array_equal(serial.inputs, parallel.inputs)
assert serial.success.shape == (plan.n_runs,)
output_available = np.isfinite(serial.outputs)
assert output_available.shape == serial.outputs.shape
```

Inspect the per-trial `serial.success` mask and per-cell `output_available` mask
before computing a summary. A successful numerical evaluation can still lack a
downstream quantity, represented by `NaN`. Do not filter the dataset to impacts
only unless that conditional estimand was declared in advance and the excluded
no-impact cohort is reported separately.

For localized or model-specific perturbations, first query the adapter's
capability. The generic scalar executor deliberately rejects time/point locus
metadata it cannot execute; silently ignoring a locus would change the study.

## Reproducible Verification

From the Tools repository root, run the shared mechanics and contract suites:

```powershell
python -m pytest src/shared/python/swing_sim/variation/tests -q
python -m pytest tests/rate_of_closure -k "variation or morris or ensemble" -q
python -m ruff check src/shared/python/swing_sim/variation src/rate_of_closure tests/rate_of_closure
python -m ruff format --check src/shared/python/swing_sim/variation src/rate_of_closure tests/rate_of_closure
```

Run the React contract and interaction suites separately:

```powershell
cd src/rate_of_closure/web
npm test -- --run variation
npx tsc --noEmit
npx eslint .
```

These focused commands are reviewer entry points, not substitutes for the
repository's protected Python-version, browser, secrets, package, and
downstream-consumer gates. Record the exact commit, dependency environment,
plan/execution-document digest, dataset digest, command, and result when using
an output as evidence.

## Performance and Scaling Evidence

The checked visualization reference uses 500 trials, 240 common-time samples,
and one three-dimensional point. Its budgets and measured reference are
documented in
[`docs/rate_of_closure/variation_visualization_performance.md`](variation_visualization_performance.md).
They are regression guards, not hardware-independent benchmarks.

Durable streaming evidence separates logical trace volume, physical archive
growth, peak resident memory, and failure-only transport throughput. The
machine-readable reference is
[`docs/rate_of_closure/ensemble_stream_scaling.v1.json`](ensemble_stream_scaling.v1.json).
Before extrapolating it, verify source revision, hardware, worker count, chunk
size, trace layout, compression, and solver participation. A transport-only
measurement cannot establish simulation throughput.

## Review and Falsification Workflow

1. State the proposition, observable, model tier, cohort, event, frame, unit,
   and comparison rule before running the ensemble.
2. Register the plan, nuisance ranges, correlation groups, seed, solver,
   stopping rules, missing-data policy, and primary/adverse outputs.
3. Retain all typed outcomes and verify plan, registry, provenance, archive,
   and dataset digests before analysis.
4. Run deterministic replay, worker-count invariance, manufactured fixtures,
   negative controls, half-step or solver-refinement checks, and adverse cases.
5. Report absolute dispersion beside responsiveness; report Morris assumptions,
   uncertainty, adequacy, and unavailable results beside rankings.
6. Test countermodels that can reproduce the same output through different
   geometry, timing, contact, damping, or control allocation.
7. Narrow or reject the proposition when its registered falsifier occurs.
   Do not change the estimand after observing the result.
8. Treat participant validation as a separate governed stage with held-out
   people and synchronized measurements appropriate to the claimed mechanism.

## Limitations and Unsupported Inferences

- Registered ranges are model inputs, not automatically population priors.
- Independent streams do not represent biological independence.
- A covariance group is a sampling assumption unless estimated from governed
  data with uncertainty.
- Rank and Morris measures do not identify unmeasured confounding or mechanism.
- Finite ensembles can miss narrow adverse regions and bifurcations.
- Interpolation, alignment, event detection, and censoring can change geometric
  and sensitivity conclusions.
- Current localized execution does not cover every registered time-varying
  input or output family.
- Current adapter coverage does not prove equivalent resampling and complete
  event/impact/shot retention for every model.
- Performance evidence is bounded to its recorded hardware and workload.
- Synthetic agreement across implementations does not establish anatomy,
  physiology, fatigue, injury risk, participant benefit, or a universal swing
  strategy.

Use the ledger's remaining partial requirements as explicit falsifiability and
implementation gaps. Closing one documented gap does not close the broader
epic or the externally governed human-data boundary.
