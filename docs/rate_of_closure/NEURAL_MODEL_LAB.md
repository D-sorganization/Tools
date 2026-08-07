# Neural Model Lab

Public epic: [Tools #4240](https://github.com/D-sorganization/Tools/issues/4240)  
Private training issue:
[Launch-Monitor-Flight-Model-Campaign #9](https://github.com/D-sorganization/Launch-Monitor-Flight-Model-Campaign/issues/9)

## Purpose and Scientific Boundary

Neural Model Lab trains and inspects multi-output regression surrogates for
traceable launch-monitor outcomes. The current artifact is
**TrackMan-Comparable**: it learns recorded TrackMan outcomes from measured
launch conditions. It is not TrackMan firmware, a certified emulator, evidence
of TrackMan's proprietary algorithm, or independent ground truth.

A vendor label is eligible only when approved row-level targets, units,
provenance, cleaning rules, and a source checksum exist. Aggregate publications
must never be expanded into synthetic shot rows.

| Vendor      | Eligible row-level corpus | Training behavior              |
| ----------- | ------------------------: | ------------------------------ |
| TrackMan    | 8,860 complete-case shots | Enabled as TrackMan-Comparable |
| Foresight   |                      None | Fails closed                   |
| FlightScope |                      None | Fails closed                   |

Changing a vendor string cannot bypass this boundary. Simulated targets must be
named for their source flight model, not relabeled as a monitor vendor.

## Repository Boundary

The public Tools repository contains UI, schemas, validation, safe inference,
training-request generation, tests, and this documentation. The private
`Launch-Monitor-Flight-Model-Campaign` repository owns restricted source rows,
training code and configuration, fitted weights, per-shot predictions, metrics,
manifests, reports, and model indexes.

No restricted training rows are bundled into Tools or its React build. A model
bundle contains weights and statistical/provenance metadata; it can still leak
information about its training distribution and therefore remains governed by
the private campaign's release decision.

## Current Training Contract

The checksum-verified cohort has 8,860 unique `shot_id` rows. The stable split
hashes the UTF-8 text `4205:shot_id` with SHA-256, converts the first eight digest
bytes to a fraction in `[0, 1)`, and applies fixed thresholds:

| Split      | Threshold                 |  Rows | Role                                 |
| ---------- | ------------------------- | ----: | ------------------------------------ |
| Train      | `fraction < 0.70`         | 6,196 | Fit bounded candidate architectures  |
| Validation | `0.70 <= fraction < 0.85` | 1,324 | Select architecture only             |
| Test       | `fraction >= 0.85`        | 1,340 | Score once after selection and refit |

The dataset SHA-256 is
`04c8af8f54cc08ef0fa042d5e9e60d1d2708bbb5100647a5e60bd0450ca457b6`.
The configuration SHA-256 is
`62245750d89c435a1cd06781e517f29363dae08b483ad467ffb6456ca8d09029`.

The bounded search compares ReLU networks `32x32`, `64x32`, and `128x64`, each
with seed 4205 and at most 600 iterations. Selection minimizes validation RMSE
after standardizing all five targets. The selected architecture is `64x32` with
validation standardized RMSE `0.1067716507`. It is then refit on the 7,520
combined train-plus-validation rows. The untouched test split is scored once;
test results do not select architecture or hyperparameters.

Inputs are standardized from the applicable fit population using
`z = (x - mean) / population_standard_deviation`; constant scales become one.
Outputs use the same transformation during fitting and are converted back to
their stated physical units for metrics and inference.

## Variables and Units

| Role   | Bundle name                  | Unit                             |
| ------ | ---------------------------- | -------------------------------- |
| Input  | `ball_speed_mph`             | mph                              |
| Input  | `launch_angle_deg`           | deg                              |
| Input  | `launch_direction_deg`       | deg                              |
| Input  | `spin_rate_rpm`              | rpm                              |
| Input  | `spin_axis_deg`              | deg                              |
| Output | `observed_carry_m`           | m                                |
| Output | `observed_lateral_m`         | m; left negative, right positive |
| Output | `observed_apex_m`            | m                                |
| Output | `observed_landing_angle_deg` | deg                              |
| Output | `observed_flight_time_s`     | s                                |

All five inputs and outputs must be finite complete cases. `shot_id` must be
present and unique, preventing the same identity from crossing split boundaries.

## Safe Portable Bundle

The portable artifact schema is `launch-monitor-neural-bundle/v1`. It is JSON
data only—never pickle, joblib, Python bytecode, ONNX extensions, or another
executable serialization. Required top-level fields are:

```text
schema, modelId, vendor, createdAt,
features[], outputs[], layers[], metrics[], learningCurve[], provenance{}
```

Each feature stores `name`, `unit`, finite `mean`, positive finite `scale`, and
finite applicability `min`/`max`. Each output stores `name`, `unit`, finite
`mean`, and positive finite `scale`. Names are unique within each role.

Each dense layer stores `activation`, `weights`, and `bias`. Weights use
`weights[output_node][input_node]`. Supported activations are `relu`, `tanh`,
and `linear`; every weight and bias must be finite, adjacent dimensions must
match, and the final width must equal the output count. The cross-client safe
profile is at most 64 features, 32 outputs, 16 layers, and 1,024 nodes per
layer. Python additionally caps any layer at five million weights. Bundles
outside the cross-client profile are not parity-safe even if one runtime can
parse them.

`metrics` retains split, target, sample count, MAE, RMSE, R2, and unit.
`learningCurve` retains training fraction/rows and validation standardized
RMSE. `provenance` retains dataset/config hashes, split method/counts, seed,
architecture search, final-fit rows, vendor availability, and the assertion
that the test split was touched only after selection.

Inference is the deterministic dense forward pass:

```text
z0 = (x - feature_mean) / feature_scale
zk = activation(Wk z(k-1) + bk)
y  = zfinal * output_scale + output_mean
```

Inputs outside a feature's recorded minimum/maximum produce extrapolation
warnings. A warning does not validate the extrapolated prediction.

## Metrics, Baselines, and Applicability

For `n` test observations, with truth `y_i`, prediction `p_i`, and truth mean
`y_bar`:

```text
MAE  = sum(abs(y_i - p_i)) / n
RMSE = sqrt(sum((y_i - p_i)^2) / n)
R2   = 1 - sum((y_i - p_i)^2) / sum((y_i - y_bar)^2)
```

The independent test results in the private report are:

| Output        | Unit |       MAE |     RMSE |       R2 |
| ------------- | ---- | --------: | -------: | -------: |
| Carry         | m    |   2.67854 |  4.87229 | 0.991282 |
| Lateral       | m    |   1.26493 |  2.31123 | 0.989858 |
| Apex          | m    |  0.537397 |  1.23547 | 0.989079 |
| Landing angle | deg  |   1.03395 |  2.32703 | 0.965406 |
| Flight time   | s    | 0.0959367 | 0.218727 | 0.980682 |

Two independent-test baselines are retained: a development-set mean predictor
and ridge regression with alpha 1.0. The neural surrogate beats both on all five
reported test RMSEs. This result applies only to the manifested Blackmore
TrackMan-comparable cohort and its feature ranges. It does not establish
performance for a different player population, ball, club, environment,
monitor generation, firmware, software normalization, or future data.

## PyQt6 and React Workflows

PyQt6 can discover the sibling private campaign, load a safe bundle, query one
row or a selected dataset, export predictions, and prepare an inspectable TOML
training request. Training runs asynchronously in the private campaign process,
where cancellation and logs remain visible. The desktop never writes fitted
weights into Tools.

React imports a user-selected safe JSON bundle, validates it locally, exposes
metrics/provenance/learning curves, and performs deterministic browser
inference. Browser security prevents it from launching the private Python CLI.
Instead it exports the metadata-only
`launch-monitor-neural-training/v1` request; the user runs the private CLI and
then imports the resulting bundle. The request names the file, row count,
columns, roles, architecture, activation, regularization, iterations, learning
rate, validation fraction, and seed, but embeds no source rows.

Both clients must agree on variable order, units, normalization, layer storage,
activation, output de-normalization, applicability warnings, and reference
predictions. Every control requires accessible help; every metric and plot must
show method, unit, split, and sample count.

## Custom Dataset Training Request

Before training a custom CSV or record-array JSON dataset:

1. Preserve the source, calculate SHA-256, and record ownership/licensing.
2. Identify a traceable vendor/model target and reject unsupported labels.
3. Declare feature/output roles and units; a column cannot serve both roles.
4. Require finite complete cases and a stable unique shot/session/player group.
5. Fix seed, split fractions, architectures, activation, alpha, and iteration
   bound before inspecting test performance.
6. Write the TOML/request, review it, run only the private CLI, and verify every
   artifact hash.
7. Compare independent test results with mean and ridge baselines, inspect the
   learning curve and residuals, then document applicability limits.

## Security and Failure Behavior

- Accept only bounded JSON and reject duplicate keys, non-finite values,
  unsupported activations, invalid dimensions, excessive sizes, missing
  features, and reversed feature bounds.
- Never deserialize executable model objects or execute commands contained in
  a bundle.
- Build the training command from fixed program/argument fields; do not invoke
  it through a shell or accept arbitrary CLI fragments.
- Verify dataset, config, bundle, prediction, metric, curve, report, and index
  hashes before claiming reproducibility.
- Keep Foresight and FlightScope training disabled until reviewed row-level
  evidence is added.

## Exact Private Commands

Run these from the private `Launch-Monitor-Flight-Model-Campaign` checkout:

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python -m lm_flight_campaign.cli --config campaign.toml neural-train `
  --training-config neural_training.toml
python scripts/verify_neural_artifacts.py
python -m lm_flight_campaign.cli --config campaign.toml neural-query `
  --bundle models/trackman_surrogate_v1.json `
  --features-json '{"ball_speed_mph":150,"launch_angle_deg":15,"launch_direction_deg":0,"spin_rate_rpm":2500,"spin_axis_deg":0}'
```

The query is an illustrative in-range input, not a benchmark shot or guarantee.
The authoritative outputs remain the private bundle, manifest, metrics,
per-shot prediction backing table, learning curve, and report.

## Limitations

- The training targets are launch-monitor-generated outcomes, not measured
  ground-truth trajectories.
- The source omits monitor generation, firmware, software, and full environment
  metadata.
- One fixed split and one bounded architecture search do not quantify all model
  or sampling uncertainty.
- High R2 within this cohort does not prove calibration or safe extrapolation.
- Applicability bounds are rectangular feature minima/maxima; they do not detect
  sparse multivariate regions inside that box.
- The portable bundle supplies point predictions, not calibrated prediction
  intervals or causal explanations.
