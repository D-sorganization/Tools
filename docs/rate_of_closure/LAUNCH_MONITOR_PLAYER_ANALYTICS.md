# Launch Monitor Player Analytics Platform

Epic: [Tools #4226](https://github.com/D-sorganization/Tools/issues/4226)

This document is the calculation, data-boundary, and parity contract for the
launch-monitor player analytics platform. It supplements the general analytics
contract in `docs/specs/LAUNCH_MONITOR_ANALYTICS.md`.

The arbitrary-variable, within-player, and population synthesis extension is
specified in
[`WITHIN_PLAYER_COVARIATION.md`](WITHIN_PLAYER_COVARIATION.md).

## Repository and Sharing Boundary

Three repositories have deliberately different responsibilities:

| Repository                             | Visibility | Permitted contents                                                                                                                                                                       |
| -------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Launch-Monitor-Data`                  | Public     | Source catalog, source URLs, vendor/field metadata, reported units, provenance, and share-safe aggregates.                                                                               |
| `Launch-Monitor-Flight-Model-Campaign` | Private    | Restricted source bytes, normalized shot rows, comparison cohort, shot-level model predictions, generated internal plots, PCA, feature-importance tables, and reproducibility manifests. |
| `Tools`                                | Public     | Analysis code, UI, schemas, help, and path/hash references to private datasets; no restricted shot rows.                                                                                 |

Tools locates the private campaign from an explicit path, the
`LAUNCH_MONITOR_CAMPAIGN_REPO` environment variable, or the standard sibling
repository layout. A saved PyQt6 project stores the campaign path, selected
dataset, source SHA-256, and user selections. It does not embed the dataset. On
reload, the source hash must match before analysis continues. A React browser
project is self-contained and embeds its retained rows because a browser cannot
safely reopen an arbitrary local path; it therefore inherits the embedded
dataset's access and redistribution restrictions.

An explicit UI export of retained data or plot backing rows is user-directed
and may contain restricted material. The export inherits the source dataset's
access, license, and redistribution limits; it is not made public merely
because the exporter is in Tools.

## Private v1 Dataset Catalog

The counts below come from the private campaign's reproducibility manifest.
They describe different analytical units and must not be summed as a total
number of independent shots.

| Dataset                   |   Rows | Analytical unit          | Purpose                                                                        |
| ------------------------- | -----: | ------------------------ | ------------------------------------------------------------------------------ |
| `trackman_normalized.csv` | 10,169 | one retained source shot | Lossless normalized and quality-annotated TrackMan source.                     |
| `analysis_cohort.csv`     |  8,860 | one eligible source shot | Preregistered complete-case cohort used for all five-output model comparisons. |
| `shot_predictions.csv`    | 62,020 | one model-shot pair      | Seven successful model simulations for each of the 8,860 cohort shots.         |

The same campaign also manifests overall metrics, stratified metrics, residual
correlations, model rankings, the report, and their SHA-256 values. Dataset
selectors should show the live manifested row/column count and hash rather than
hard-code those ancillary table sizes.

TrackMan trajectory fields are vendor outputs, not independently measured
ground truth. The source does not identify hardware generation, firmware,
software version, normalization state, or environmental settings. Results are
therefore agreement studies against TrackMan-generated outputs. UI and reports
must say `TrackMan-Comparable` or `Foresight-Comparable`; they must not claim
device emulation, certification, or vendor interchangeability.

## Units and Plot Backing Data

Every numerical axis, table column, and exported analytical field must state a
unit or explicitly say `unitless`. Canonical suffixes such as `_m`, `_yd`,
`_mps`, `_mph`, `_rpm`, `_deg`, and `_s` take precedence over name heuristics.
Distances are converted once into the display unit; calculations retain their
documented canonical unit. Signed lateral distance is shown in yards for the
player-facing dispersion and strokes-gained views, with negative values left
of target and positive values right of target.

Plots can be saved as PNG, SVG, or PDF. The exact rows used to draw the current
plot can be exported as CSV or record-array JSON. Backing exports retain the
source index or stable shot identifier plus all derived values necessary to
recalculate the mark. A screenshot is never the only retained evidence.

The relationship view uses one point per finite `(x, y)` pair and performs no
aggregation. The private paired-comparison view uses one color for every
observed TrackMan shot and another color for every corresponding flight-model
prediction. It retains `shot_id`, model name, observed value, predicted value,
and `predicted - observed` residual so paired inspection is possible even when
points overlap.

## Directional Dispersion

For signed lateral outcome `L_i` in yards and a user-selected center tolerance
`t >= 0`:

```text
left   : L_i < -t
center : -t <= L_i <= t
right  : L_i > t
```

The summary reports the three counts, arithmetic mean, sample standard
deviation, and the 50th and 80th percentiles of `abs(L_i)`. When downrange
distance is available, it also reports an 80% bivariate-normal covariance
ellipse. If `lambda_1 >= lambda_2` are covariance eigenvalues, the radii are:

```text
r_j = sqrt(-2 ln(1 - 0.80)) sqrt(lambda_j),  j in {1, 2}
```

The major-axis angle is taken from the first eigenvector. This ellipse is a
normal-model contour, not a guarantee that 80% of a non-normal sample lies
inside it. At least three complete two-dimensional shots are required. The
backing export contains the converted lateral and downrange values for every
included shot.

## Range-Shot Strokes-Gained Proxy

The reference table is Table 9 in Mark Broadie's _Assessing Golfer Performance
Using Golfmetrics_, estimated from more than eight million PGA TOUR shots from
2003 through 2010:
[Broadie source PDF](https://www.columbia.edu/~mnb2/broadie/Assets/strokes_gained_pga_broadie_20110408.pdf).
The complete reference grid is retained with the calculation output and can be
exported with the per-shot backing data.

For target distance `D`, carry `C_i`, and signed lateral result `L_i`, the
planar remaining distance is:

```text
d_after,i = sqrt((D - C_i)^2 + L_i^2)
```

Expected strokes `E(distance, lie)` are linearly interpolated from the selected
start- and end-lie columns of the Broadie table. The shot proxy is:

```text
SG_proxy,i = E(D, start_lie) - 1 - E(d_after,i, end_lie)
```

Distances outside the available table are clamped to its nearest endpoint and
the backing row records that fact. The UI reports sample count, mean, median,
and clamped fraction.

This is a range-shot proxy, not official ShotLink strokes gained. Target
distance and both lies are user assumptions. Planar endpoint geometry ignores
obstacles, penalties, elevation, wind, green position, roll, and shot context.
It should not be used for player ranking or betting without a validated,
context-complete baseline.

## Session and Longitudinal Analysis

The user selects a metric, session identifier, and optional player identifier.
For every resulting group, the platform reports shot count, arithmetic mean,
and sample standard deviation. Groups receive a displayed session sequence
`k = 1, ..., m`. The trend is the ordinary least-squares slope in:

```text
session_mean_k = intercept + slope * k + error_k
```

When a timestamp column is selected, PyQt6 additionally orders sessions by
their earliest valid timestamp and fits the same relationship against elapsed
days. Player identifiers always create separate sequences and fits; the UI
never connects or regresses different players as one time series.

The unit is the selected metric unit per session. Positive does not always mean
improvement: the preferred direction depends on the metric. The slope is
descriptive, gives each displayed session mean equal weight, and does not
adjust for time gaps, club mix, player mix, environment, target, monitor,
fatigue, or sample size. Preserve session order and contextual fields in the
project/export before making a longitudinal claim.

## Private PCA and Difference-Driver Analysis

Advanced model-difference analysis belongs in the private campaign repository
because its inputs and shot-level outputs are restricted. The primary response
for each model/output is the paired residual:

```text
residual = model_prediction - TrackMan_output
```

PCA is performed on a declared complete-case feature matrix after each feature
is standardized as `z = (x - mean) / population_standard_deviation` (`ddof=0`,
matching scikit-learn `StandardScaler`). Eigenvectors of the standardized
covariance matrix define component loadings; row scores are the standardized
matrix projected onto those loading vectors; explained variance ratio is each
eigenvalue divided by the sum of eigenvalues. Export the feature list,
means/scales, loadings, scores, eigenvalues, explained-variance ratios,
missing-row exclusions, and software/random-seed provenance.

PCA is unsupervised: it describes correlated variation in inputs and does not
by itself explain a model residual. Loading signs are arbitrary, and correlated
features can rotate or redistribute loading magnitude.

Residual-driver analysis should therefore include complementary, explicitly
associational methods:

- standardized multivariable linear regression for signed direction and an
  interpretable conditional coefficient;
- rank correlation with Benjamini-Hochberg adjusted p-values for monotonic
  screening;
- held-out permutation importance from a nonlinear model for interactions and
  curvature, reported with repeat variability;
- model/output-specific error stratification and residual plots to reveal
  heteroscedasticity and regime changes.

All validation splits must group on stable `shot_id`. The 62,020 prediction
rows repeat each source shot seven times; row-random splitting would leak the
same shot into training and validation. Report held-out score, baseline score,
split seed, folds, predictor definitions, missing-data policy, and exact cohort
hash. Importance, coefficients, PCA loadings, and correlations do not establish
causality or disclose how a proprietary vendor algorithm is implemented.

## Calculation Help and Accessibility

Every button, selector, input, plot canvas, and export control requires a
visible label plus a `title`, tooltip, or accessible name. Help for each method
must include:

1. what is calculated and the exact formula or algorithm;
2. input and output units;
3. missing-data and inclusion rules;
4. assumptions and minimum sample size;
5. interpretation and major limitations;
6. source URL or internal provenance reference; and
7. where to export the exact backing rows.

The detailed text must remain available in an inspectable Calculation Guide,
not only in transient hover text.

## PyQt6 and React/Vite Parity Gate

The PyQt6 workbench is the first integration surface for private local files.
React/Vite must expose equivalent analytical meaning even if browser security
requires an explicit file/directory selection instead of automatic sibling
discovery. Before epic release, parity tests must pin:

- stable primary-tab identity and accessible controls;
- dataset/project schema and source-hash behavior;
- units, signs, formulas, and sample counts;
- plot/backing exports and help content;
- dispersion, strokes-gained proxy, and session-trend reference fixtures; and
- private PCA/feature-importance artifact import without embedding private rows
  in the public web bundle.

## Validation Commands

Run focused checks first, then the repository gates:

```powershell
python -m pytest tests/rate_of_closure/test_launch_monitor_analysis.py tests/rate_of_closure/test_launch_monitor_analytics_tab.py tests/rate_of_closure/test_launch_monitor_player_metrics.py
python -m ruff check src/rate_of_closure/launch_monitor*.py src/rate_of_closure/ui/pyqt6/launch_monitor*.py tests/rate_of_closure/test_launch_monitor*.py
python -m ruff format --check src/rate_of_closure/launch_monitor*.py src/rate_of_closure/ui/pyqt6/launch_monitor*.py tests/rate_of_closure/test_launch_monitor*.py
python -m mypy src/rate_of_closure/launch_monitor_data.py src/rate_of_closure/launch_monitor_player_metrics.py src/rate_of_closure/ui/pyqt6/launch_monitor_analytics_tab.py
python -m pytest tests/rate_of_closure src/shared/python/swing_sim -n auto --timeout=60

Set-Location src/rate_of_closure/web
npm test
npm run type-check
npm run lint
npm run build
```

Record exact test counts, commit SHA, platform, and any excluded/blocked lane in
the handoff only after the command completes.
