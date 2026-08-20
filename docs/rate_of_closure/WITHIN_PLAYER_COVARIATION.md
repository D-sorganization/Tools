# Within-Player Covariation and Population Meta-Analysis

Epic: [Tools #4277](https://github.com/D-sorganization/Tools/issues/4277)

This document defines the calculation, interpretation, export, and user-interface
contract for covariation inside Launch Monitor Player Analytics. It supplements
[`LAUNCH_MONITOR_PLAYER_ANALYTICS.md`](LAUNCH_MONITOR_PLAYER_ANALYTICS.md).

## Question and Evidence Boundary

The workbench answers three different descriptive questions for any two selected
numeric variables `X` and `Y`:

1. Within a selected player, do the variables move together from shot to shot?
2. Is that within-player pattern consistent across eligible players?
3. Do players with different average `X` also have different average `Y`?

Those questions are not interchangeable. A pooled correlation can be driven by
differences between player averages even when each player's shot-to-shot pattern
has the opposite sign. The application therefore displays pooled, within-player,
between-player, and cross-player meta-analytic results separately.

All outputs are observational associations. They do not establish that changing
one variable will cause the other to change. Plausible confounders include club,
target, session, monitor, environment, shot intent, skill, fatigue, measurement
error, and restricted ranges. Mechanical or causal conclusions need a designed
experiment or an appropriately controlled longitudinal model.

## Required Data and Inclusion Rules

The user explicitly selects:

- a player or grouping identifier;
- numeric `X` and `Y` variables;
- a minimum complete-pair count per player; and
- a confidence level.

The software never infers player identity from club, session, row order, file
name, or another proxy. Missing and blank identities are excluded. `X` and `Y`
are converted to finite numeric values, and incomplete pairs are excluded. Each
player's status states whether the group is eligible, too small, or constant in
one or both variables. Constant inputs have no defined correlation.

The qualified private corpus contains 11,699 TrackMan-labelled rows, including
9,298 strict five-input rows, but no approved repeating player/session split
group. It can support qualified shot- and model-level comparisons, not
within-player inference or leakage-safe vendor-surrogate training. Player
analysis becomes available only when the user loads a dataset containing an
explicit identity column and attests that identity. The UI states this
limitation rather than treating a source, session, club, filename, or row order
as one known player.

## Per-Player Estimates

For each eligible player, pairwise-complete shots produce:

- Pearson product-moment correlation `r`;
- Spearman rank correlation as a descriptive monotonic-association check;
- ordinary least-squares slope and intercept for `Y = intercept + slope * X`;
- `R^2 = r^2` for the one-predictor ordinary least-squares fit; and
- a Fisher-transformation confidence interval for Pearson `r`.

For `n > 3`, the Pearson interval uses:

```text
z       = atanh(r)
SE(z)   = 1 / sqrt(n - 3)
CI_z    = z +/- NormalQuantile(1 - alpha / 2) * SE(z)
CI_r    = tanh(CI_z)
```

The interval assumes independent paired observations and the usual
approximately bivariate-normal Pearson model. It is not applied to the
descriptive Spearman result. Repeated shots within sessions can violate
independence; users should stratify or model session effects before treating an
interval as confirmatory.

Slope units are `Y unit / X unit`; the intercept uses the `Y` unit. Correlations
and `R^2` are unitless. Every chart axis and export field carries the inferred or
declared unit.

## Pooled, Within-Player, and Between-Player Associations

The pooled estimate correlates all included raw `(X, Y)` pairs. It mixes
shot-to-shot and player-to-player variation.

The within-player estimate first subtracts each player's variable means:

```text
X_centered(i,j) = X(i,j) - mean_j(X)
Y_centered(i,j) = Y(i,j) - mean_j(Y)
```

It then correlates the centered rows. This describes average shot-to-shot
co-movement after removing differences in player means. It does not give every
player equal influence; players with more retained shots contribute more rows.
The per-player table and random-effects result provide complementary views.

The between-player estimate correlates each player's mean `X` with that player's
mean `Y`. Its analytical unit is the player, not the shot, and it must not be
used to infer an individual player's shot-to-shot mechanics.

Opposite signs between pooled and within-player results are flagged as a
possible aggregation reversal. The flag is a prompt to inspect player and
session structure, not proof of a named paradox or causal mechanism.

## Cross-Player Fisher-Z Synthesis

At least two eligible player correlations are required for a combined effect.
Each Pearson correlation is transformed to Fisher `z_i = atanh(r_i)`, with
sampling variance `v_i = 1 / (n_i - 3)`. The fixed-effect estimate weights each
player by `w_i = 1 / v_i`.

The random-effects estimate uses the DerSimonian-Laird one-step estimate:

```text
Q       = sum(w_i * (z_i - z_fixed)^2)
C       = sum(w_i) - sum(w_i^2) / sum(w_i)
tau^2   = max(0, (Q - (k - 1)) / C)
w_RE,i  = 1 / (v_i + tau^2)
z_RE    = sum(w_RE,i * z_i) / sum(w_RE,i)
```

The displayed random correlation is `tanh(z_RE)`. Heterogeneity is summarized
with `Q`, `tau^2`, and `I^2 = max(0, (Q - (k - 1)) / Q) * 100%` when `Q > 0`.
With few players, heterogeneity and normal-approximation intervals are unstable;
results remain exploratory. A large average association with high heterogeneity
means the direction or magnitude may not generalize to every player.

The implementation follows the standard Fisher transformation exposed by
[SciPy `pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
and the DerSimonian-Laird random-effects method described in the
[original paper](https://pubmed.ncbi.nlm.nih.gov/3802833/) and available in
[statsmodels meta-analysis](https://www.statsmodels.org/stable/examples/notebooks/generated/metaanalysis1.html).

## Pair Scan and Multiple Comparisons

The pair scan evaluates every selected numeric-variable pair using the same
eligibility rules and ranks the available cross-player results by absolute
association with deterministic tie breaking. It reports contributing players,
sample counts, heterogeneity, and direction consistency.

Because scanning many pairs increases the chance of finding an extreme result,
the ranking is explicitly exploratory. A ranked pair is a hypothesis generator,
not a validated discovery. Users should inspect the exact backing rows, effect
sizes, intervals, heterogeneity, and domain plausibility, then confirm important
pairs on held-out players or future sessions.

## Player-Facing Interpretation

For a face-angle/club-path example:

- a positive per-player correlation means more-positive face values tended to
  accompany more-positive path values for that player's retained shots;
- a negative correlation means they tended to move in opposite directions;
- a near-zero linear correlation does not rule out curvature, regimes, or
  session-specific relationships;
- a steep slope can reflect a narrow `X` range or outliers and must be read with
  the scatter plot and sample count; and
- a consistent meta result suggests a recurring association across represented
  players, not a universal swing law.

Correlations are sensitive to range restriction and mixtures. Compare like
clubs, intents, monitors, and conditions where possible. Track changes over
time by retaining session and timestamp fields rather than pooling an evolving
player into one timeless distribution.

## Persistence, Plots, and Exports

Saved projects retain the selected identity, variables, minimum sample count,
confidence level, source path/hash, and other player-analysis settings. Reload
must validate source provenance before restoring results.

The PyQt scatter view distinguishes players and displays variable units. Plot
images remain exportable through the existing PNG, SVG, and PDF workflow. CSV
and JSON exports retain the complete centered backing rows, per-player estimates,
eligibility status, normalized fixed/random weights, and summary calculations
needed to reproduce the visible result.

The React/Vite surface provides equivalent selection, calculations, explanatory
text, plot semantics, and accessible hover guidance. Browser exports are
user-directed and inherit the loaded dataset's redistribution restrictions.

## Validation Contract

Tests must cover:

- a constructed aggregation reversal with different pooled, within, and
  between-player signs;
- missing, blank, too-small, and constant groups;
- Fisher intervals and fixed/random weights;
- heterogeneous player effects and unavailable single-player meta-analysis;
- deterministic all-pairs ranking and exploratory warnings;
- explicit identity selection, units, persistence, exports, accessibility, and
  PyQt/React calculation parity.

Use the repository's focused Python and web test, lint, type-check, and build
commands before publication. Hosted exact-head checks and protected branch
rules remain the release authority.
