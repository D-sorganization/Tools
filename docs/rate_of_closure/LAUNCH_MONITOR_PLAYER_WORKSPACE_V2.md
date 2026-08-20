# Launch Monitor Player Workspace v2

## Scope

The Player Covariation Workspace is a selective recovery of the player-analysis
work formerly isolated on `feat/4226-launch-monitor-player-platform`. It is
implemented against the current Rate of Closure application shell instead of
merging that historical branch.

The React and PyQt clients expose the same workflow:

1. Load or select a launch-monitor dataset.
2. Choose the column that genuinely identifies a player.
3. Explicitly attest that the identity was supplied by the data owner.
4. Choose any two eligible numeric variables.
5. Run per-player correlation through the existing analysis adapter.
6. Save a reference-only project or explicitly export a full evidence bundle.

Both clients also expose a Performance Analytics workspace with:

- carry/lateral dispersion converted from an explicitly selected metre or yard
  source unit into yards left/right of target;
- a unit-labeled plot and PNG/SVG/PDF (PyQt) or SVG (React) export;
- radial target error in yards, distinctly labeled as not strokes gained;
- source-backed strokes gained only when a versioned expected-strokes artifact
  passes schema, provenance, uniqueness, and SHA-256 validation and the shot
  supplies exact before/after lie and distance state;
- separately labelled user-supplied expected-strokes bookkeeping only when
  expected-strokes-before, expected-strokes-after, and an HTTP(S) citation are
  supplied;
- session and cumulative means only after player identity, session identity,
  and an explicit numeric session-order column are attested; and
- fingerprint-bound performance-analysis save/load plus backing CSV export.

The directional convention is negative lateral = left and positive lateral =
right. Radial target error is
`hypot(target_yards - carry_yards, lateral_yards)`. True strokes gained is
`expected_before - 1 - expected_after`. Cumulative session means give each
session equal weight; they are not a fitted improvement slope.

## Identity boundary

Player identity is never inferred from session, club, monitor, filename, source
row, file layout, or row order. Player analysis remains disabled until a user
selects and attests an identity column. Changing the identity column clears the
attestation and invalidates the prior result.

This rule prevents session identifiers and anonymous corpus partitions from
being misrepresented as people. A dataset without a trustworthy player column
is ineligible for player-level or longitudinal conclusions.

## Persistent project versus full export

Contract `2.0.0` project documents contain only:

- project name;
- immutable dataset reference and SHA-256 fingerprint;
- repository, revision, and relative-path provenance;
- explicit identity binding; and
- selected variables and uncertainty settings.

Rows are intentionally absent. Loading a project against a different dataset
fingerprint fails closed.

The separately named **full export** is an explicit disclosure action. It
contains `project.json`, `result.json`, `backing_rows.csv`, and `manifest.json`.
The manifest records the SHA-256 digest and byte count of every evidence file.
Repository permissions and source redistribution restrictions still govern
where that bundle may be stored.

## Calculation authority and current limitations

UpstreamDrift contract v2 is the canonical cross-service envelope for generic
launch-monitor analytics, evidence lineage, backing records, and typed
unavailable states. Tools currently owns the specialized grouped estimators
described in [WITHIN_PLAYER_COVARIATION.md](WITHIN_PLAYER_COVARIATION.md) and
[LONGITUDINAL_PLAYER_ANALYSIS.md](LONGITUDINAL_PLAYER_ANALYSIS.md). Python is the
desktop calculation authority; the React implementation is a tested browser
twin with the same eligibility, formula, unit, warning, and export contracts.
This is an explicit local release boundary, not a claim that the grouped
operations are already part of the UpstreamDrift v2 API.

The released grouped surface includes player-mean-centered pooled effects,
between-player decomposition, per-player correlations and regressions,
fixed/random Fisher-z synthesis, session uncertainty, per-player longitudinal
slopes, fixed/random population trends, and complete backing exports. The
source-backed strokes-gained contract is documented in
[SOURCE_BACKED_STROKES_GAINED.md](SOURCE_BACKED_STROKES_GAINED.md); no baseline
table is bundled, so that mode remains unavailable until the user supplies a
valid artifact and complete course-state inputs.

The remaining unavailable or deliberately bounded capabilities are:

- mixed-effects longitudinal models with session/player dependence beyond the
  released summary and DerSimonian--Laird synthesis;
- clustered or repeated-measures confidence limits for the centered pooled
  correlation;
- out-of-core private-corpus querying from the browser client;
- vendor-model training when the capability manifest denies eligibility; and
- causal improvement, swing-mechanism, or device-certification claims.

The PyQt client can load the complete authorized source-partitioned Parquet
authority from the directory selected by the user or
`LAUNCH_MONITOR_DATA_ROOT`. The loader verifies the corpus manifest, total row
count, and exact source-ID set before exposing the frame. The current governed
authority contains 261,666 rows across 27 sources. All rows remain available to
desktop analysis and explicit export; the linked scatter renders a deterministic
maximum of 2,000 points and reports both the finite-pair count and full retained
row count. Ordinary untrusted CSV/JSON imports retain the shared 250,000-row,
8-MiB and dense-cell limits. React consumes immutable references and private API
responses and never bundles restricted rows.

Saved projects continue to reference the private corpus by exact repository
revision, relative path, row count, and hash; they do not copy corpus rows.

All displayed associations are descriptive and do not establish causation.
