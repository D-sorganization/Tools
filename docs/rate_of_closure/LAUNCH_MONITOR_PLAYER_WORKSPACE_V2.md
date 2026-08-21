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

They may additionally pin a canonical authorized-corpus reference containing
only an opaque server root alias, repository, 40-character commit, manifest
and content SHA-256 values, and expected row count. Filesystem paths and rows
are rejected from this reference.

Rows are intentionally absent. Loading a project against a different dataset
fingerprint fails closed.

The separately named **full export** is an explicit disclosure action. It
contains `project.json`, `result.json`, `backing_rows.csv`, and `manifest.json`.
The manifest records the SHA-256 digest and byte count of every evidence file.
Repository permissions and source redistribution restrictions still govern
where that bundle may be stored.

## Calculation authority and current limitations

UpstreamDrift commit `453346806a2950354f5b72cc46c2646e66459c8c` is the
canonical cross-service authority for immutable dataset jobs and evidence-
bearing selected-pair/player-population covariation. Both clients validate the
same pinned golden, identity evidence, backing lineage, safe claims, page
bounds, and contract versions before displaying results. Dataset jobs accept
only server-authorized references and return aggregates; they never return shot
rows. Inline canonical covariation accepts at most 20,000 rows. The 261,666-row
authority therefore remains eligible for reference-only aggregate jobs but not
for an inline browser covariation request.

The shared qualification golden pins the private dataset repository at
`d469b8a427418fa00e99b0ad488e4310b067697d`, its Parquet manifest at SHA-256
`b45fd9100e6786d32dce229224ed901f02c20ef5c44962769faf6cc94700c299`, and
the sorted path-plus-content corpus digest at SHA-256
`7bedf88ba473c947db2d4d078a73ee0ccd3512ffa182b751ea0a23298d1ab10c`.

The embedded Python and React estimators described in
[WITHIN_PLAYER_COVARIATION.md](WITHIN_PLAYER_COVARIATION.md) and
[LONGITUDINAL_PLAYER_ANALYSIS.md](LONGITUDINAL_PLAYER_ANALYSIS.md) remain
explicitly labelled `offline compatibility`. They do not replace or silently
impersonate the canonical service when its URL, identity evidence, or row
eligibility is unavailable.

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
- out-of-core shot-row querying from the browser client (bounded canonical
  aggregate jobs are available);
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
