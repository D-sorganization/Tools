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

The workspace does not introduce a second statistical engine. Its current
compatibility adapter delegates Pearson estimates and confidence intervals to
the existing Rate analysis boundary. The reference-only request contract is
prepared for the UpstreamDrift contract-v2 backend, which is the intended
cross-client authority.

Current grouped results are per-player correlations, not a complete hierarchical
within-player meta-analysis. The following remain unavailable until the
UpstreamDrift v2 endpoint is implemented and qualified:

- player-mean-centered pooled effects;
- fixed- and random-effects Fisher-z synthesis;
- between-player versus within-player decomposition;
- longitudinal improvement models;
- out-of-core private-corpus querying from the Tools clients; and
- neural-vendor training or inference.

The private corpus may be referenced by exact repository revision, relative
path, row count, and hash, but it is not copied into a saved Tools project.

All displayed associations are descriptive and do not establish causation.
