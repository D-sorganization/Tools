# Launch Monitor Workspace v3

`launch-monitor-workspace/v3` is the shared PyQt/React persistence contract for
player covariation and performance/session analysis. It stores settings,
aggregate results, units, formulas, exclusions, immutable source references,
authority commit and manifest hashes when available, and explicit
player/session/order attestations. It does not establish a new statistical
authority: canonical computations remain UpstreamDrift v2 responses, while
local computations are labelled `offline-compatibility-v1`.

Saved projects are always row-free. Per-shot, per-player, and per-session point
collections remain outside project JSON. Approved desktop bundles can add
`backing_rows.csv` plus `backing_join.csv`; each join uses the SHA-256 of
canonical JSON for the corresponding source row. Integral JSON numbers are
normalized consistently across Python and TypeScript. Restricted rows require
an explicit desktop approval for every export. Browser clients fail closed and
return a typed unavailable reason instead of embedding those rows.

## Compatibility and format parity

- PyQt and React write v3 and read v3.
- The player adapter reads legacy `2.0.0` projects as labelled compatibility
  imports. The performance adapter reads legacy
  `launch-monitor-performance/1.0` documents the same way.
- PyQt renders plots as SVG, PNG, or PDF and exports explicitly approved backing
  rows as CSV or JSON. React renders SVG or PNG; PDF and restricted backing-row
  exports are visibly unavailable because the browser has no approved private
  filesystem authority.
- Source-backed strokes-gained and longitudinal child workspaces retain their
  own evidence documents. Their child-specific controls/results are not folded
  into the parent performance document in either client; this avoids creating a
  second incomplete schema for those canonical contracts.

The cross-client golden fixture is
`src/rate_of_closure/web/src/model/__fixtures__/launch_monitor_workspace_v3_golden.json`.
