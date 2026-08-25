# Launch Monitor: Shared Golden Fixtures, Full-Corpus Gates, and Program Manifest Reconciliation

Issue: [#4605](https://github.com/D-sorganization/Tools/issues/4605)
Parent epic: [#4226](https://github.com/D-sorganization/Tools/issues/4226)
Program epic: [#4583](https://github.com/D-sorganization/Tools/issues/4583)

## Summary of Deliverables

1. **Backend-Authoritative Golden Fixtures**:
   - Landed `src/rate_of_closure/web/src/model/__fixtures__/launch_monitor_conformance_bundle_golden_v1.json` spanning 10 scenarios across 5 analysis families (`analysis_v2`, `player_covariation`, `attested_longitudinal`, `source_backed_strokes_gained`, `distance_target_proxy`) with available and unavailable states.
   - Landed `src/rate_of_closure/web/src/model/__fixtures__/launch_monitor_player_covariation_golden_v1.json` with synthetic aggregation-reversal test vectors.
   - Preserved `launch_monitor_workspace_v3_golden.json` for row-free project serialization.
   - Added symmetric Python (`tests/rate_of_closure/test_launch_monitor_conformance_golden.py`) and TypeScript (`src/rate_of_closure/web/src/model/launchMonitorConformanceGolden.test.ts`) contract verification suites, maintaining 100% fixture parity.

2. **Forbidden-Identity Policy Tests**:
   - Added Python (`tests/rate_of_closure/test_launch_monitor_forbidden_identity.py`) and TypeScript (`src/rate_of_closure/web/src/model/launchMonitorForbiddenIdentity.test.ts`) tests proving player identity is never inferred from `session_id`, `club`, `source_id`, `filename`, `row_index`, `monitor_vendor`, or row order.
   - Verified that un-attested, missing, or blank identity columns fail closed across all workspace schemas, payload builders, and canonical response validators.

3. **261,666-Row Full-Corpus Gates**:
   - Expanded `tests/rate_of_closure/test_launch_monitor_private_corpus.py` to test the full governed 261,666-row corpus across 27 sources.
   - Added fail-closed gates for row-count mismatch, unmanifested source partitions, manifest schema errors, and desktop row-budget limits (`MAX_RETAINED_ROWS = 300,000`).

4. **Persistence/Load & Provenance Tests**:
   - Added `tests/rate_of_closure/test_launch_monitor_provenance_and_unavailable.py` and `src/rate_of_closure/web/src/model/launchMonitorProvenanceAndUnavailable.test.ts` validating SHA-256 integrity, commit hash formatting, deterministic joins, and typed unavailable states.

5. **Cross-Repository Program Manifest Reconciliation**:
   - Regenerated `docs/release/launch_monitor_program.v1.json` from exact current repository HEADs:
     - `D-sorganization/Tools`: `535b389e12f574e4a761b5d7a7f41039fe942a36`
     - `D-sorganization/UpstreamDrift`: `99cfe284bc9e80d576e027db7f81b8bb9b2af61e`
     - `D-sorganization/AffineDrift`: `c9c5db7470553ef544262c62c7273bcfb8654812`
     - `D-sorganization/Launch-Monitor-Data`: `5bb753f6d1c1a866c226eacb3672a211fedc04d0`
     - `D-sorganization/Launch-Monitor-Flight-Model-Campaign`: `d469b8a427418fa00e99b0ad488e4310b067697d`
   - Reconciled capability tracking statuses: marked `#4277` (within-player covariation) and `#4584` (strokes gained and proxy) as `complete`.
