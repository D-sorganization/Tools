# Launch Monitor Analytics

Issue: [Tools #4205](https://github.com/D-sorganization/Tools/issues/4205)
Parent epic: [Tools #4226](https://github.com/D-sorganization/Tools/issues/4226)
Program epic: [Tools #4583](https://github.com/D-sorganization/Tools/issues/4583) / [UpstreamDrift #8790](https://github.com/D-sorganization/UpstreamDrift/issues/8790)

## Purpose

Launch Monitor Analytics is a primary Rate of Closure workspace in both PyQt6
and React/Vite. It preserves every imported CSV or JSON field and supports
flexible statistical analysis across any compatible numeric variables.

The public, source-traceable evidence database remains the separate
[Launch-Monitor-Data repository](https://github.com/D-sorganization/Launch-Monitor-Data).
The restricted 261,666-row shot corpus across 27 sources is governed by
[Launch-Monitor-Flight-Model-Campaign](https://github.com/D-sorganization/Launch-Monitor-Flight-Model-Campaign).
Published aggregates must remain aggregates; neither UI expands them
into fabricated shot rows.

## Shared Contract & Upstream Analytics v2

Both presentation surfaces implement contract version `1.0.0` for local analysis
and consume UpstreamDrift `2.0.0` canonical statistical services:

| Capability                                                   | PyQt6         | React/Vite      | Backend Authority |
| ------------------------------------------------------------ | ------------- | --------------- | ----------------- |
| CSV/JSON record import with source columns retained          | Yes           | Yes             | Tools local       |
| 261,666-row governed private corpus loader                   | Yes (desktop) | API / reference | Campaign Parquet  |
| Any numeric outcome and multiple predictors                  | Yes           | Yes             | Upstream v2       |
| Pearson, Spearman, and Kendall correlation                   | Yes           | Yes             | Upstream v2       |
| Pairwise, listwise, and fail-closed missingness              | Yes           | Yes             | Upstream v2       |
| Benjamini-Hochberg multiplicity correction                   | Yes           | Yes             | Upstream v2       |
| Multivariable OLS with coefficient intervals                 | Yes           | Yes             | Upstream v2       |
| R², adjusted R², RMSE, MAE, Durbin-Watson, influential count | Yes           | Yes             | Upstream v2       |
| Within-player covariation and meta-analysis                  | Yes           | Yes             | Upstream v2       |
| Source-backed strokes gained with course state               | Yes           | Yes             | Upstream v2       |
| Longitudinal session analytics and trends                    | Yes           | Yes             | Upstream v2       |
| Dataset SHA-256 and complete JSON evidence export            | Yes           | Yes             | Tools v3 export   |
| Row-free project save / load (Workspace v3)                  | Yes           | Yes             | Tools v3 schema   |

## Scientific Policies & Forbidden Identity

- **Explicit Player Identity**: Player identity is never inferred from session, club, monitor,
  source, filename, or row order. Player analytics requires an explicitly user-attested identity column
  and fails closed otherwise. Changing the identity column revokes attestation.
- **Row-Free Persistence**: Saved `.lmproject.json` documents are row-free by design. Raw shot rows
  are never embedded into project JSON.
- **Controlled Export**: Backing rows can only be exported in desktop bundles with explicit user approval.
  Browser clients fail closed for restricted backing rows.
- **Unavailable States**: Incomplete inputs, missing baseline tables, un-attested identity, or constant series
  report typed unavailable states with structured reasons rather than crashing or guessing.

## ADR-0046 Stage 2 — Canonical-Layer Mapping

ADR-0046 Stage 2 re-points both workbenches at the canonical model layer
`src/shared/python/launch_monitor/` (Stage 1 port ladder P1-P20 complete
2026-09-02), "retiring its private copy only when each module's consumers are on
the canonical one **and its tests pass against it**". This section is the Impact
Explorer tab's half of that: every `rate_of_closure` launch-monitor Python
module, the canonical module it maps to, and whether it can retire now.

**Result: zero pure duplicates, zero retirements.** ADR-0046 G1
([ADR-0048](../adr/ADR-0048-launch-monitor-port-plan.md)) found `already-home`
empty in the UpstreamDrift-to-canonical direction — every UpstreamDrift module
carried outputs its Tools counterpart could not produce. Measuring the reverse
direction gives the mirror-image result: no canonical module reproduces a
`rate_of_closure` module's outputs through the same result shape, so nothing
here is a re-point away from retirement. Across 2,830 legacy lines and 9,233
canonical lines only **six** definitions are AST-identical after docstring
stripping, none longer than nine lines (`DatasetSummary`, `CoefficientEstimate`,
`RegressionEstimate`, `GroupAnalysis`, `LoadedPrivateCorpus.source_name`,
`_has_finite_numeric`), and each sits in a module whose siblings diverge under a
pinned ruling.

Classification vocabulary:

- **pure-duplicate** — the canonical module is identical, or a superset
  producing identical numbers through the same result shape. Re-point the
  consumers and retire the local copy. **No module qualifies.**
- **divergent-by-ruling** — the divergence is pinned in
  `docs/shared/divergence_ledger.v1.json` and in UpstreamDrift's
  `tests/integration/launch_monitor_drift/`. The legacy posture stays until a
  paired cross-repo PR retires it; changing it from a Tools-only PR is refused
  by `scripts/check_divergence_ledger.py`.
- **twinned-golden** — the Python must stay bit-matched to a TypeScript twin
  under a cross-runtime golden fixture, so a Python-only re-point would break
  parity even where the numbers agree.
- **already-home** — this module _is_ the canonical authority for its capability
  (ADR-0046 Stage 1's already-home set). Retiring it would break the canonical
  layer.
- **disjoint** — no canonical module computes this at all.

| `rate_of_closure` module (LOC)                    | Canonical twin                                                             | Class                               | Retire now | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `launch_monitor_analysis.py` (228)                | `flexible_analysis.analyze_variables`                                      | divergent-by-ruling, twinned        | No         | Ledger `split`/`pinned`, rulings D15/D17/G1-D4. Canonical `CorrelationEstimate` adds `is_boolean_projected` and drops the `float \| None` blanking; `FlexibleAnalysisResult` adds `units` and carries no `contract_version`. D17 makes the canonical layer analyse the boolean columns this module refuses. TS twin `launchMonitorAnalysis.ts`.                                                                                                                                                                                                                             |
| `_launch_monitor_analysis_statistics.py` (200)    | `flexible_analysis._correlations`/`_regression`, `relationships.py`        | divergent-by-ruling                 | No         | Private half of the row above. ADR-0048 records the `relationships.py` counterpart as "plain correlation only".                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `_launch_monitor_analysis_types.py` (137)         | `flexible_analysis` dataclasses                                            | divergent-by-ruling                 | No         | Four of eight dataclasses are AST-identical to the canonical ones; `CorrelationEstimate`, `ResidualDiagnostics`, `AnalysisRequest` and `AnalysisResult` are not. UpstreamDrift's gate imports `CONTRACT_VERSION` and `AnalysisRequest` from this exact module path.                                                                                                                                                                                                                                                                                                         |
| `launch_monitor_longitudinal.py` (307)            | `longitudinal.py` + `longitudinal_statistics.py` + `longitudinal_types.py` | divergent-by-ruling, twinned        | No         | Ledger `split`/`pinned`, D10-D14 + G1-D1/G1-D2. `dl-random-effects/1` was ported _from_ this module's `_population`, but the canonical result is the pydantic `LongitudinalSessionResultV1` wire, not this dataclass, and the legacy estimator is a pinned number. TS twin `launchMonitorLongitudinal.ts`.                                                                                                                                                                                                                                                                  |
| `launch_monitor_strokes_gained.py` (485)          | `strokes_gained.py` (+ `strokes_gained_types`, `_scoring_statistics`)      | divergent-by-ruling, twinned-golden | No         | Ledger `split`/**`paired-open`**, D1-D5 + G1-D3. UpstreamDrift re-pins D1/D2 in the paired vendor-bump PR; G1-D2 makes the canonical estimand `session-cell-sg-trend/1` against this module's shot-level fit. Golden: `launchMonitorProvenanceAndUnavailable.test.ts`.                                                                                                                                                                                                                                                                                                      |
| `launch_monitor_strokes_gained_baseline.py` (206) | — (this module is the authority)                                           | already-home                        | No         | ADR-0048: the expected-strokes baseline half "is genuinely already home", and G0 pinned both digests identical. Canonical `strokes_gained.py` types its `baseline` argument structurally as `ExpectedStrokesBaselineLike` **so this module flows straight in** without the canonical layer importing `rate_of_closure`. Retiring it would break the canonical layer.                                                                                                                                                                                                        |
| `launch_monitor_performance.py` (294)             | `dispersion.py`, `trends.py`, `outcome_proxy.py` (partial)                 | divergent, twinned-golden           | No         | Ledger `split`/`pinned`, D6-D9: the two dispersion results share zero field names, and `radial_rmse` 11.365 against `rms_yards` 8.397 is the same fixture 35% apart. `TrendResult` here is session-ordinal; canonical `TemporalTrendResult` is per calendar day, deliberately with no alias. `calculate_target_error` _does_ agree with `analyze_outcome_proxy` to delta exactly `0.0`, but returns `ScoreResult(values, mean)` against `OutcomeProxyResultV1(row_results, value_summary)`, and `calculate_strokes_gained` in the same module has no canonical twin at all. |
| `launch_monitor_numeric.py` (31)                  | — none                                                                     | disjoint                            | No         | `finite_launch_monitor_scalar` refuses booleans and radix text. Ruling D17 makes the canonical layer analyse booleans as 0/1 via `pd.to_numeric`, so a canonical equivalent is ruled out rather than merely missing.                                                                                                                                                                                                                                                                                                                                                        |
| `launch_monitor_import.py` (245)                  | `importer.py`                                                              | disjoint, twinned-golden            | No         | Bounded defensive reader whose resource limits are pinned cross-runtime by `launch_monitor_import_limits_golden_v1.json` (64 KiB field cap, 250,000 rows, 256 columns, 2,000,000 cells, lone-surrogate rejection). Canonical `importer.py` does profile detection and ADR-0031 unit conversion instead; its docstring: "nothing here re-exports or aliases anything there".                                                                                                                                                                                                 |
| `launch_monitor_linked_scatter.py` (194)          | — none                                                                     | already-home, twinned-golden        | No         | ADR-0046 lists linked scatter among the Tools capabilities that are already home. Golden `launch_monitor_linked_scatter_golden_v1.json`, TS twin `launchMonitorLinkedScatter.ts`. Also the definition site of `MAX_RETAINED_ROWS`, which UpstreamDrift's corpus gate imports by name and canonical `corpus.py` re-derives as `300_000` behind a seam test.                                                                                                                                                                                                                  |
| `launch_monitor_private_corpus.py` (106)          | `corpus.py` (P19 merge)                                                    | divergent-by-ruling                 | No         | Ledger `split`/**`paired-open`**, D28-D31. The canonical module is a capability superset but not an output superset: of fifteen columns each, only `club` and `smash_factor` are shared, because canonical returns ADR-0031 units. `corpus.py`'s own docstring: "Retiring either legacy posture is a coordinated cross-repo change … tracked rather than smuggled."                                                                                                                                                                                                         |
| `launch_monitor_canonical_v2.py` (397)            | `contract_v2.py`                                                           | disjoint                            | No         | Zero shared definitions. This is the pinned client-side validator (`validate_dataset_job_page`/`_status`, `build_*_payload`, `CanonicalDatasetReference`, the `CANONICAL_DATASET_METRICS` allow-list) with its own `launch_monitor_canonical_v2_golden.json`; `contract_v2.py` is the 38-symbol pydantic server model layer. ADR-0048 records it as "pinned client half only".                                                                                                                                                                                              |
| `player_covariation.py` (371)                     | `player_covariation.py` (P18 union port)                                   | divergent-by-ruling, twinned-golden | No         | Ledger `split`/**`paired-open`**, D22/D23. The low-dof between-player Fisher interval and the `start_distance_yards` → `"s"` suffix heuristic are exactly the postures the owner ruled _against_ for canonical and that remain the pinned legacy in `test_player_covariation_drift.py`. Golden `launch_monitor_player_covariation_golden_v1.json`, TS twin `launchMonitorCovariation.ts`.                                                                                                                                                                                   |
| `_player_covariation_types.py` (99)               | `player_covariation_types.py`                                              | divergent-by-ruling                 | No         | Same D22/D23 pairing; UpstreamDrift's gate imports `MIN_FISHER_SAMPLES`, `CovariationRequest` and `PairScanRequest` from this exact module path.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `_player_covariation_scan.py` (100)               | `player_covariation_core.py`                                               | divergent-by-ruling                 | No         | Same D22/D23 pairing. Its `_has_finite_numeric` is one of the six AST-identical definitions, at two lines.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

### What has to happen before any of these retire

1. **The paired halves land first.** Three rows are `paired-open` in the
   divergence ledger (`launch_monitor_strokes_gained`,
   `launch_monitor_private_corpus`, `player_covariation`). Neither repo's change
   is correct alone, and `scripts/check_divergence_ledger.py` requires a
   `UD-PAIR:` reference in the PR body for any diff that touches them.
2. **The drift gates keep resolving.** UpstreamDrift's
   `tests/integration/launch_monitor_drift/` imports twenty symbols from ten
   `rate_of_closure` modules **by name**; a retirement that does not leave a
   re-export at the old path turns those gates red — or, worse, leaves them
   skipping green.
   `tests/rate_of_closure/test_launch_monitor_drift_gate_surface.py` pins that
   surface from the Tools side so the breakage is caught here first.
3. **The TypeScript twins are not re-pointed by this ADR.** The owner's
   2026-09-02 deferred-twin ruling keeps each twin a tracked follow-up, so a
   Python-only retirement of a twinned module would leave the two runtimes
   computing different things behind one pinned golden.

## Golden Fixtures & Parity Verification

Cross-client parity is governed by backend-authoritative golden fixtures in `src/rate_of_closure/web/src/model/__fixtures__/`:

- `launch_monitor_conformance_bundle_golden_v1.json`: Spans available and unavailable cases across all 5 analysis families.
- `launch_monitor_player_covariation_golden_v1.json`: Synthetic aggregation-reversal test vector.
- `launch_monitor_workspace_v3_golden.json`: Row-free workspace v3 round-trip fixture.

## Verification

```powershell
python -m pytest tests/rate_of_closure/test_launch_monitor*
python -m ruff check src/rate_of_closure/
cd src/rate_of_closure/web
npm test
npm run type-check
npm run lint
npm run build
```
