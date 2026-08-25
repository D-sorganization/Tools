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

| Capability | PyQt6 | React/Vite | Backend Authority |
| --- | --- | --- | --- |
| CSV/JSON record import with source columns retained | Yes | Yes | Tools local |
| 261,666-row governed private corpus loader | Yes (desktop) | API / reference | Campaign Parquet |
| Any numeric outcome and multiple predictors | Yes | Yes | Upstream v2 |
| Pearson, Spearman, and Kendall correlation | Yes | Yes | Upstream v2 |
| Pairwise, listwise, and fail-closed missingness | Yes | Yes | Upstream v2 |
| Benjamini-Hochberg multiplicity correction | Yes | Yes | Upstream v2 |
| Multivariable OLS with coefficient intervals | Yes | Yes | Upstream v2 |
| R², adjusted R², RMSE, MAE, Durbin-Watson, influential count | Yes | Yes | Upstream v2 |
| Within-player covariation and meta-analysis | Yes | Yes | Upstream v2 |
| Source-backed strokes gained with course state | Yes | Yes | Upstream v2 |
| Longitudinal session analytics and trends | Yes | Yes | Upstream v2 |
| Dataset SHA-256 and complete JSON evidence export | Yes | Yes | Tools v3 export |
| Row-free project save / load (Workspace v3) | Yes | Yes | Tools v3 schema |

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
