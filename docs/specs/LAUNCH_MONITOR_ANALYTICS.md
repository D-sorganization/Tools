# Launch Monitor Analytics

Issue: [Tools #4205](https://github.com/D-sorganization/Tools/issues/4205)
Program epic: [UpstreamDrift #8364](https://github.com/D-sorganization/UpstreamDrift/issues/8364)

## Purpose

Launch Monitor Analytics is a primary Rate of Closure workspace in both PyQt6
and React/Vite. It preserves every imported CSV or JSON field and supports
flexible statistical analysis across any compatible numeric variables.

The public, source-traceable evidence database remains the separate
[Launch-Monitor-Data repository](https://github.com/D-sorganization/Launch-Monitor-Data).
That repository records source URLs, monitor identity, environment, measurement
status, reported and canonical units, aggregation level, and verification
checks. Published aggregates must remain aggregates; neither UI expands them
into fabricated shot rows.

## Shared Contract

Both surfaces implement contract version `1.0.0`:

| Capability | PyQt6 | React/Vite |
| --- | --- | --- |
| CSV/JSON record import with source columns retained | Yes | Yes |
| Any numeric outcome and multiple predictors | Yes | Yes |
| Pearson, Spearman, and Kendall correlation | Yes | Yes |
| Pairwise, listwise, and fail-closed missingness | Yes | Yes |
| Benjamini-Hochberg multiplicity correction | Yes | Yes |
| Multivariable OLS with coefficient intervals | Yes | Yes |
| R², adjusted R², RMSE, MAE, Durbin-Watson, influential count | Yes | Yes |
| Arbitrary categorical grouping | Yes | Yes |
| Dataset SHA-256 and complete JSON evidence export | Yes | Yes |
| Aggregate-regression and pooled `source::` safeguards | Yes | Yes |

The UpstreamDrift implementation exposes the same semantics through
`POST /tools/launch-monitor-analytics/analyze` and publishes its machine-readable
capability set through `GET /tools/launch-monitor-analytics/capabilities`.

## Convention and Provenance Boundary

Both tabs consume the immutable registry in
`shared/python/swing_sim/conventions` and its TypeScript twin. Labels are
**TrackMan-Comparable** and **Foresight-Comparable**. They identify sourced
parameter definitions, reference points, event times, frames, units, and
availability rules; they do not claim vendor-device emulation, certification,
or interchangeability.

Vendor-specific source fields are blocked from analysis pooled across multiple
monitor vendors. Cross-monitor comparison should use canonical fields only
after the reference point, time, frame, geometry, sign, unit, and availability
contracts have been reconciled.

## Statistical Interpretation

- Pairwise correlation uses the pair-specific sample count; listwise mode uses
  one complete-case population; fail-closed mode rejects missing or nonnumeric
  selections.
- Pearson intervals use Fisher's z transform. Spearman and Kendall intervals
  are intentionally omitted rather than represented as Pearson intervals.
- OLS uses complete cases for the selected outcome and predictors and rejects
  rank-deficient or undersized designs.
- Group results are calculated independently and keep their own sample counts,
  estimates, diagnostics, and warnings.
- Association and held-sample fit do not establish causality or transport to a
  different player, club, ball, environment, software release, or monitor.
- Aggregate correlations are descriptive and subject to ecological bias.
  Aggregate observations never enter regression.

## Verification

```powershell
python -m pytest tests/rate_of_closure/test_launch_monitor_analysis.py tests/rate_of_closure/test_launch_monitor_analytics_tab.py tests/rate_of_closure/test_primary_navigation.py
python -m ruff check src/rate_of_closure/launch_monitor_analysis.py src/rate_of_closure/ui/pyqt6/launch_monitor_analytics_tab.py
cd src/rate_of_closure/web
npm test
npm run type-check
npm run lint
npm run build
```
