# Coverage Gate Policy

## Current Gate

Coverage is enforced in CI via `scripts/check_coverage_policy.py` using:

- `config/coverage_policy.json` (minimum + package thresholds)
- `config/coverage_baseline.json` (non-regression baseline)

## Enforcement

- CI fails if total coverage drops below `minimum_total_percent`.
- CI fails if total coverage regresses by more than `max_total_drop_percent` from baseline.
- CI fails if tracked package coverage drops below configured thresholds.

## Trend Reporting

Each CI matrix run uploads `coverage_trend_<python>.json` as an artifact.
This provides machine-readable trend snapshots for review.
