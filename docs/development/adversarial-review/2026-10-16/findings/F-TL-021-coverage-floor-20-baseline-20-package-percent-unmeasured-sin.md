# F-TL-021

**Coverage floor 20 = baseline 20, package_percent {} unmeasured since 2026-09-03; ratchet tracks 29-line shim at 100 %**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | S      | proposed |

**Locations:** `pyproject.toml:435`, `config/coverage_baseline.json:3`, `config/coverage_baseline.json:4`, `config/coverage_policy.json:12`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: pyproject.toml:435; config/coverage_baseline.json:3-4; config/coverage_policy.json:12
- Evidence:

```json
"total_percent": 20.0,
"package_percent": {}
...
"src/shared/python/upstream_drift_tools": 100.0
```

- Failure scenario: six weeks after "update it from the coverage-trend artifact of the first green tests (3.11) gate", the measured total is still unrecorded; `max_total_drop_percent: 2.0` compares against a number that was never measured. The only 100 % package is the deprecated alias (`src/shared/python/upstream_drift_tools/__init__.py`, 29 lines); the 78k-LOC `sidekick` has no package ratchet.
- Why a newer engineer misses this: the JSON looks like data; it is a placeholder promoted to policy by being committed.
- Proposed change: record the real number from the last green run; replace the shim entry with `src/shared/python/sidekick/process_calculators: <measured>`; test that `coverage_baseline.total_percent > fail_under` strictly or carries a `measured_on` within 30 days.
- Verification: the test.
- Effort: S

## Draft Pull Request

**Title:** `chore(coverage): record the measured baseline and package ratchets`

### Why This Matters (The Pitch)

The coverage ratchet compares against a number nobody measured. The floor is 20 percent, the baseline is 20 percent, and package_percent has been empty since 2026-09-03, six weeks after a note said to fill it from the first green run. The only package held at 100 percent is the 29-line deprecated alias, while the 78k-LOC sidekick has no package ratchet. So the max_total_drop_percent of 2.0 guards a placeholder that was promoted to policy by being committed. The fix is to record the real number, replace the shim entry with the process_calculators package at its measured value, and add a test that the baseline strictly exceeds the floor or carries a measured_on date within 30 days. The maintainers get a ratchet that actually ratchets, and the numeric code that matters gets a guardrail.

### Proposed PR Body

## Problem

`pyproject.toml:435` sets `fail_under = 20`; `config/coverage_baseline.json:3-4` has `"total_percent": 20.0` and `"package_percent": {}`; `config/coverage_policy.json:12` tracks `src/shared/python/upstream_drift_tools` at 100.0, a 29-line shim.

## Change

- [ ] Record the measured total from the last green run
- [ ] Replace the shim entry with `src/shared/python/sidekick/process_calculators: <measured>`
- [ ] Add a test: `total_percent > fail_under` strictly, or a `measured_on` within 30 days

## Verification

The new baseline-freshness test passes.

## Risk and Rollback

A real baseline above 20 percent tightens the gate and may fail PRs that lower coverage; that is the intent. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-021

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
