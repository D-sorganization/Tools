# F-TL-038

**Meta-test/script volume: 117 scripts (26.8k LOC), 35 check\_\*.py, six coverage scripts, 12.9k LOC of YAML-parsing tests**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | L      | proposed |

**Locations:** `scripts/check_coverage_gates.py:1`, `scripts/verify_coverage.py:1`, `scripts/run_coverage.sh:1`, `scripts/check_coverage_policy.py:1`, `scripts/check_sidekick_coverage.py:1`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: scripts/check_coverage_gates.py; scripts/check_coverage_policy.py; scripts/check_sidekick_coverage.py; scripts/measure_coverage.py; scripts/run_coverage.sh; scripts/verify_coverage.py; tests/ops/ (30 files); tests/architecture/ (43 files)
- Evidence: 60 test files reference `.github/workflows` or `.yml`; six coverage scripts for one coverage number.
- Failure scenario: the governance layer is ~8 % of the Python in the repo with no consumer except CI. Every workflow edit needs a test edit; this is the maintenance cost that produces F-TL-019..024 — the layers drift faster than anyone reconciles them.
- Proposed change: cap `scripts/` at 10k LOC and `tests/ops+architecture` at 4k; delete `check_coverage_gates.py`, `verify_coverage.py`, `run_coverage.sh`; one `test_workflow_invariants.py`.
- Effort: L

## Draft Pull Request

**Title:** `chore(governance): cap scripts and meta-tests, drop overlapping tooling`

### Why This Matters (The Pitch)

The governance layer is about 8 percent of the Python in the repo and has no consumer except CI: 117 scripts at 26.8k lines, 35 check scripts at 6.2k, six overlapping coverage scripts for one coverage number, and 12.9k lines of tests that parse YAML and markdown. Every workflow edit needs a matching test edit, and the drift between layers is what produced the stale gates in F-TL-019 through 024. The change caps scripts at 10k lines and the ops and architecture tests at 4k, deletes check_coverage_gates.py, verify_coverage.py and run_coverage.sh, and consolidates into one test_workflow_invariants.py. Maintainers get fewer files to keep consistent, faster CI, and smaller PRs when workflows change. The tests that remain should each name the human decision they protect.

### Proposed PR Body

## Problem

`scripts/check_coverage_gates.py`, `check_coverage_policy.py`, `check_sidekick_coverage.py`, `measure_coverage.py`, `run_coverage.sh` and `verify_coverage.py` all serve one coverage number; `tests/ops/` (30 files) and `tests/architecture/` (43 files) include 60 test files that reference `.github/workflows` or `.yml`.

## Change

- [ ] Cap `scripts/` at 10k LOC and `tests/ops` plus `tests/architecture` at 4k
- [ ] Delete `check_coverage_gates.py`, `verify_coverage.py` and `run_coverage.sh`
- [ ] Consolidate workflow checks into one `test_workflow_invariants.py`

## Verification

Line counts of `scripts/` and `tests/ops` plus `tests/architecture` are under the caps and CI is green.

## Risk and Rollback

Large deletion; do it in stages so each removed check can be reinstated individually. Rollback is a revert of the stage.

## Handoff

Follow-ups: overlaps with F-TL-023 and F-TL-019.

Refs: adversarial review 2026-10-16, F-TL-038

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
