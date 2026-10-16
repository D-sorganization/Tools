# F-TL-025

**76 same-named test files duplicated between tests/ and src/**/tests; tests/ copies import deprecated alias\*\*

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | hygiene  | high       | M      | proposed |

**Locations:** `tests/shared/python/process_calculators/test_baghouse_calculator.py:9`, `src/shared/python/sidekick/tests/process_calculators/test_baghouse_calculator.py:2`

## Finding

- Severity: Medium - Category: hygiene
- Confidence: high
- Location: tests/shared/python/process_calculators/test_baghouse_calculator.py:9; src/shared/python/sidekick/tests/process_calculators/test_baghouse_calculator.py:2 (and 75 more pairs: test_wgs_reactor_calculator.py, test_acid_gas_dewpoint_calculator.py, test_constants.py, test_contracts.py, …)
- Evidence: `from upstream_drift_tools.process_calculators.baghouse_calculator import (` (tests/) vs `from sidekick.process_calculators.baghouse_calculator import (` (src/). 125 vs 77 lines; ~60 % overlap.
- Failure scenario: each fix needs two edits; each DeprecationWarning from `upstream_drift_tools` is emitted by the repo's own suite; `ci_test_shards.py` runs both. When they diverge, nobody knows which is canonical.
- Why a newer engineer misses this: two agents each "added tests" in the directory their prompt mentioned.
- Proposed change: `tests/` for cross-module/contract tests, `src/<pkg>/tests` for unit tests; delete duplicates; migrate imports to `shared.python.sidekick`.
- Verification: `comm -12 <(…tests…) <(…src/**/tests…)` count drops to 0.
- Effort: M

## Draft Pull Request

**Title:** `test: collapse 76 duplicated test files into one canonical tree`

### Why This Matters (The Pitch)

Seventy-six test files exist twice, once under tests/ and once under src/\*\*/tests, with about 60 percent overlap, and the tests/ copies import the deprecated upstream_drift_tools alias. Every fix needs two edits, every DeprecationWarning from that alias is emitted by the repo's own suite, and when the copies diverge nobody knows which is canonical. ci_test_shards.py runs both, so CI pays twice. The change keeps tests/ for cross-module and contract tests, keeps src/<pkg>/tests for unit tests, deletes the duplicates and migrates imports to the sidekick path. An estimated 8 to 10k lines go away, along with one class of confusion for agents asked to add tests. Faster shards, fewer warnings, one place to look. Deleting the copies also halves the places a future numeric fix has to be remembered, which matters given how many of this review's findings are in files with two test suites.

### Proposed PR Body

## Problem

`tests/shared/python/process_calculators/test_baghouse_calculator.py:9` imports `upstream_drift_tools.process_calculators...` while `src/shared/python/sidekick/tests/process_calculators/test_baghouse_calculator.py:2` imports `sidekick.process_calculators...`; 125 vs 77 lines, about 60 % overlap, 76 pairs in total including `test_wgs_reactor_calculator.py` and `test_acid_gas_dewpoint_calculator.py`.

## Change

- [ ] Decide the split: `tests/` for cross-module and contract tests, `src/<pkg>/tests` for unit tests
- [ ] Merge unique cases from each pair, delete the duplicate
- [ ] Migrate remaining imports to `shared.python.sidekick`
- [ ] Update `ci_test_shards.py` if it lists paths

## Verification

`comm -12` of the same-named test file lists between `tests/` and `src/**/tests` returns 0.

## Risk and Rollback

Risk of dropping a unique case when merging pairs; review each pair diff. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-026 for strengthening the surviving assertions.

Refs: adversarial review 2026-10-16, F-TL-025

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
