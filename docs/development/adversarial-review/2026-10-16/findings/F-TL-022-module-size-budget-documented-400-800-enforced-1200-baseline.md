# F-TL-022

**Module-size budget documented 400/800, enforced 1200 + baseline pointing at deleted files**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | S      | proposed |

**Locations:** `AGENTS.md:175`, `.github/workflows/ci-standard.yml:487`, `config/module_size_budget_baseline.json:5`, `config/module_size_budget_baseline.json:6`, `config/module_size_budget_baseline.json:14`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: AGENTS.md:175; .github/workflows/ci-standard.yml:487; config/module_size_budget_baseline.json:5-6, :14
- Evidence: AGENTS.md:175 "DO NOT create files exceeding 400 lines. Files >800 lines are critical violations." CI: `python3 scripts/check_module_size_budget.py --max-lines 1200 --include src`. Baseline `src/electrode_advisor/...main_window.py: 4386`, `.../upstream_drift_tools/.../syngas_compression_calculator.py: 1218` — neither exists.
- Failure scenario: 433 src files exceed 400 lines, 65 exceed 800, 4 exceed 1200 (`rotation_converter/modern_robotics.py` 2123, `chat/_chat_dock_widget_qt.py` 1551, `p1am .../main.py` 1433, `rotation_converter/ui/pyqt6/main_window.py` 1302). Agents read 400; CI enforces 1200 + allowlist; every agent either violates the doc or splits files CI never asked about.
- Why a newer engineer misses this: the number in prose is aspirational; the number in YAML is law; nobody reconciles them.
- Proposed change: pick one number (800), put it in `config/`, have AGENTS.md and the script read it; make `check_module_size_budget.py` fail on baseline paths that no longer exist (pattern already in `ci_test_shards.py --check`).
- Verification: `python3 scripts/check_module_size_budget.py --max-lines 800` passes with an honest baseline.
- Effort: S

## Draft Pull Request

**Title:** `ci(size-budget): enforce one limit and fail on dead baseline paths`

### Why This Matters (The Pitch)

The repo documents a 400-line limit with 800 as critical, and CI enforces 1200 plus an allowlist. 433 source files exceed 400 lines, 65 exceed 800 and 4 exceed 1200, so agents either violate the document or split files CI never asked about. Three of 13 baseline entries point at files that were deleted, including a 4386-line main_window, and nothing notices. The change picks a single number, 800, puts it in config where the script and AGENTS.md both read it, and makes the checker fail on baseline paths that no longer exist, using the pattern ci_test_shards.py already has. Contributors and agents get one rule they can predict, and the baseline stops accumulating dead weight. The payoff is that agents stop guessing which number is real, and review comments about file length stop being arguments about folklore.

### Proposed PR Body

## Problem

`AGENTS.md:175` says files over 400 lines are discouraged and over 800 critical; `.github/workflows/ci-standard.yml:487` runs `check_module_size_budget.py --max-lines 1200`; `config/module_size_budget_baseline.json:5-6, :14` lists deleted files (e.g. `src/electrode_advisor/...main_window.py: 4386`).

## Change

- [ ] Choose 800 and store it in `config/`
- [ ] Have AGENTS.md and `scripts/check_module_size_budget.py` read it
- [ ] Fail the checker on baseline paths that no longer exist
- [ ] Regenerate an honest baseline

## Verification

`python3 scripts/check_module_size_budget.py --max-lines 800` passes with an honest baseline.

## Risk and Rollback

Lowering the CI limit from 1200 to 800 requires baseline entries for existing large files. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-TL-022

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
