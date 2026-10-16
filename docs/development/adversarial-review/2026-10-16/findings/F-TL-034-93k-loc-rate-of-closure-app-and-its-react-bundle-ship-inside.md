# F-TL-034

**93k-LOC rate_of_closure app and its React bundle ship inside the shared-library wheel**

| Severity  | Category     | Confidence | Effort | Status   |
| --------- | ------------ | ---------- | ------ | -------- |
| 🟨 Medium | architecture | high       | L      | proposed |

**Locations:** `pyproject.toml:259`, `pyproject.toml:285`, `pyproject.toml:286`, `build_hooks.py:1`

## Finding

- Severity: Medium - Category: architecture
- Confidence: high
- Location: pyproject.toml:259, :270-290; build_hooks.py; src/rate_of_closure/ (465 .py, 651 .ts/.tsx)
- Evidence: `"rate_of_closure*"` in `include`; package-data `"web/dist/index.html", "web/dist/*.json", "web/dist/assets/*"`; `build_py = "build_hooks.RateWebBuildPy"`.
- Failure scenario: Gasification_Model installs `ud-tools` for `process_calculators` and receives a React bundle, PyQt windows and a FastAPI "loopback authority"; ruff/mypy/test shards all pay for it on every PR (1,821 test files). The API-contract policy explicitly excludes it, so the biggest package has the weakest contract.
- Proposed change: split into `ud-tools` (shared kernels) and `rate-of-closure` (app) with separate `pyproject.toml` and CI path filters — the `movement_optimizer` precedent already exists.
- Verification: `ud-tools` wheel < 5 MB and contains no `web/dist`.
- Effort: L

## Draft Pull Request

**Title:** `build(packaging): split rate_of_closure out of the shared ud-tools wheel`

### Why This Matters (The Pitch)

Gasification_Model installs ud-tools for process_calculators and receives a 93k-LOC golf application, a React bundle, PyQt windows and a FastAPI authority it never uses. Ruff, mypy and test shards all pay for that app on every PR across 1,821 test files, and the API-contract policy excludes it, so the biggest package has the weakest contract. The movement_optimizer package already shows the pattern: give rate_of_closure its own pyproject and CI path filters. The shared wheel gets small, the library consumers stop downloading a web bundle, and the shared CI gets faster. It is a large change, which is why the verification target is concrete: a wheel under 5 MB with no web/dist. Library consumers get a smaller install and the app team gets its own release cadence, so neither side has to wait on the other's CI.

### Proposed PR Body

## Problem

`pyproject.toml:259, :270-290` includes `"rate_of_closure*"` and package-data `web/dist/*`; `build_hooks.py` registers `RateWebBuildPy`. `src/rate_of_closure/` has 465 .py and 651 .ts/.tsx files.

## Change

- [ ] Create a separate `rate-of-closure` distribution with its own `pyproject.toml`, following the `movement_optimizer` precedent
- [ ] Remove `rate_of_closure*` and `web/dist` from `ud-tools` package config
- [ ] Add CI path filters for each distribution
- [ ] Update `build_hooks.py`

## Verification

The `ud-tools` wheel is under 5 MB and contains no `web/dist`.

## Risk and Rollback

Release and install paths change for anyone installing rate_of_closure from this wheel; stage with a deprecation note. Rollback is a revert of the packaging change.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: apply the same split to `p1am_control_system` if desired.

Refs: adversarial review 2026-10-16, F-TL-034

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
