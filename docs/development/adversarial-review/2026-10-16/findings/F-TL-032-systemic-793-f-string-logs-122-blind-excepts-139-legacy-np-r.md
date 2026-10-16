# F-TL-032

**Systemic: 793 f-string logs, 122 blind excepts, 139 legacy np.random, 44 global, 28 naive datetimes**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | hygiene  | high       | M      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/_friction_factors.py:110`, `src/rate_of_closure/web_authority/jobs.py:1`, `src/p1am_control_system/backend/historian_spool.py:1`, `src/p1am_control_system/backend/timescale_sink.py:1`

## Finding

- Severity: Low - Category: hygiene
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/\_friction_factors.py:110; src/rate_of_closure/web_authority/jobs.py (9 blind excepts); src/p1am_control_system/backend/historian_spool.py (6); src/p1am_control_system/backend/timescale_sink.py (5)
- Evidence: `_logger.debug(f"Colebrook converged in {i + 1} iterations: f = {f_new:.6f}")` — formatted even when DEBUG is off, inside a numeric loop. Counts from `ruff check src --select G004,BLE001,NPY002,PLW0603,DTZ --isolated`.
- Failure scenario: the repo's ruff `select` omits G, BLE, NPY, DTZ, PL, so none of this is visible; blind excepts in historian/timescale sinks are where an industrial logger swallows a disk-full error.
- Proposed change: add `"G", "BLE", "NPY", "DTZ"` to `select` with `--add-noqa` once, then ratchet.
- Verification: `ruff check src --select G004,BLE001,NPY002` → 0.
- Effort: M

## Draft Pull Request

**Title:** `chore(lint): enable G, BLE, NPY, DTZ with a one-time noqa baseline`

### Why This Matters (The Pitch)

The ruff selection omits logging-format, blind-except, legacy-numpy-RNG and naive-datetime rules, so 793 f-string log calls, 122 blind excepts, 139 legacy np.random calls and 28 naive datetimes are invisible. Blind excepts in the historian and timescale sinks are where an industrial logger swallows a disk-full error. Add the four rule families with a one-time add-noqa and ratchet down from there.

### Proposed PR Body

## Problem

`ruff.toml` select omits G, BLE, NPY, DTZ and PL. Counts from `ruff check src --select G004,BLE001,NPY002,PLW0603,DTZ --isolated`: 793 f-string log calls, 122 blind `except Exception`, 139 legacy `np.random.*`, 44 `global`, 28 naive datetimes; examples at `_friction_factors.py:110`, `src/rate_of_closure/web_authority/jobs.py`, `src/p1am_control_system/backend/historian_spool.py`, `timescale_sink.py`.

## Change

- [ ] Add `"G", "BLE", "NPY", "DTZ"` to `select`
- [ ] Run `--add-noqa` once to baseline
- [ ] Ratchet the noqa count down in later PRs

## Verification

`ruff check src --select G004,BLE001,NPY002` returns 0 after the baseline.

## Risk and Rollback

Adds noqa comments across many files, which is a noisy diff but behaviour-neutral. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-TL-032

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
