# F-TL-030

**Colebrook docstring says Newton-Raphson (code is fixed-point); turbulent f applied in 2300-4000 without flag**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | docs     | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/_friction_factors.py:70`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/_friction_factors.py:73`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/_friction_factors.py:95`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/_friction_factors.py:107`

## Finding

- Severity: Low - Category: docs
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/\_friction_factors.py:70-73, :95-96, :102-113
- Evidence: "turbulent flow, Re > 4000 … Solved iteratively using Newton-Raphson method" vs `f_new = 0.25 / (math.log10(term1 + term2) ** 2)` and `if reynolds_number < RE_LAMINAR_UPPER: return friction_factor_laminar(...)`.
- Failure scenario: fixed-point converges here (tests pass), but between Re 2300 and 4000 the function returns a turbulent f with no flag — Churchill exists for exactly that case.
- Proposed change: fix the docstring; flag `transitional=True` for 2300 ≤ Re < 4000 or route to Churchill.
- Effort: S

## Draft Pull Request

**Title:** `docs(pressure-drop): fix Colebrook docstring, flag transitional Re`

### Why This Matters (The Pitch)

The Colebrook docstring says Newton-Raphson but the code is a fixed-point iteration, and between Re 2300 and 4000 it returns a turbulent friction factor with no flag. Correct the docstring and either flag transitional flow or route it to Churchill, which exists for that case. A small honesty fix in a friction-factor module that is otherwise correct.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/pressure_drop_calculator/engine/_friction_factors.py:70-73, :95-96, :102-113` documents Newton-Raphson but computes `f_new = 0.25 / (math.log10(term1 + term2) ** 2)`, and applies the turbulent correlation for Re between 2300 and 4000 without a flag.

## Change

- [ ] Fix the docstring to describe the fixed-point iteration
- [ ] Flag `transitional=True` for 2300 <= Re < 4000, or route to Churchill

## Verification

Test that a Re in [2300, 4000) returns the transitional flag or the Churchill value.

## Risk and Rollback

Callers may see a new flag or slightly different transitional values. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-030

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
