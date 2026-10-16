# F-TL-005

**WGS extent clamped to [0,x_max]; reverse WGS never computed**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:372`, `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:384`, `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:388`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:372-388
- Evidence:

```python
x_max = min(n_CO_0, n_H2O_0)
candidates = [r for r in (root1, root2) if -1e-9 <= r <= x_max + 1e-9]
return float(min(max(x, 0.0), x_max))
```

- Failure scenario (run): feed CO 5 / H₂ 40 / CO₂ 40 / H₂O 5 at 1100 K (K = 0.845) → `conversion = 0.0`, composition unchanged; mass-action quotient at the returned composition is 64, not 0.845. Gasifier syngas at high T routinely sits on the reverse-WGS side.
- Why a newer engineer misses this: "extent between 0 and the limiting reactant" holds for an irreversible reaction; for an equilibrium the window is [−min(n_CO2, n_H2), min(n_CO, n_H2O)].
- Proposed change: `x_min = -min(n_CO2_0, n_H2_0)`; accept roots in `[x_min, x_max]`; clamp to that window.
- Verification: assert `(n_CO2·n_H2)/(n_CO·n_H2O) == pytest.approx(K, rel=1e-6)` on the returned composition for a CO₂/H₂-rich feed.
- Effort: S

## Draft Pull Request

**Title:** `fix(wgs): allow negative extent so reverse shift reaches equilibrium`

### Why This Matters (The Pitch)

The WGS equilibrium function returns a composition that is not at equilibrium. The reaction extent is clamped to between zero and the limiting reactant, which is right for an irreversible reaction and wrong for an equilibrium. For a feed of CO 5, H2 40, CO2 40, H2O 5 at 1100 K, where K is 0.845, it returns conversion 0.0 and an unchanged composition whose mass-action quotient is 64. Gasifier syngas at high temperature routinely sits on the reverse-shift side, so this is the regime users actually run. The fix widens the window to minus min(n_CO2, n_H2) through min(n_CO, n_H2O) and accepts the root inside it, a small edit to one function. The verification test checks the thing the function claims to do: that the returned composition satisfies the equilibrium constant. Anyone using the calculator for hot syngas conditioning benefits.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:372-388` restricts roots to `[-1e-9, x_max + 1e-9]` and clamps `x` to `[0, x_max]`. For CO 5 / H2 40 / CO2 40 / H2O 5 at 1100 K the function returns conversion 0.0 with a mass-action quotient of 64 instead of K = 0.845.

## Change

- [ ] Define `x_min = -min(n_CO2_0, n_H2_0)`
- [ ] Accept roots in `[x_min, x_max]`
- [ ] Clamp to that window instead of `[0, x_max]`
- [ ] Add a CO2/H2-rich equilibrium test

## Verification

Assert `(n_CO2*n_H2)/(n_CO*n_H2O) == pytest.approx(K, rel=1e-6)` on the returned composition for a CO2/H2-rich feed.

## Risk and Rollback

Results change only for feeds on the reverse-shift side; forward-shift cases are unchanged. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-005

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
