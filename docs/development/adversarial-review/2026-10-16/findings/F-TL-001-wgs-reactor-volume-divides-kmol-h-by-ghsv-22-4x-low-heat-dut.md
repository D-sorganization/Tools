# F-TL-001

**WGS reactor volume divides kmol/h by GHSV (22.4x low); heat duty off by 1000\*y_CO**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:444`, `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:456`, `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:553`

## Finding

- Severity: High - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:441-457, :553
- Evidence:

```python
ghsv = WGS_TYPICAL_GHSV  # h^-1 (typical for WGS)
reactor_volume = feed_rate / ghsv  # m3
...
heat_duty = (
    feed_rate * conversion / 100 * WGS_HEAT_KJ_PER_MOL / KJ_HR_TO_KW
)  # kW
```

GUI (:553): `self.feed_rate.setSuffix(" kmol/h")`.

- Failure scenario: 100 kmol/h feed, 25 % CO, 673 K → code returns `reactor_volume=0.033 m³`, `heat_duty=991 kW`. GHSV is defined on standard volumetric flow: V = 100 kmol/h × 22.414 m³/kmol / 3000 h⁻¹ = 0.747 m³ (22.4× larger). Heat duty should use CO moles shifted: 25 kmol/h × 0.8663 × 41.2 kJ/mol × 1000/3600 = 248 kW, not 991 kW. A vessel bought from this number is one-third the diameter it needs.
- Why a newer engineer misses this: the variable is a bare float; nothing says "kmol/h" vs "Nm³/h". Unit mixing only shows up when someone recomputes by hand.
- Proposed change: in `size_wgs_reactor`, rename `feed_rate_kmol_h`; `q_std_m3_h = feed_rate_kmol_h * MOLAR_VOLUME_STP_OLD * 1000`; `reactor_volume = q_std_m3_h / ghsv`; take `co_fraction` as an argument: `heat_duty_kw = feed_rate_kmol_h * 1000 * co_fraction * conversion/100 * WGS_HEAT_KJ_PER_MOL / 3600`.
- Verification: `assert sizing["reactor_volume"] == pytest.approx(0.747, rel=1e-3)`; `assert sizing["heat_duty"] == pytest.approx(248, rel=1e-2)` for the inputs above.
- Effort: S

## Draft Pull Request

**Title:** `fix(wgs): size reactor on standard volumetric flow, not kmol/h`

### Why This Matters (The Pitch)

A reactor sized from this calculator comes out 22.4 times too small by volume. The code divides a feed rate in kmol/h by a GHSV defined on standard volumetric flow and labels the quotient m3, so 100 kmol/h of feed returns 0.033 m3 where the right answer is 0.747 m3. The heat duty is wrong in the same breath: it multiplies the whole feed by conversion instead of the CO moles actually shifted, giving 991 kW where the correct figure is 248 kW. Anyone buying a vessel or sizing a cooler from these numbers gets a unit one-third the diameter it needs. The fix is a handful of lines in one function, with two reference assertions that fail loudly the next time someone mixes units. Gasification engineers using the sidekick calculator, and everyone downstream in Gasification_Model, benefit first. This is the cheapest high-severity money fix in the queue: small effort, concrete before-and-after numbers, and a test that finally checks a number instead of checking that a number exists.

### Proposed PR Body

## Problem

`size_wgs_reactor` in `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:441-457` computes `reactor_volume = feed_rate / ghsv` (kmol/h divided by an h^-1 GHSV, labelled m3) and a heat duty that multiplies the full feed by conversion. The GUI at `:553` sets the suffix to kmol/h. For 100 kmol/h, 25 % CO, 673 K it returns 0.033 m3 and 991 kW; the correct values are 0.747 m3 and 248 kW.

## Change

- [ ] Rename the argument to `feed_rate_kmol_h`
- [ ] Convert to standard volumetric flow: `q_std_m3_h = feed_rate_kmol_h * MOLAR_VOLUME_STP_OLD * 1000`; `reactor_volume = q_std_m3_h / ghsv`
- [ ] Add `co_fraction` argument; `heat_duty_kw = feed_rate_kmol_h * 1000 * co_fraction * conversion/100 * WGS_HEAT_KJ_PER_MOL / 3600`
- [ ] Update GUI call site and replace the `> 0` assertions with reference values

## Verification

`assert sizing["reactor_volume"] == pytest.approx(0.747, rel=1e-3)` and `assert sizing["heat_duty"] == pytest.approx(248, rel=1e-2)` for 100 kmol/h, 25 % CO, 673 K.

## Risk and Rollback

Callers get materially different (corrected) numbers and the signature gains an argument; any caller that passes positional arguments needs updating. Rollback is a plain revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-005 and F-TL-006 touch the same file.

Refs: adversarial review 2026-10-16, F-TL-001

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
