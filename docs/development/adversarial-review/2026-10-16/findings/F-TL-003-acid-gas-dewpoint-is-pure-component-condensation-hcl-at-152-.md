# F-TL-003

**Acid-gas dewpoint is pure-component condensation (HCl at -152 C), not aqueous acid dew point**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | numerics | high       | M      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py:483`, `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py:486`, `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py:332`, `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py:508`

## Finding

- Severity: High - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py:452-486, :499-513, :332-337
- Evidence:

```python
# Inverse Antoine equation: T = B / (A - log10(P_mmHg)) - C
return float(antoine_temperature_c(A, B, C, partial_pressure_pa))
...
self.validity_limits = {... "HCl": {"T_min": -85, "T_max": 51, ...}}   # declared, never checked
```

- Failure scenario (run): 15 % H₂O, 1000 ppm HCl, 2000 ppm H₂S at 1 bar, 120 °C → `hcl_dewpoint_c = -152.3`, `h2s_dewpoint_c = -140.0`, risk "VERY LOW". The quantity needed is the aqueous acid dew point (Kiang 1981 / Verhoff–Banchero): for the same stream Kiang gives 61 °C for HCl — above the water dew point (54 °C) the tool reports as limiting. In a 90 °C baghouse inlet the tool says "margin 36 °C"; the real HCl margin is 29 °C and shrinks fast with more HCl. The declared validity limits (`T_min = -85` for HCl) are never applied, so −152 °C is not even flagged.
- Why a newer engineer misses this: Antoine inversion is textbook-correct for a _pure_ vapour; this is a model-selection error, not an algebra error, so no unit test on the formula can catch it.
- Proposed change: add `calculate_acid_dewpoint(p_h2o_pa, p_acid_pa, acid)` implementing Kiang's correlations (H₂SO₄, HCl, HBr, HNO₃) and Okkes for SO₃; keep pure-component Antoine only for H₂O; return `nan`/raise outside `validity_limits`; make `limiting_component` consider acid dew points.
- Verification: `assert kiang_hcl(0.15*760, 0.001*760) == pytest.approx(61.4, abs=1.0)`; `calculate_dewpoint(100.0, "HCl")` raises or returns `nan`.
- Effort: M

## Draft Pull Request

**Title:** `fix(acid-gas): use aqueous acid dew point, not pure Antoine`

### Why This Matters (The Pitch)

The acid-gas dewpoint tool tells you HCl condenses at minus 152 C. For a stream with 15 percent H2O, 1000 ppm HCl and 2000 ppm H2S at 1 bar and 120 C, it reports risk VERY LOW, while the aqueous HCl dew point by Kiang's correlation is 61 C, above the 54 C water dew point the tool treats as limiting. In a 90 C baghouse inlet the tool claims a 36 C margin; the real margin is 29 C and shrinks fast as HCl rises. That is the non-conservative direction for corrosion: acid attack on a bag filter or duct that the calculator signed off. The declared validity limits are never applied, so the absurd number is not even flagged. The proposed change adds the right model, returns nan or raises outside validity limits, and makes the limiting component consider acid dew points. Engineers sizing gas cleaning and anyone trusting the tool for material selection benefit directly.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py:452-486, :499-513, :332-337` inverts the Antoine equation for each acid as if it were a pure vapour, and declares `validity_limits` (HCl `T_min = -85`) that are never checked. Result for 15 % H2O, 1000 ppm HCl, 2000 ppm H2S, 1 bar: `hcl_dewpoint_c = -152.3`, `h2s_dewpoint_c = -140.0`, risk VERY LOW.

## Change

- [ ] Add `calculate_acid_dewpoint(p_h2o_pa, p_acid_pa, acid)` implementing Kiang's correlations (H2SO4, HCl, HBr, HNO3) and Okkes for SO3
- [ ] Keep pure-component Antoine only for H2O
- [ ] Return nan or raise outside `validity_limits`
- [ ] Make `limiting_component` consider acid dew points

## Verification

`assert kiang_hcl(0.15*760, 0.001*760) == pytest.approx(61.4, abs=1.0)`; `calculate_dewpoint(100.0, "HCl")` raises or returns nan.

## Risk and Rollback

Reported dew points and risk ratings will change, mostly upward; downstream consumers that threshold on them need review. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-003

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
