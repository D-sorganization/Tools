# F-TL-010

**Polytropic compression branch is isentropic-with-efficiency; discharge T ignores inefficiency**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:272`, `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:273`, `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:279`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:271-280
- Evidence:

```python
elif stage.compression_type == "polytropic":
    n = gamma
    temp_out_actual = stage.inlet_temperature * (pr ** ((n - 1) / n))
    work_actual = ((n / (n - 1)) * self.R * stage.inlet_temperature * (pr ** ((n - 1) / n) - 1) / stage.efficiency)
```

- Failure scenario: η_p = 0.75, PR = 3, γ = 1.4 → code T_out = T_in·3^0.286 (same as isentropic); the polytropic exponent (n−1)/n = (γ−1)/(γ·η_p) = 0.381 gives T_out ≈ 45 K hotter at 400 K inlet. Aftercooler duty and the downstream water-dropout (computed by this very engine) are under-sized.
- Why a newer engineer misses this: "polytropic" and "isentropic efficiency" are both ways to describe a real compressor; mixing them produces plausible numbers.
- Proposed change: `n_exp = (gamma - 1) / (gamma * stage.efficiency)`; `temp_out = T_in * pr**n_exp`; `work = (R*T_in/n_exp) * (pr**n_exp - 1)`.
- Verification: assert polytropic T_out > isentropic T_out for η < 1.
- Effort: S

## Draft Pull Request

**Title:** `fix(compression): use polytropic exponent for discharge temperature`

### Why This Matters (The Pitch)

The compression engine's polytropic mode is an isentropic calculation with an efficiency divide bolted on, so the discharge temperature ignores inefficiency entirely. At eta_p 0.75, pressure ratio 3 and gamma 1.4 the code returns the isentropic outlet temperature; the polytropic exponent (gamma minus 1) over (gamma times eta) gives roughly 45 K hotter gas at a 400 K inlet. That error flows straight into aftercooler duty and into the water dropout the same engine computes downstream, both of which come out under-sized. The fix is three lines using the correct exponent, and the test is as simple as asserting that polytropic outlet temperature exceeds isentropic for any efficiency below one. Anyone sizing syngas compression trains and their coolers benefits. The people who feel it are whoever sizes the aftercooler and the knockout drum downstream.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:271-280` sets `n = gamma` for the polytropic branch and computes `temp_out_actual = T_in * pr ** ((n-1)/n)`, identical to isentropic. At eta_p = 0.75, PR = 3, gamma = 1.4 the correct exponent (gamma-1)/(gamma\*eta_p) = 0.381 gives about 45 K hotter discharge at 400 K inlet.

## Change

- [ ] `n_exp = (gamma - 1) / (gamma * stage.efficiency)`
- [ ] `temp_out = T_in * pr**n_exp`
- [ ] `work = (R*T_in/n_exp) * (pr**n_exp - 1)`

## Verification

Assert polytropic T_out > isentropic T_out for efficiency below 1.

## Risk and Rollback

Polytropic-mode discharge temperatures and work rise; isentropic mode is untouched. Rollback is a revert.

## Handoff

Follow-ups: F-TL-011 is in the same file.

Refs: adversarial review 2026-10-16, F-TL-010

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
