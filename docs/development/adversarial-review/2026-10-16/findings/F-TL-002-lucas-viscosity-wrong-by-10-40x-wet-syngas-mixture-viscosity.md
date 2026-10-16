# F-TL-002

**Lucas viscosity wrong by 10-40x; wet syngas mixture viscosity 4x low**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | numerics | high       | M      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:555`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:586`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:598`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:651`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:700`

## Finding

- Severity: High - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:555-561, :586-592, :594-607, :651-659, :694-700
- Evidence:

```python
tr_diff = 0.292 - T_r if T_r < 0.292 else 0.0          # Lucas uses (0.292 - Z_c), not T_r
...
mu_low = (0.807 * ((M * T_c) ** 0.5) / ((props.critical_pressure / 1e6) ** (2 / 3)) * eta_low * 1e-7)
...
rho_r = P_r / T_r  # Approximate reduced density
mu = mu_low * (1.0 + delta_mu)        # JST is [(mu-mu0)·xi + 1e-4]^(1/4) = poly(rho_r), not mu0*(1+poly)
```

`SUTHERLAND_CONSTANTS` (651–659) covers only Air/N2/O2/CO2/H2/CO/CH4; everything else routes to Lucas (694–700).

- Failure scenario (run): `calculate_pure_gas_viscosity_lucas(300, 1e5, GAS_DATABASE["H2O"])` → 3.1e-7 Pa·s (reference 1.23e-5, ratio 0.03); H₂S 1.0e-6 (ref 1.26e-5); Ar 2.4e-6 (ref 2.27e-5). `calculate_gas_properties({"H2":.3,"CO":.3,"H2O":.2,"CO2":.2}, 800, 2e5)` → viscosity 9e-6 vs 3.5e-5 for the dry equivalent. Reynolds number 4× too high → friction factor too low → pipe ΔP and blower head under-predicted for any wet syngas line.
- Why a newer engineer misses this: the function has a citation, a docstring and a passing test (`assert mu > 0`). The correct Lucas form needs ξ = 0.176·(Tc/(M³Pc⁴))^(1/6) with Pc in bar and F_P built from Z_c; the code mixes three half-remembered formulas.
- Proposed change: replace the Lucas body with the Reid–Prausnitz–Poling form, or add Sutherland constants for H₂O (S≈1064 K, μ_ref 8.9e-6 @ 273 K), H₂S, Ar, and raise `ValueError` for species without constants instead of guessing.
- Verification: `assert mu == pytest.approx(ref, rel=0.10)` for H₂O/H₂S/Ar at 300 K and 600 K against NIST WebBook values.
- Effort: M

## Draft Pull Request

**Title:** `fix(gas-props): correct Lucas viscosity for wet syngas`

### Why This Matters (The Pitch)

Wet syngas lines are being designed with a viscosity about four times too low. The Lucas implementation mixes three half-remembered formulas, so H2O at 300 K comes out at 3.1e-7 Pa.s against a reference of 1.23e-5, and H2S and Ar are off by a similar order. A mixture of H2, CO, H2O and CO2 at 800 K reports 9e-6 where the dry equivalent is 3.5e-5. Reynolds number is therefore four times too high, the friction factor too low, and pipe pressure drop and blower head are under-predicted on exactly the streams gasification engineers care about. The only test is assert mu > 0, so it has passed all along. The change either rewrites the correlation in its Reid-Prausnitz-Poling form or adds Sutherland constants for the missing species and raises for anything it cannot compute rather than guessing. That is the right trade for a design calculator: a loud error costs minutes, a quiet wrong viscosity costs a blower.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:555-561, :586-592, :594-607, :651-659, :694-700` implements Lucas with `(0.292 - T_r)` where Z_c is required, a wrong prefactor and a `mu_low * (1 + delta_mu)` dense-gas term. Sutherland constants cover only seven species, so everything else routes to Lucas. `calculate_pure_gas_viscosity_lucas(300, 1e5, GAS_DATABASE["H2O"])` returns 3.1e-7 Pa.s versus 1.23e-5 reference.

## Change

- [ ] Replace the Lucas body with the Reid-Prausnitz-Poling form (xi = 0.176\*(Tc/(M^3 Pc^4))^(1/6), Pc in bar, F_P from Z_c)
- [ ] Or add Sutherland constants for H2O (S about 1064 K, mu_ref 8.9e-6 at 273 K), H2S and Ar
- [ ] Raise `ValueError` for species without constants instead of guessing
- [ ] Replace `assert mu > 0` tests with reference values

## Verification

`assert mu == pytest.approx(ref, rel=0.10)` for H2O, H2S and Ar at 300 K and 600 K against NIST WebBook values.

## Risk and Rollback

Pressure-drop outputs for wet gas will shift upward; this is the intended correction. New `ValueError` for unsupported species could surface in callers that relied on silent output. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-009 (Z-factor clamp) is in the same file.

Refs: adversarial review 2026-10-16, F-TL-002

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
