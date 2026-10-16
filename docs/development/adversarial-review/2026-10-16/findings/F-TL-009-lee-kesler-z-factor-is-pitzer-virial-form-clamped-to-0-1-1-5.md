# F-TL-009

**'Lee-Kesler' Z-factor is Pitzer virial form clamped to [0.1,1.5]**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:420`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:469`, `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:479`

## Finding

- Severity: Low - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:417-420, :468-479
- Evidence: docstring "Lee-Kesler correlation for Z-factor"; body `B0 = 0.083 - 0.422 / (T_r**1.6)` … `Z = max(0.1, min(Z, 1.5))`.
- Failure scenario: the virial form is valid for roughly P_r ≲ T_r/2; beyond that it goes negative and the clamp returns 0.1 — a 10× density error presented as a physical bound. Kay's rule for H₂-rich mixtures also needs the Newton correction.
- Why a newer engineer misses this: both correlations use ω, Tr, Pr; the clamp makes the output always look plausible.
- Proposed change: rename to `calculate_compressibility_pitzer_virial`, document validity, `warnings.warn`/raise instead of clamping when `P_r > 0.5 * T_r`.
- Verification: test that Z for CH₄ at Tr = 1.5, Pr = 3 raises/warns rather than returning 0.1.
- Effort: S

## Draft Pull Request

**Title:** `fix(gas-props): rename Pitzer virial Z and stop clamping to 0.1`

### Why This Matters (The Pitch)

The function named Lee-Kesler is actually the Pitzer second-virial form, and it clamps its answer to between 0.1 and 1.5. Beyond roughly reduced pressure above half the reduced temperature the virial form goes negative, the clamp returns 0.1, and density is wrong by ten times while looking like a physical bound. Renaming the function and warning or raising outside its validity range keeps the output honest and costs a few lines.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py:417-420, :468-479` documents Lee-Kesler but computes `B0 = 0.083 - 0.422 / (T_r**1.6)` and then `Z = max(0.1, min(Z, 1.5))`. Beyond roughly P_r > T_r/2 the virial form goes negative and the clamp returns 0.1.

## Change

- [ ] Rename to `calculate_compressibility_pitzer_virial` and document validity
- [ ] `warnings.warn` or raise instead of clamping when `P_r > 0.5 * T_r`

## Verification

Test that Z for CH4 at Tr = 1.5, Pr = 3 raises or warns rather than returning 0.1.

## Risk and Rollback

Callers of the old name need the new one (keep an alias if needed); out-of-range inputs now warn. Rollback is a revert.

## Handoff

Follow-ups: F-TL-002 is in the same file.

Refs: adversarial review 2026-10-16, F-TL-009

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
