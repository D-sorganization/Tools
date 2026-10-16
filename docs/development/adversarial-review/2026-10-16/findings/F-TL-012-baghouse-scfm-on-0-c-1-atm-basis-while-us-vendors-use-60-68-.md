# F-TL-012

**Baghouse SCFM on 0 C/1 atm basis while US vendors use 60/68 F**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | numerics | medium     | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/baghouse_calculator.py:203`

## Finding

- Severity: Low - Category: numerics
- Confidence: medium
- Location: src/shared/python/sidekick/process_calculators/baghouse_calculator.py:202-203
- Evidence: `vol_std_m3_s = molar_flow * R_UNIVERSAL * STP_TEMPERATURE_K / ATM_PA` with `STP_TEMPERATURE_K = 273.15`.
- Failure scenario: an SCFM figure handed to a US filter vendor is 7.3 % (293.15/273.15) below what they will compute; the unit-converter package in the same repo already offers STP/NTP/`STP_OLD` bases.
- Proposed change: expose `standard_basis` and label the result `flow_scfm_<basis>`.
- Verification: unit test for each basis.
- Effort: S

## Draft Pull Request

**Title:** `feat(baghouse): expose the standard-condition basis for SCFM`

### Why This Matters (The Pitch)

The baghouse calculator reports SCFM on a 0 C, 1 atm basis, while US filter vendors quote at 60 F or 68 F. A figure handed to a vendor is 7.3 percent below what they compute from the same gas. The repo's unit-converter package already offers the common bases, so exposing a standard_basis option and labelling the result with it is a small change that stops a quiet mismatch in vendor conversations.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/baghouse_calculator.py:202-203` computes `vol_std_m3_s = molar_flow * R_UNIVERSAL * STP_TEMPERATURE_K / ATM_PA` with `STP_TEMPERATURE_K = 273.15`, so SCFM is 7.3 % (293.15/273.15) below a 68 F basis.

## Change

- [ ] Add a `standard_basis` parameter, defaulting to current behaviour
- [ ] Label the result `flow_scfm_<basis>`
- [ ] Add one unit test per basis

## Verification

Unit test for each basis.

## Risk and Rollback

Default behaviour unchanged; only the output key naming is new. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-012

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
