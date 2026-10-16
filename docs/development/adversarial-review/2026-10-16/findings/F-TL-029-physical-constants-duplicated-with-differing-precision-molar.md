# F-TL-029

**Physical constants duplicated with differing precision; MOLAR_VOLUME_STP_ATM aliases \_OLD**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/constants.py:365`, `src/shared/python/sidekick/process_calculators/constants.py:813`, `src/shared/python/sidekick/process_calculators/constants.py:819`, `src/shared/python/sidekick/process_calculators/constants.py:308`

## Finding

- Severity: Low - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/constants.py:364-376, :812-822, :308
- Evidence: `MW_O2 = 0.031998` vs `MW_O2_KG = 0.032`; `MW_AR_KG = 0.040` (actual 0.039948); `MW_H2_KG = 0.002` (actual 0.0020159); `MOLAR_VOLUME_STP_ATM: Final[float] = MOLAR_VOLUME_STP_OLD`.
- Failure scenario: two calculators in the same package disagree on syngas MW by up to 0.8 %; `*_STP_ATM` aliasing `*_OLD` while `STP_PRESSURE_PA` means 1 bar invites the wrong basis.
- Proposed change: one `MW_*` set (NIST, 6 s.f., kg/mol); delete the `_KG` set; rename `MOLAR_VOLUME_STP_ATM` → `MOLAR_VOLUME_0C_1ATM`.
- Verification: test that every `MW_*` is within 0.05 % of NIST.
- Effort: S

## Draft Pull Request

**Title:** `refactor(constants): one MW set and unambiguous STP names`

### Why This Matters (The Pitch)

Physical constants appear twice with different precision, and two calculators in the same package disagree on syngas molecular weight by up to 0.8 percent. The aliases also mislead: MOLAR_VOLUME_STP_ATM points at the \_OLD value while STP_PRESSURE_PA means 1 bar. One NIST-based MW set and a clear 0C/1atm name remove a small, steady source of wrong-basis bugs.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/constants.py:364-376, :812-822, :308` define `MW_O2 = 0.031998` and `MW_O2_KG = 0.032`, `MW_AR_KG = 0.040` (actual 0.039948), `MW_H2_KG = 0.002` (actual 0.0020159), and `MOLAR_VOLUME_STP_ATM = MOLAR_VOLUME_STP_OLD`.

## Change

- [ ] Keep one `MW_*` set (NIST, 6 s.f., kg/mol) and delete the `_KG` set
- [ ] Rename `MOLAR_VOLUME_STP_ATM` to `MOLAR_VOLUME_0C_1ATM`

## Verification

Test that every `MW_*` is within 0.05 % of NIST.

## Risk and Rollback

Small numeric shifts (up to 0.8 % in mixture MW) in dependent calculators; golden values may need refreshing. Rollback is a revert.

## Handoff

Follow-ups: F-TL-004 removes `MW_DEFAULT_KG` in the same file family.

Refs: adversarial review 2026-10-16, F-TL-029

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
