# F-TL-028

**Baghouse full-thermo mode is dead code behind HAS_THERMO=False; docstring says it is live**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | hygiene  | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/baghouse_calculator.py:52`, `src/shared/python/sidekick/process_calculators/baghouse_calculator.py:99`, `src/shared/python/sidekick/process_calculators/baghouse_calculator.py:229`

## Finding

- Severity: Low - Category: hygiene
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/baghouse_calculator.py:52-55, :97-100, :229-269
- Evidence: `HAS_THERMO: bool = False` … `if (self.thermo_calc is not None and HAS_THERMO and GasStream is not None and FlowUnit is not None):` — 40 unreachable lines; class docstring: "It can operate in two modes: 1. Full mode (with thermo module)".
- Failure scenario: per-file coverage is permanently depressed by unreachable lines; readers trust the docstring.
- Proposed change: delete :50-55 and :229-269; fix the docstring; if injection is wanted, accept a `Protocol` and test with a fake.
- Verification: `vulture` on the file reports nothing.
- Effort: S

## Draft Pull Request

**Title:** `refactor(baghouse): delete the unreachable HAS_THERMO path`

### Why This Matters (The Pitch)

The baghouse calculator has a full thermodynamic mode behind HAS_THERMO set to False, so about forty lines can never run, and the class docstring still advertises two modes. Delete the dead branch and fix the docstring: less to read, and per-file coverage stops being permanently depressed.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/baghouse_calculator.py:52-55, :97-100, :229-269` hard-codes `HAS_THERMO: bool = False`, making the guarded block of roughly 40 lines unreachable, while the class docstring describes a live full mode.

## Change

- [ ] Delete `:50-55` and `:229-269`
- [ ] Fix the class docstring
- [ ] If injection is wanted later, accept a `Protocol` and test it with a fake

## Verification

`vulture` on the file reports nothing.

## Risk and Rollback

The code is unreachable today, so behaviour is unchanged. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-028

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
