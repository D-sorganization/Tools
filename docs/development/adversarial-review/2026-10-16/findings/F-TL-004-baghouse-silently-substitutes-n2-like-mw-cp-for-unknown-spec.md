# F-TL-004

**Baghouse silently substitutes N2-like MW/Cp for unknown species keys; no composition normalisation**

| Severity | Category    | Confidence | Effort | Status   |
| -------- | ----------- | ---------- | ------ | -------- |
| 🟧 High  | correctness | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/baghouse_calculator.py:156`, `src/shared/python/sidekick/process_calculators/baghouse_calculator.py:157`, `src/shared/python/sidekick/process_calculators/baghouse_calculator.py:190`

## Finding

- Severity: High - Category: correctness
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/baghouse_calculator.py:155-161, :189-197
- Evidence:

```python
cp_mol += frac * cp_data.get(species, CP_DEFAULT_FALLBACK)
mw_avg += frac * mw_data.get(species, MW_DEFAULT_KG)
```

- Failure scenario (run): `{"H2":0.5,"CO":0.5}` → ACFM 17,918, A/C 8.96 ft/min. `{"h2":0.5,"co":0.5}` → ACFM 9,596, A/C 4.80. `{"H2":50,"CO":50}` (percent, as the WGS GUI uses) → ACFM 179, A/C 0.09. No warning, no error. A baghouse sized at A/C 4.8 that actually sees 9 ft/min blinds in weeks.
- Why a newer engineer misses this: `dict.get(key, default)` reads as defensive programming; in a physical-property lookup a default is a lie with units.
- Proposed change: validate at entry: `unknown = set(composition) - set(mw_data); if unknown: raise ValueError(...)`; `if not 0.99 <= sum(composition.values()) <= 1.01: raise ValueError(...)`. Delete `MW_DEFAULT_KG`, `CP_DEFAULT_FALLBACK`, `CP_MASS_DEFAULT_FALLBACK`.
- Verification: `pytest.raises(ValueError)` for the lowercase and percent inputs above.
- Effort: S

## Draft Pull Request

**Title:** `fix(baghouse): reject unknown species and unnormalised composition`

### Why This Matters (The Pitch)

A baghouse can be sized at half the real air-to-cloth ratio because someone typed a gas name in lowercase. The calculator uses dict.get with a default for every species it does not recognise, so {"h2": 0.5, "co": 0.5} yields 9,596 ACFM and A/C 4.80 where {"H2": 0.5, "CO": 0.5} yields 17,918 ACFM and A/C 8.96. Pass percent, as the WGS GUI does, and {"H2": 50, "CO": 50} gives ACFM 179 and A/C 0.09. No warning, no error. A bag filter sized on the low figure that actually sees about 9 ft/min blinds in weeks. The fix is validation at entry plus deleting the three fallback constants, which makes the defaults impossible to reintroduce by accident. Anyone sizing gas cleaning downstream of the gasifier benefits, and the whole change is small, with the failing inputs already written as test cases.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/baghouse_calculator.py:155-161, :189-197` uses `mw_data.get(species, MW_DEFAULT_KG)` and `cp_data.get(species, CP_DEFAULT_FALLBACK)` and never normalises composition. Lowercase keys and percent inputs silently change ACFM and A/C by 2x to 100x.

## Change

- [ ] Validate at entry: `unknown = set(composition) - set(mw_data)` raises `ValueError`
- [ ] Raise `ValueError` unless `0.99 <= sum(composition.values()) <= 1.01`
- [ ] Delete `MW_DEFAULT_KG`, `CP_DEFAULT_FALLBACK`, `CP_MASS_DEFAULT_FALLBACK`
- [ ] Add tests for lowercase and percent inputs

## Verification

`pytest.raises(ValueError)` for `{"h2":0.5,"co":0.5}` and `{"H2":50,"CO":50}`.

## Risk and Rollback

Callers that currently pass percent or nonstandard keys will start raising; that is the intent but they will need a one-line fix. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-011 and F-TL-029 touch related defaults.

Refs: adversarial review 2026-10-16, F-TL-004

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
