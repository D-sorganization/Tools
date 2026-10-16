# F-TL-011

**Compression engine mole-averages gamma and silently drops unknown species**

| Severity  | Category    | Confidence | Effort | Status   |
| --------- | ----------- | ---------- | ------ | -------- |
| 🟨 Medium | correctness | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:171`, `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:172`, `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:178`

## Finding

- Severity: Medium - Category: correctness
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:168-178
- Evidence:

```python
if not species:
    logger.warning("Species %s not found in database, using defaults", comp)
    continue          # nothing is "used"; the fraction is simply lost
mix_gamma += frac * gamma
```

- Failure scenario: `{"H2":0.4,"CO":0.3,"NH3":0.3}` → mixture MW, Tc, Pc, γ computed from 70 % of the gas and not renormalised; MW low by 30 %, power wrong by the same. γ_mix should be Cp_mix/(Cp_mix − R), not Σ yᵢγᵢ (1–2 % for H₂/CO₂ blends).
- Why a newer engineer misses this: the warning text promises a default; the code never implements one.
- Proposed change: raise `ValueError` on unknown species; `cp_mix = Σ y_i·γ_i/(γ_i−1)·R`, `gamma_mix = cp_mix/(cp_mix − R)`.
- Verification: `pytest.raises(ValueError)` with an unknown species; γ(50/50 H₂/CO₂) ≈ 1.337.
- Effort: S

## Draft Pull Request

**Title:** `fix(compression): raise on unknown species, derive gamma from Cp`

### Why This Matters (The Pitch)

The compression engine logs a warning saying it will use defaults for an unknown species, then does nothing of the kind: it skips the species and leaves its fraction out of every sum. A feed of 40 percent H2, 30 percent CO and 30 percent NH3 gets its molecular weight, critical properties and gamma computed from 70 percent of the gas, with no renormalisation, so MW is low by 30 percent and compressor power is wrong by the same. Separately, gamma is averaged by mole fraction, when it should come from mixture Cp over Cp minus R, a 1 to 2 percent error for H2 and CO2 blends. The fix raises a ValueError for unknown species and uses the proper Cp-based formula. The engineers who benefit are those sizing compressors on real syngas, where trace species are normal and silent omission is dangerous.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:168-178` logs `Species %s not found in database, using defaults` and then `continue`s, so the fraction is lost, and sets `mix_gamma += frac * gamma` instead of deriving gamma from mixture Cp. `{"H2":0.4,"CO":0.3,"NH3":0.3}` is computed from 70 % of the gas.

## Change

- [ ] Raise `ValueError` on unknown species
- [ ] `cp_mix = sum(y_i*gamma_i/(gamma_i-1)*R)`; `gamma_mix = cp_mix/(cp_mix - R)`
- [ ] Add tests for unknown species and 50/50 H2/CO2

## Verification

`pytest.raises(ValueError)` with an unknown species; gamma for 50/50 H2/CO2 is approximately 1.337.

## Risk and Rollback

Callers passing species missing from the database will now raise; mixture gamma shifts by 1-2 % for some blends. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-014 concerns the species database import in the same file.

Refs: adversarial review 2026-10-16, F-TL-011

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
