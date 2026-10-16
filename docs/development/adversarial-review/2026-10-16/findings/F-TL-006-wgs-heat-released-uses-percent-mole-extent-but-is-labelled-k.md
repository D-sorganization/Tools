# F-TL-006

**WGS heat_released uses percent-mole extent but is labelled kJ/mol**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:335`, `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:684`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:335, :684
- Evidence: `heat_released = x_eq * WGS_HEAT_KJ_PER_MOL`; GUI `f"  Heat Released: {equilibrium['heat_released']:.2f} kJ/mol\n\n"`.
- Failure scenario (run): default GUI inputs (25/20/10/5 %) → `heat_released = 892.25 "kJ/mol"`; the WGS exotherm is 41.2 kJ/mol, so the number is kJ per 100 mol of feed. Any heat balance that trusts the label is off by 100×.
- Why a newer engineer misses this: `_prepare_initial_moles` accepts whatever numbers it is given; the GUI feeds percents, tests feed fractions, and both "work".
- Proposed change: normalise `inlet_composition` to fractions in `_prepare_initial_moles` and name the field `heat_released_kj_per_mol_feed`.
- Verification: assert `heat_released == pytest.approx(0.8663 * 0.25 * 41.2)` for the default inputs.
- Effort: S

## Draft Pull Request

**Title:** `fix(wgs): normalise composition so heat_released is kJ per mol feed`

### Why This Matters (The Pitch)

The WGS calculator prints a heat release of 892.25 kJ/mol for its default inputs. The reaction exotherm is 41.2 kJ/mol; the printed figure is kJ per 100 mol of feed because the GUI hands the function percentages while the tests hand it fractions, and the code accepts both without complaint. Any heat balance that trusts the label is off by a factor of 100. The change normalises composition to fractions in one helper and renames the field so the unit is in the name, which closes the percent-versus-fraction ambiguity for good rather than patching the display. Engineers building energy balances around the shift reactor benefit, and the check is a single reference value for the default inputs: 0.8663 times 0.25 times 41.2.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:335, :684` computes `heat_released = x_eq * WGS_HEAT_KJ_PER_MOL` from an extent in percent-moles, and the GUI labels it kJ/mol. Default GUI inputs (25/20/10/5 %) show 892.25 kJ/mol.

## Change

- [ ] Normalise `inlet_composition` to fractions in `_prepare_initial_moles`
- [ ] Rename the field `heat_released_kj_per_mol_feed`
- [ ] Update the GUI label and tests

## Verification

`assert heat_released == pytest.approx(0.8663 * 0.25 * 41.2)` for the default inputs.

## Risk and Rollback

The renamed field is an API change for any consumer reading `heat_released`. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-001 and F-TL-005 are in the same file.

Refs: adversarial review 2026-10-16, F-TL-006

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
