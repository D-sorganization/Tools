# Fleet Critic — Tools — 2026-10-30

## Executive Summary

This pass focused on syngas process calculators touched by the October 2026
physics board (`#5795`, shipped TOOLS-01 through TOOLS-05). Several
high-severity **model-selection** defects were remediated on `main` before
this report landed ([#5851](https://github.com/D-sorganization/Tools/issues/5851) /
[#5853](https://github.com/D-sorganization/Tools/pull/5853),
[#5799](https://github.com/D-sorganization/Tools/issues/5799)). The findings
below are **evidence and validation limits** that remain after those fixes —
gaps a hostile reviewer can still exploit when judging whether documented
claims are bounded.

1. **Medium — Kiang (1981) aqueous HCl dew point is wired, but applicability
   is undocumented and uneven across acid gases.** HCl uses Kiang only when
   moisture is present; HF and H₂S still invert pure-component Antoine for
   dew-point temperature, and declared `validity_limits` are never enforced.
   See `weaknesses.md` §1. Tracks adversarial finding
   [F-TL-003](../development/adversarial-review/2026-10-16/findings/F-TL-003-acid-gas-dewpoint-is-pure-component-condensation-hcl-at-152-.md)
   (code path fixed for HCl; evidence envelope not closed).
2. **Medium — Syngas water knockout at compression pressures still rests on
   ideal-gas saturation and a relative-humidity heuristic.** PR #5799
   corrected _where_ saturation is evaluated (cooler outlet vs superheated
   discharge), but not _how_ vapor–liquid equilibrium is modeled at the
   10–80 bar stage pressures typical of syngas trains. See `weaknesses.md` §2.
3. **Medium — Multistage compressor sizing chains constant per-species γ through
   an ideal-gas Cp/Cv mixture formula; same-stage work uses pre-knockout γ, and
   unknown species can still be dropped silently.** TOOLS-04 removed
   mole-fraction averaging of γ, but tabulated γ is not temperature- or
   pressure-dependent. See `weaknesses.md` §3. Related:
   [F-TL-010](../development/adversarial-review/2026-10-16/findings/F-TL-010-polytropic-compression-branch-is-isentropic-with-efficiency-.md),
   [F-TL-011](../development/adversarial-review/2026-10-16/findings/F-TL-011-compression-engine-mole-averages-gamma-and-silently-drops-un.md).

## What Would Change My Mind

- §1 would soften if `acid_gas_dewpoint_README.md` (or the design manual
  chapter) published an explicit Kiang validity envelope, enforced
  `validity_limits`, and either extended aqueous models to HF/H₂S or
  documented why pure-component Antoine remains conservative.
- §2 would soften with IAPWS- or CoolProp-backed knockout benchmarks at
  representative stage pressures (for example 30 bar, 40 °C cooler outlet)
  showing the RH heuristic error band.
- §3 would soften with stage-wise γ(T, P) from a mixture Cp library, a
  documented work-vs-knockout sequencing policy, and a contract test that
  unknown species raise instead of logging-and-skipping.

## Scope

Reviewed at HEAD `35ff4e210` (2026-10-30 review date):

- `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py`
  and tests under `tests/shared/python/process_calculators/test_acid_gas_dewpoint*.py`
- `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py`,
  `syngas_water_calculator.py`, and syngas compression tests
- User-facing claims in `docs/user_manual/03_process_calculators.md` and
  `acid_gas_dewpoint_README.md`
- October board context: `docs/development/adversarial-review/2026-10-16/`,
  `docs/development/DEVELOPMENT_LOG.md` entries DL-#5851 and DL-#5799

Remediation tracking: [issue #5870](https://github.com/D-sorganization/Tools/issues/5870)
(this report). Implementation work stays in the physics epic
[#5795](https://github.com/D-sorganization/Tools/issues/5795) and deferred
board tracker [#5807](https://github.com/D-sorganization/Tools/issues/5807).
