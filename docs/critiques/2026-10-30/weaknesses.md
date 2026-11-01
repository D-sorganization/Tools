# Weaknesses — Tools — 2026-10-30

## Weakness: Kiang aqueous HCl dew point — partial adoption and unenforced validity envelope

### Summary of Concern

The October board shipped the correct _model class_ for aqueous HCl (Kiang
1981) when moisture is present ([#5853](https://github.com/D-sorganization/Tools/pull/5853)).
What remains for evidence is narrower but still load-bearing: the calculator
does not tell the user when Kiang applies, when it does not, or what happens
at the edge of the correlation; HF and H₂S dew points in wet syngas are still
pure-component Antoine inversions, which was the original failure mode for HCl.

### Location

- **File/Document:** `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py`;
  `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_README.md`
- **Section:** `calculate_hcl_aqueous_dewpoint`; `calculate_dewpoint` branch
  for `component == "HCl" and h2o_partial_pressure_pa > 0`; `validity_limits`
- **Claim or Equation:** Kiang (1981) correlation for aqueous HCl; README
  still describes multi-component acid dew points without distinguishing
  aqueous vs pure-component paths per acid.

### Nature of the Issue

- [ ] Logical gap
- [x] Unstated assumption (Kiang fitted data range applies wherever the code
      returns a finite number)
- [ ] Overgeneralization
- [x] Empirical insufficiency (`validity_limits` for HCl are declared at
      `T_min`/`T_max`/`P_max` but never checked; no multi-point golden vectors
      beyond unit tests)
- [ ] Terminological ambiguity
- [x] Literature conflict (Verhoff–Banchero and later aqueous-acid work cover
      HF/H₂S; tool still uses Antoine for those acids in mixtures)

### Why This Is a Problem

A reviewer who reads only the README sees "multi-component acid gas dewpoint"
and reasonable Antoine citations. They must diff into `calculate_dewpoint` to
learn that only HCl switches to Kiang, and that H₂S in a wet stream can still
report a pure-component dew point near −140 °C while water limits corrosion
earlier. The declared HCl limits (`T_min = -85 °C`, `T_max = 51 °C`) are never
applied to Kiang output, so out-of-range inputs return silent finite numbers
instead of `nan`/warnings — the same class of gap F-TL-003 flagged before the
HCl path was fixed.

### Evidence / References

- `grep validity_limits acid_gas_dewpoint_calculator.py` — assigned once, never read.
- `calculate_dewpoint` — Kiang branch only when `component == "HCl"`.
- Tests: `tests/shared/python/process_calculators/test_acid_gas_dewpoint_edge_cases.py`
  (`TestKiangAqueousHClDewpoint`) anchor one HCl/H₂O point; no HF/H₂S aqueous
  coverage.
- Adversarial review:
  [F-TL-003](../development/adversarial-review/2026-10-16/findings/F-TL-003-acid-gas-dewpoint-is-pure-component-condensation-hcl-at-152-.md).

### Severity

- [ ] Low
- [x] Medium (model fix shipped; publication and enforcement of limits still
      open)
- [ ] High

### Suggested Remedies

- Document per-acid model choice in README and user manual (aqueous vs Antoine).
- Enforce `validity_limits` on returned dew points; surface warnings in
  `DewpointResult.warnings`.
- Add golden fixtures from Kiang (1981) table or Verhoff–Banchero for at least
  two HCl/H₂O points; track HF/H₂S aqueous work under `#5795` / `#5807`.

---

## Weakness: High-pressure syngas water knockout — ideal-gas saturation without validation band

### Summary of Concern

Multistage compression evaluates liquid water dropout via
`calculate_water_dropout`, which compares bulk water mole percent to an
ideal-gas saturation mole fraction derived from Antoine vapor pressure over
total pressure. Board fix TOOLS-05 ([#5799](https://github.com/D-sorganization/Tools/issues/5799))
correctly evaluates that saturation at the **cooler outlet temperature** when
intercooling is enabled, but the thermodynamic model is unchanged: it is still
a relative-humidity-style cutoff with no fugacity, dissolved gas, or
high-pressure steam-table anchor.

### Location

- **File/Document:** `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py`;
  `syngas_water_calculator.py`
- **Section:** `calculate_water_dropout` (lines ~186–224);
  `calculate_multistage_compression` saturation temperature selection (~376–387)
- **Claim or Equation:** `relative_humidity = (water_content/100)*pressure /
  water_vp_bar`; dropout when RH > 1

### Nature of the Issue

- [ ] Logical gap
- [x] Unstated assumption (water vapor behaves as ideal gas at stage outlet
      pressures, typically tens of bar)
- [ ] Overgeneralization
- [x] Empirical insufficiency (no regression tying dropout mass to IAPWS/CoolProp
      at representative P/T; user manual advertises "comprehensive" compression
      analysis without stating the knockout simplification)
- [ ] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

Intercooler knockout drums are sized on condensed water flow. At 30–40 bar,
Antoine-derived `P_sat` divided into total pressure understates non-ideal
suppression of water activity in syngas; the error direction depends on T but
is not bounded anywhere in-repo. A reviewer comparing against plant mass
balances or steam tables has no cited tolerance — only the October temperature
placement fix, which does not answer "how wrong is the amount?"

### Evidence / References

- `syngas_compression_engine.py:208–216` — RH heuristic, no real-gas factor.
- `docs/user_manual/03_process_calculators.md` — syngas compression section
  lists water dropout without model limits.
- DEVELOPMENT_LOG DL-#5799 — documents cooler-temperature fix, not VLE validation.
- Downstream coupling noted in
  [F-TL-010](../development/adversarial-review/2026-10-16/findings/F-TL-010-polytropic-compression-branch-is-isentropic-with-efficiency-.md)
  (polytropic T fix shipped; knockout model unchanged).

### Severity

- [ ] Low
- [x] Medium
- [ ] High

### Suggested Remedies

- Publish an explicit "knockout model" paragraph in the user manual (ideal-gas
  RH, valid pressure band, expected error vs IAPWS at one benchmark point).
- Add a `@pytest.mark.scientific` test comparing dropout at 1 bar vs 30 bar
  against CoolProp when available, skipped with reason when not.
- Track high-P VLE upgrade under `#5795` / `#5807`.

---

## Weakness: Constant-γ ideal-gas compression chain — frozen mixture properties and silent species loss

### Summary of Concern

`SyngasCompressionEngine` derives mixture γ from tabulated per-species
`_APPROX_GAMMA` and ideal-gas relations `Cp_i = γ_i/(γ_i−1)·R`. TOOLS-04
removed mole-fraction averaging of γ, but each species still carries a fixed
γ independent of stage temperature and pressure. After water knockout the engine
can refresh `mixture_props` for later stages, yet the **same** stage's
compression work is computed before knockout renormalization, so polytropic
head on a stage with condensate still uses the wet-blend γ. Unknown species
still log a warning and `continue`, omitting their mole fraction without
renormalizing (F-TL-011).

### Location

- **File/Document:** `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py`
- **Section:** `_APPROX_GAMMA`; `calculate_mixture_properties`; loop in
  `calculate_multistage_compression` (~356–410)
- **Claim or Equation:** Constant γ per species; `mix_gamma = total_cp/total_cv`;
  polytropic exponent `(γ−1)/(γ·η_p)`

### Nature of the Issue

- [ ] Logical gap
- [x] Unstated assumption (γ independent of temperature and pressure along the
      train; same-stage work uses pre-knockout γ even when knockout removes H₂O
      at that stage outlet)
- [x] Overgeneralization (user-facing "comprehensive" compression analysis
      implies stage-resolved thermodynamics)
- [x] Empirical insufficiency (no test asserting γ refresh after knockout; no
      benchmark vs variable-Cp mixture at discharge T)
- [ ] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

Board item TOOLS-04 fixed mole-fraction **averaging** of γ, which was a real
bug, but the remaining chain is still a constant-γ ideal-gas compressor model
— acceptable for screening if labeled, risky if read as validated multistage
performance. Using the wet-blend γ for polytropic work on a stage where water
condenses at the outlet can mis-state discharge temperature and power for that
stage. Silent species omission can still skew MW and power by tens of percent
for feeds containing ammonia or other trace components not in the database.

### Evidence / References

- `syngas_compression_engine.py:124–133` — `_APPROX_GAMMA` table.
- `syngas_compression_engine.py:156–160` — unknown species `continue`.
- `calculate_multistage_compression` — `mixture_props` refreshed after
  knockout when `dropout_pct > 0` (~409–410), but stage work runs first with
  the pre-knockout blend; no refresh when dropout is zero.
- Polytropic branch now uses `(γ−1)/(γ·η)` (TOOLS-04 / F-TL-010 fix); γ
  source limitations remain.
- [F-TL-011](../development/adversarial-review/2026-10-16/findings/F-TL-011-compression-engine-mole-averages-gamma-and-silently-drops-un.md).

### Severity

- [ ] Low
- [x] Medium
- [ ] High

### Suggested Remedies

- Recompute `mixture_props` before stage work when prior-stage knockout changed
  composition, or document that work and knockout are sequentially decoupled.
- Raise `ValueError` on unknown species (stop silent `continue`).
- Document constant-γ screening limits in user manual; add stage-wise γ(T)
  work to `#5795`.
- Contract test: multistage run with H₂O knockout changes reported γ or power
  on stage 2 vs a frozen-γ baseline.
