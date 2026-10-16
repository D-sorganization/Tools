# F-TL-026

**477 tests assert only is-not-None/>0, 400 have no assertion, 175 mock-only; every numeric bug above passes**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | tests    | high       | L      | proposed |

**Locations:** `tests/shared/python/process_calculators/test_wgs_reactor_calculator.py:60`, `tests/shared/python/process_calculators/test_wgs_reactor_calculator.py:64`, `tests/shared/python/process_calculators/test_pressure_drop_gas_properties.py:105`, `tests/shared/python/process_calculators/test_gas_properties_physics.py:164`

## Finding

- Severity: High - Category: tests
- Confidence: high
- Location: tests/shared/python/process_calculators/test_wgs_reactor_calculator.py:54-64; tests/shared/python/process_calculators/test_pressure_drop_gas_properties.py:100-105; tests/shared/python/process_calculators/test_gas_properties_physics.py:160-164
- Evidence:

```python
assert sizing["reactor_volume"] > 0      # 22.4x error passes
assert sizing["heat_duty"] > 0            # 4x error passes
mu = calculate_pure_gas_viscosity_lucas(temperature=600.0, pressure=5e5, props=props)
assert mu > 0                             # 30x error passes
rho = calculate_ideal_gas_density(28.97, 293.15, 101325.0)
assert_allclose(rho, 1.225, rtol=0.02)   # 1.225 is the 15 C value; at 20 C it is 1.204 - passes by 0.3 %
```

- Failure scenario: every numeric defect in F-TL-001..011 is covered by a passing test (21,328 tests scanned by AST). The CI "Changed Test Assertion Check" requires _an_ assert, so `assert result is not None` is the path of least resistance for an agent told to "add tests".
- Why a newer engineer misses this: coverage and test count both rise; green stays green.
- Proposed change: one hand-computed reference case per `process_calculators` entry point with units in the test name (`test_wgs_volume_100kmolh_is_0_747_m3`); forbid `> 0`-only assertions in `scripts/check_test_assertions.py` for `process_calculators`, `swing_sim/flight`, `signal_toolkit`; Hypothesis property tests for monotonicity/limits.
- Verification: `mutmut run --paths-to-mutate src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py` — today most mutants survive.
- Effort: L

## Draft Pull Request

**Title:** `test(calculators): replace > 0 asserts with hand-computed references`

### Why This Matters (The Pitch)

Every numeric defect in this review is covered by a passing test. 477 tests assert only is not None, greater than zero or True, 400 assert nothing, and 175 assert only that a mock was called, across 21,328 tests scanned. The WGS sizing test asserts reactor volume is greater than zero, so a 22.4 times error passes; the Lucas viscosity test asserts mu is greater than zero, so a 30 times error passes; an air-density test passes by 0.3 percent using the 15 C value at 20 C. The CI assertion check only demands that some assert exists, so asserting not None is the path of least resistance for an agent told to add tests. One hand-computed case per process_calculators entry point, named with units, plus a rule forbidding greater-than-zero-only checks in the numeric packages, turns coverage into evidence. Everyone who relies on these calculators benefits.

### Proposed PR Body

## Problem

`tests/shared/python/process_calculators/test_wgs_reactor_calculator.py:54-64` asserts `sizing["reactor_volume"] > 0` and `sizing["heat_duty"] > 0`; `test_pressure_drop_gas_properties.py:100-105` asserts `mu > 0`; `test_gas_properties_physics.py:160-164` compares density to 1.225 (the 15 C value) at 20 C with `rtol=0.02`.

## Change

- [ ] Add one hand-computed reference test per `process_calculators` entry point, with units in the test name (e.g. `test_wgs_volume_100kmolh_is_0_747_m3`)
- [ ] Forbid `> 0`-only assertions in `scripts/check_test_assertions.py` for `process_calculators`, `swing_sim/flight`, `signal_toolkit`
- [ ] Add Hypothesis property tests for monotonicity and limits

## Verification

`mutmut run --paths-to-mutate src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py` shows most mutants killed; today most survive.

## Risk and Rollback

New strict tests will fail until F-TL-001..011 land; sequence this after, or mark the known-defect tests as xfail with the finding id. Rollback is a revert.

## Handoff

Follow-ups: depends on F-TL-001 through F-TL-011 for the tests to go green.

Refs: adversarial review 2026-10-16, F-TL-026

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
