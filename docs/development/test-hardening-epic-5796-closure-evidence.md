# Test hardening epic #5796 — closure evidence

Machine-readable registry: `config/test_hardening_epic_5796_registry.json`  
Guarded in CI by: `tests/ops/test_test_hardening_epic_5796_registry.py`

## Landed theme-suite children (closed on `main`)

| Finding | Child issue | Merge PR | Primary regression test(s) |
| ------- | ----------- | -------- | -------------------------- |
| TOOLS-02 | [#5797](https://github.com/D-sorganization/Tools/issues/5797) | [#5815](https://github.com/D-sorganization/Tools/pull/5815) | `test_isentropic_efficiency_le_1` |
| TOOLS-03 | [#5798](https://github.com/D-sorganization/Tools/issues/5798) | [#5816](https://github.com/D-sorganization/Tools/pull/5816) | `test_polytropic_tout_depends_on_efficiency` |
| TOOLS-05 | [#5799](https://github.com/D-sorganization/Tools/issues/5799) | [#5818](https://github.com/D-sorganization/Tools/pull/5818) | `test_water_dropout_after_aftercooler` |
| TOOLS-06 | [#5800](https://github.com/D-sorganization/Tools/issues/5800) | [#5814](https://github.com/D-sorganization/Tools/pull/5814) | `test_steam_phase_from_saturation` |
| TOOLS-10, TOOLS-11 | [#5801](https://github.com/D-sorganization/Tools/issues/5801) | [#5813](https://github.com/D-sorganization/Tools/pull/5813) | `test_wgs_volume_units`, `test_wgs_reverse_shift` |
| TOOLS-15 | [#5802](https://github.com/D-sorganization/Tools/issues/5802) | [#5812](https://github.com/D-sorganization/Tools/pull/5812) | `test_urdf_left_right_mirror` |
| TOOLS-16 | [#5803](https://github.com/D-sorganization/Tools/issues/5803) | [#5811](https://github.com/D-sorganization/Tools/pull/5811) | `test_box_inertia_analytic` |
| TOOLS-18 | [#5804](https://github.com/D-sorganization/Tools/issues/5804) | [#5808](https://github.com/D-sorganization/Tools/pull/5808) | `test_coulomb_regularised_rk45_step_count_bounded` |
| TOOLS-20 | [#5805](https://github.com/D-sorganization/Tools/issues/5805) | [#5809](https://github.com/D-sorganization/Tools/pull/5809) | `test_eckert_flow_param_uses_gas_flux` |
| GLASS-06, TOOLS-30/31, TP-04 | [#5806](https://github.com/D-sorganization/Tools/issues/5806) | [#5819](https://github.com/D-sorganization/Tools/pull/5819) | `test_glass_interface_defaults_equal_across_copies`, `test_glass_fallback_catches_runtime_error` |

## Routed to deferred verification (#5807)

Findings listed in the epic theme suites as `→ #5807` remain on the
[deferred tracker #5807](https://github.com/D-sorganization/Tools/issues/5807).
They are intentionally **not** checked off as landed here until verify-first
work completes per row.

## Test-quality lane (PD-2)

| Issue | PR | Notes |
| ----- | -- | ----- |
| [#5920](https://github.com/D-sorganization/Tools/issues/5920) | [#5935](https://github.com/D-sorganization/Tools/pull/5935) | Finite positive `pressure_drop_pa` oracle |
| [#5924](https://github.com/D-sorganization/Tools/issues/5924) | [#5942](https://github.com/D-sorganization/Tools/pull/5942) | P1AM import guard |

## Epic #5796 checklist guidance

- Check **landed** rows above against the epic body (5797–5806 suite).
- Leave **5807-routed** rows open until the tracker row closes with test output.
- Close epic #5796 when #5807 backlog for this board slice is exhausted and
  in-flight test-quality PRs merge.
