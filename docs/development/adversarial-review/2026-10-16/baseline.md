# Mechanical Baseline — Tools (2026-10-16)

Every tracked Python file was scanned. This page is evidence for the review, not a finding list; the reviewer
decided which of these matter and wrote those up as findings.

## The Repo's Own Ruff Check (as CI Would Run It)

Command: `ruff check .`

```
16	RUF100	[*] unused-noqa
 8	TRY002	[ ] raise-vanilla-class
 6	UP017 	[*] datetime-timezone-utc
 5	I001  	[*] unsorted-imports
 3	ISC004	[ ] implicit-string-concatenation-in-collection-literal
 3	TRY004	[ ] type-check-without-type-error
 2	UP042 	[ ] replace-str-enum
 1	DTZ006	[ ] call-datetime-fromtimestamp
 1	SIM102	[ ] collapsible-if
 1	N999  	[ ] invalid-module-name
 1	EXE001	[ ] shebang-not-executable
Found 47 errors.
[*] 27 fixable with the `--fix` option (5 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

## Isolated Adversarial Ruff Rules

Rules the repo does not enable; counts over `src` and `scripts`.

| count | rule                                               |
| ----- | -------------------------------------------------- |
| 16127 | `S101` [ ] assert                                  |
| 10558 | `TRY003` [ ] raise-vanilla-args                    |
| 2814  | `PLC0415` [ ] import-outside-top-level             |
| 1464  | `N806` [ ] non-lowercase-variable-in-function      |
| 1003  | `RUF100` [*] unused-noqa                           |
| 562   | `ARG001` [ ] unused-function-argument              |
| 430   | `N803` [ ] invalid-argument-name                   |
| 393   | `ARG002` [ ] unused-method-argument                |
| 382   | `PLR0917` [ ] too-many-positional-arguments        |
| 291   | `ARG005` [ ] unused-lambda-argument                |
| 290   | `C901` [ ] complex-structure                       |
| 280   | `N802` [ ] invalid-function-name                   |
| 264   | `T201` [ ] print                                   |
| 247   | `RUF001` [ ] ambiguous-unicode-character-string    |
| 243   | `TRY400` [-] error-instead-of-exception            |
| 243   | `RUF022` [-] unsorted-dunder-all                   |
| 220   | `E402` [ ] module-import-not-at-top-of-file        |
| 212   | `TRY004` [ ] type-check-without-type-error         |
| 194   | `RUF002` [ ] ambiguous-unicode-character-docstring |
| 172   | `TRY300` [ ] try-consider-else                     |
| 152   | `ERA001` [ ] commented-out-code                    |
| 139   | `PTH123` [ ] builtin-open                          |
| 126   | `BLE001` [ ] blind-except                          |
| 125   | `S603` [ ] subprocess-without-shell-equals-true    |
| 107   | `PLR0912` [ ] too-many-branches                    |

## Structure From the AST

- functions: 47,570; files > 800 lines: 77; functions > 80 lines: 407; functions > 6 params: 60
- largest files:
  - 2123 lines `src/rotation_converter/modern_robotics.py`
  - 1551 lines `src/shared/python/chat/_chat_dock_widget_qt.py`
  - 1433 lines `src/p1am_control_system/backend/main.py`
  - 1302 lines `src/rotation_converter/ui/pyqt6/main_window.py`
  - 1174 lines `src/shared/python/calc_backend/tests/test_calc_backend.py`
  - 1160 lines `src/shared/python/model_generation/api/rest_api_routes.py`
  - 1147 lines `tests/ops/test_automerge_guard.py`
  - 1133 lines `src/movement_optimizer/models/swingset.py`
  - 1130 lines `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py`
  - 1121 lines `src/data_processing/data_processor/python/data_processor/core/uncertainty_quantification.py`
  - 1111 lines `src/pendulum_simulator/src/double_pendulum_golf/gui/equations_data.py`
  - 1107 lines `src/shared/python/sidekick/calculators/thermo/steam_engine.py`
- longest functions:
  - 441 lines `scripts/generate_tools_calculations.py:26` `build_freshness_manifest`
  - 352 lines `src/shared/python/chat/_qt/ui_builder.py:106` `build_chat_dock_ui`
  - 336 lines `scripts/tools_calculation_freshness_contract.py:243` `load_calculation_freshness`
  - 280 lines `rust_core/tools-core/tests/python_ground_reference_runtime.py:33` `main`
  - 268 lines `src/shared/python/swing_sim/impact_interval/solver.py:172` `solve_impact_interval`
  - 240 lines `src/shared/python/swing_sim/variation/registry.py:137` `_register_builtins`
  - 228 lines `src/shared/python/theme/catppuccin.py:44` `get_stylesheet`
  - 228 lines `src/rate_of_closure/web_authority/api.py:112` `create_authority_app`
  - 224 lines `scripts/tools_publication_projection_contract.py:108` `load_publication_projection`
  - 219 lines `src/p1am_control_system/backend/tuning_router.py:81` `create_tuning_router`
  - 213 lines `src/p1am_control_system/backend/pid_tuning.py:320` `identify_fopdt_and_tune`
  - 203 lines `src/shared/python/sidekick/lab/mocap/reconstruction.py:266` `triangulate_n_views`
  - 198 lines `src/rate_of_closure/derivation.py:264` `derivation_steps`
  - 194 lines `src/rate_of_closure/ui/pyqt6/impact_scene_renderer.py:59` `draw_impact_scene_3d`
  - 191 lines `src/rotation_converter/modern_robotics.py:1933` `SimulateControl`

## Test-Suite Texture

Greps over `tests/` and `src/`.

- `pytest.skip` / `@pytest.mark.skip` sites: 335
- `MagicMock` / `patch(` sites: 2332
- literal `assert True` / `assert 1` lines: 25
  - `tests/shared/python/signal_toolkit/test_widget_mixins.py:118`
  - `tests/shared/python/process_calculators/test_wgs_reactor_engine.py:59`
  - `tests/shared/python/process_calculators/test_pressure_drop_engine.py:367`
  - `tests/shared/python/process_calculators/test_pressure_drop_gas_properties.py:49`
  - `tests/shared/python/process_calculators/test_pressure_drop_gas_properties.py:133`
  - `tests/shared/python/process_calculators/test_gas_properties_physics.py:139`
  - `tests/shared/python/golf_club/test_transient_vibroacoustics.py:151`
  - `tests/test_calculators_expanded.py:208`
  - `tests/test_calculators_expanded.py:732`
  - `tests/test_calculators_expanded.py:750`

## Bandit Findings at Medium and High Severity

| sev    | id   | text                                                                   | n   |
| ------ | ---- | ---------------------------------------------------------------------- | --- |
| MEDIUM | B608 | Possible SQL injection vector through string-based query construction. | 5   |
| MEDIUM | B604 | Function call with shell=True parameter identified, possible security  | 1   |

## Dangerous-Pattern Greps

Over `src` and `scripts`; many `exec(` hits are Qt `dialog.exec()` — the reviewer triaged these.

- `exec(`: 145
- `eval(`: 19
- `md5`: 8
- `pickle.load`: 2
- `shell=True`: 1

## Radon: Functions at Cyclomatic Complexity D, E or F

Count at or above 21: 63

- `F 252:0 validate_expression - D (24)`
- `C 141:0 InputHistory - F (47)`
- `M 155:4 InputHistory.__post_init__ - F (46)`
- `C 28:0 ReplayExecutionPolicy - D (26)`
- `M 49:4 ReplayExecutionPolicy.__post_init__ - D (25)`
- `F 177:0 solve - D (30)`
- `F 42:0 _validate - D (23)`
- `F 266:0 triangulate_n_views - E (40)`
- `C 35:0 DataReader - D (21)`
- `F 18:0 _walk - D (23)`
- `F 111:0 detect_gh_intent - D (28)`
- `F 209:0 import_session - D (23)`
- `F 443:0 analyze_variables - D (28)`
- `F 181:0 compute_correlations - D (24)`
- `M 745:4 ModelGenerationAPI.inertia_from_mesh - D (21)`
- `F 102:0 cmd_convert - D (21)`
- `F 122:0 _assert_check_contract - E (35)`
- `F 281:0 test_complete_rest_projection_preserves_metrics_target_and_profile - D (25)`
- `F 96:0 test_strategy_analysis_returns_scatter_cost_and_common_random_regret - D (25)`
- `M 212:4 ImpactStudyV1.__post_init__ - D (25)`
- `F 355:0 verify_study_qualification - D (21)`
- `F 172:0 solve_impact_interval - D (24)`
- `F 58:0 add_stream - D (24)`
- `F 44:0 collect_issues - D (26)`
- `M 150:4 ThemeColors.model_post_init - D (25)`

## Vulture Dead-Code Hits

At 80 % confidence or higher: 86 hits

- `src/c3d_viewer/tests/test_c3d_viewer_gui.py:36: unused variable 'mock_qt_app' (100% confidence)`
- `src/c3d_viewer/tests/test_c3d_viewer_gui.py:45: unused variable 'mock_qt_app' (100% confidence)`
- `src/c3d_viewer/tests/test_c3d_viewer_gui.py:293: unused variable 'qt_display' (100% confidence)`
- `src/data_processing/data_processor/python/data_processor/core/anova.py:200: unused variable 'f_obs' (100% confidence)`
- `src/data_processing/data_processor/python/data_processor/core/dat_importer.py:127: unused variable 'dbf_path' (100% confidence)`
- `src/data_processing/data_processor/python/data_processor/core/plot_zoom.py:308: unused variable 'enable_pan' (100% confidence)`
- `src/data_processing/data_processor/python/data_processor/core/spectral_analysis.py:88: redundant if-condition (100% confidence)`
- `src/data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py:113: unused variable 'width_attr' (100% confidence)`
- `src/flow_rate_converter/tests/test_flow_rate_converter_gui.py:182: unused variable 'mock_qt_app' (100% confidence)`
- `src/flow_rate_converter/tests/test_flow_rate_converter_gui.py:191: unused variable 'mock_qt_app' (100% confidence)`
- `src/function_generator/tests/test_function_generator_gui.py:162: unused variable 'mock_qt_app' (100% confidence)`
- `src/humanoid_builder_gui/tests/test_humanoid_builder_gui.py:35: unused variable 'mock_qt_app' (100% confidence)`
- `src/humanoid_builder_gui/tests/test_humanoid_builder_gui.py:44: unused variable 'mock_qt_app' (100% confidence)`
- `src/humanoid_builder_gui/tests/test_humanoid_builder_gui.py:290: unused variable 'mock_qt_modules' (100% confidence)`
- `src/humanoid_builder_gui/tests/test_humanoid_builder_gui.py:313: unused variable 'mock_qt_modules' (100% confidence)`
- `src/inertia_calculator/tests/test_inertia_calculator_gui.py:41: unused variable 'mock_qt_app' (100% confidence)`
- `src/inertia_calculator/tests/test_inertia_calculator_gui.py:50: unused variable 'mock_qt_app' (100% confidence)`
- `src/media_processing/video_processor/python/video_processor_src/constants.py:23: unused import 'UNIVERSAL_GAS_CONSTANT_J_MOL_K' (90% confidence)`
- `src/multi_param_analysis/tests/test_multi_param_analysis_gui.py:36: unused variable 'mock_qt_app' (100% confidence)`
- `src/multi_param_analysis/tests/test_multi_param_analysis_gui.py:45: unused variable 'mock_qt_app' (100% confidence)`
- `src/ode_solver/tests/test_ode_solver_gui.py:39: unused variable 'mock_qt_app' (100% confidence)`
- `src/ode_solver/tests/test_ode_solver_gui.py:48: unused variable 'mock_qt_app' (100% confidence)`
- `src/ode_solver/tests/test_ode_solver_gui.py:57: unused variable 'mock_qt_app' (100% confidence)`
- `src/p1am_control_system/backend/tests/test_route_authz_enforcement.py:19: unused import '_authenticated_deployment' (90% confidence)`
- `src/p1am_control_system/backend/tests/test_route_authz_matrix.py:54: unused import '_authenticated_deployment' (90% confidence)`
