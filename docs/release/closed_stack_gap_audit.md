# Closed-stack gap audit (Tools #4921)

Generated: 2026-09-03T01:14:15Z  
Base: `origin/main` @ `7bc0f652c254d5b951204f50531f4723a2d2097e`  
Repo: `D-sorganization/Tools`

## Method

For each PR the head branch (or recorded head SHA) is diffed against `origin/main` with the three-dot form. Files added relative to the merge-base **and** absent from `origin/main` are listed and classified:

- `landed-elsewhere`: >=50% of the file's public top-level symbols (max 8, `_private` and generic names skipped) exist on main under `src/`, `tests/` or `rust_core/`.
- `obsolete`: drafts, `.gaai/`, assessments, agent worktrees/scratch (`.codex-worktrees/`, `_codex_*`, `_wt_claude_*`), or file names with the words `codex`/`jules`/`plan`/`pr_details`.
- `missing`: everything else.

Groups take the majority class of their files. Evidence JSON: `docs/release/closed_stack_gap_audit.v1.json`.

**Unreachable refs:** none - every PR head was diffable.

## #4169 - Add Screw-Axis Analytics and Motion Glyphs

- State: `MERGED` | head `feat/screw-axis-analysis` (`391c505278eb`) | base `feat/4144-variation-visualizations` (on origin: True) | merged 2026-08-06T02:31:40Z | closed 2026-08-06T02:31:40Z
- URL: https://github.com/D-sorganization/Tools/pull/4169
- Diff ref: `origin/feat/screw-axis-analysis` | diffstat: 738 files changed, 84399 insertions(+), 2434 deletions(-) | added 428, modified 310, deleted 0
- Recommendation: drop (nothing missing; landed elsewhere or obsolete)

No `missing` files.

## #4173 - feat(rate-of-closure): add shared impact inspector

- State: `MERGED` | head `feat/4163-impact-inspector` (`137c6587d4f7`) | base `feat/4144-variation-visualizations` (on origin: True) | merged 2026-08-14T00:02:02Z | closed 2026-08-14T00:02:02Z
- URL: https://github.com/D-sorganization/Tools/pull/4173
- Diff ref: `origin/feat/4163-impact-inspector` | diffstat: 812 files changed, 93904 insertions(+), 2574 deletions(-) | added 500, modified 312, deleted 0
- Recommendation: drop (only non-product groups are missing)

| group                        | class    | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | -------- | ----: | ------------------------------------- |
| `.codex-worktrees`           | obsolete |     6 | 0 / 0 / 6                             |
| `src/movement_optimizer/gui` | missing  |     1 | 0 / 1 / 0                             |

Missing files (1):

- `src/movement_optimizer/gui/motion_helpers.py`

## #4174 - feat: add swept wedge ground-clearance analysis

- State: `MERGED` | head `feat/4161-wedge-ground-clearance` (`3e1b44cf42f4`) | base `feat/4163-impact-inspector` (on origin: True) | merged 2026-08-13T23:37:44Z | closed 2026-08-13T23:37:44Z
- URL: https://github.com/D-sorganization/Tools/pull/4174
- Diff ref: `origin/feat/4161-wedge-ground-clearance` | diffstat: 812 files changed, 93904 insertions(+), 2574 deletions(-) | added 500, modified 312, deleted 0
- Recommendation: drop (only non-product groups are missing)

| group                        | class    | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | -------- | ----: | ------------------------------------- |
| `.codex-worktrees`           | obsolete |     6 | 0 / 0 / 6                             |
| `src/movement_optimizer/gui` | missing  |     1 | 0 / 1 / 0                             |

Missing files (1):

- `src/movement_optimizer/gui/motion_helpers.py`

## #4209 - feat(rate-of-closure): add launch direction conventions

- State: `MERGED` | head `feat/4193-launch-direction-registry-integration` (`98589174273e`) | base `feat/4181-launch-monitor-registry` (on origin: True) | merged 2026-08-07T03:07:08Z | closed 2026-08-07T03:07:08Z
- URL: https://github.com/D-sorganization/Tools/pull/4209
- Diff ref: `origin/feat/4193-launch-direction-registry-integration` | diffstat: 865 files changed, 103125 insertions(+), 2559 deletions(-) | added 551, modified 314, deleted 0
- Recommendation: drop (only non-product groups are missing)

| group                        | class   | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | ------- | ----: | ------------------------------------- |
| `src/movement_optimizer/gui` | missing |     1 | 0 / 1 / 0                             |

Missing files (1):

- `src/movement_optimizer/gui/motion_helpers.py`

## #4212 - feat(rate-of-closure): add launch monitor analytics tabs

- State: `MERGED` | head `feat/4205-launch-monitor-analytics` (`a4dcddde6122`) | base `feat/4181-launch-monitor-registry` (on origin: True) | merged 2026-08-07T02:33:59Z | closed 2026-08-07T02:33:59Z
- URL: https://github.com/D-sorganization/Tools/pull/4212
- Diff ref: `origin/feat/4205-launch-monitor-analytics` (branch tip `2ec1decaf2c9` differs from PR head SHA) | diffstat: 881 files changed, 105548 insertions(+), 2560 deletions(-) | added 567, modified 314, deleted 0
- Recommendation: keep for review: missing product/test groups src/rate_of_closure, src/rate_of_closure/ui, src/rate_of_closure/web, tests/rate_of_closure

| group                        | class   | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | ------- | ----: | ------------------------------------- |
| `docs/rate_of_closure`       | missing |     1 | 0 / 1 / 0                             |
| `src/movement_optimizer/gui` | missing |     1 | 0 / 1 / 0                             |
| `src/rate_of_closure`        | missing |     2 | 0 / 2 / 0                             |
| `src/rate_of_closure/ui`     | missing |     5 | 1 / 4 / 0                             |
| `src/rate_of_closure/web`    | missing |    10 | 1 / 9 / 0                             |
| `tests/rate_of_closure`      | missing |     1 | 0 / 1 / 0                             |

Missing files (18):

- `docs/rate_of_closure/LAUNCH_MONITOR_PLAYER_ANALYTICS.md`
- `src/movement_optimizer/gui/motion_helpers.py`
- `src/rate_of_closure/launch_monitor_data.py`
- `src/rate_of_closure/launch_monitor_player_metrics.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_analysis_mixin.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_player_controls.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_plot_widget.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_table_preview.py`
- `src/rate_of_closure/web/src/components/LaunchMonitorCharts.test.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorCharts.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorImportedResults.test.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorImportedResults.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorPlayerInsights.tsx`
- `src/rate_of_closure/web/src/model/launchMonitorImportedResults.test.ts`
- `src/rate_of_closure/web/src/model/launchMonitorImportedResults.ts`
- `src/rate_of_closure/web/src/model/launchMonitorPlayerAnalytics.test.ts`
- `src/rate_of_closure/web/src/model/launchMonitorPlayerAnalytics.ts`
- `tests/rate_of_closure/test_launch_monitor_player_metrics.py`

## #4217 - feat: integrate ball-flight and launch-monitor campaign

- State: `MERGED` | head `codex/ballflight-campaign-integration` (`655fea08f62b`) | base `feat/4181-launch-monitor-registry` (on origin: True) | merged 2026-08-07T16:53:54Z | closed 2026-08-07T16:53:54Z
- URL: https://github.com/D-sorganization/Tools/pull/4217
- Diff ref: `origin/codex/ballflight-campaign-integration` | diffstat: 992 files changed, 126599 insertions(+), 2559 deletions(-) | added 678, modified 314, deleted 0
- Recommendation: drop (only non-product groups are missing)

| group                        | class   | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | ------- | ----: | ------------------------------------- |
| `src/movement_optimizer/gui` | missing |     1 | 0 / 1 / 0                             |

Missing files (1):

- `src/movement_optimizer/gui/motion_helpers.py`

## #4233 - feat(rate-of-closure): add launch monitor player analytics platform

- State: `MERGED` | head `feat/4226-launch-monitor-player-platform` (`fd0ad5c7deaa`) | base `feat/4205-launch-monitor-analytics` (on origin: True) | merged 2026-08-07T04:40:23Z | closed 2026-08-07T04:40:23Z
- URL: https://github.com/D-sorganization/Tools/pull/4233
- Diff ref: `origin/feat/4226-launch-monitor-player-platform` (branch tip `d4ebc8a1872e` differs from PR head SHA) | diffstat: 913 files changed, 110484 insertions(+), 2560 deletions(-) | added 599, modified 314, deleted 0
- Recommendation: keep for review: missing product/test groups src/rate_of_closure, src/rate_of_closure/ui, src/rate_of_closure/web, tests/rate_of_closure

| group                        | class   | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | ------- | ----: | ------------------------------------- |
| `docs/rate_of_closure`       | missing |     2 | 0 / 2 / 0                             |
| `src/movement_optimizer/gui` | missing |     1 | 0 / 1 / 0                             |
| `src/rate_of_closure`        | missing |     4 | 0 / 4 / 0                             |
| `src/rate_of_closure/ui`     | missing |    11 | 1 / 10 / 0                            |
| `src/rate_of_closure/web`    | missing |    15 | 1 / 14 / 0                            |
| `tests/rate_of_closure`      | missing |     3 | 0 / 3 / 0                             |

Missing files (34):

- `docs/rate_of_closure/LAUNCH_MONITOR_PLAYER_ANALYTICS.md`
- `docs/rate_of_closure/NEURAL_MODEL_LAB.md`
- `src/movement_optimizer/gui/motion_helpers.py`
- `src/rate_of_closure/launch_monitor_data.py`
- `src/rate_of_closure/launch_monitor_player_metrics.py`
- `src/rate_of_closure/neural_model.py`
- `src/rate_of_closure/neural_training.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_analysis_mixin.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_covariation_presenter.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_covariation_scan_plot.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_covariation_state.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_data_mixin.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_player_controls.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_plot_widget.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_table_preview.py`
- `src/rate_of_closure/ui/pyqt6/neural_model_outputs.py`
- `src/rate_of_closure/ui/pyqt6/neural_training_controls.py`
- `src/rate_of_closure/web/src/components/LaunchMonitorCharts.test.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorCharts.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorImportedResults.test.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorImportedResults.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorPlayerInsights.tsx`
- `src/rate_of_closure/web/src/components/NeuralModelCharts.tsx`
- `src/rate_of_closure/web/src/model/launchMonitorImportedResults.test.ts`
- `src/rate_of_closure/web/src/model/launchMonitorImportedResults.ts`
- `src/rate_of_closure/web/src/model/launchMonitorPlayerAnalytics.test.ts`
- `src/rate_of_closure/web/src/model/launchMonitorPlayerAnalytics.ts`
- `src/rate_of_closure/web/src/model/neuralModelBundle.test.ts`
- `src/rate_of_closure/web/src/model/neuralModelBundle.ts`
- `src/rate_of_closure/web/src/model/neuralTrainingRequest.test.ts`
- `src/rate_of_closure/web/src/model/neuralTrainingRequest.ts`
- `tests/rate_of_closure/test_launch_monitor_player_metrics.py`
- `tests/rate_of_closure/test_neural_model_contract.py`
- `tests/rate_of_closure/test_neural_model_lab_tab.py`

## #4246 - feat(rate-of-closure): add Neural Model Lab

- State: `MERGED` | head `feat/4240-neural-model-lab` (`5874fbb0bfc0`) | base `feat/4226-launch-monitor-player-platform` (on origin: True) | merged 2026-08-07T08:46:02Z | closed 2026-08-07T08:46:02Z
- URL: https://github.com/D-sorganization/Tools/pull/4246
- Diff ref: `origin/feat/4240-neural-model-lab` | diffstat: 896 files changed, 108082 insertions(+), 2560 deletions(-) | added 582, modified 314, deleted 0
- Recommendation: keep for review: missing product/test groups src/rate_of_closure, src/rate_of_closure/ui, src/rate_of_closure/web, tests/rate_of_closure

| group                        | class   | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | ------- | ----: | ------------------------------------- |
| `docs/rate_of_closure`       | missing |     2 | 0 / 2 / 0                             |
| `src/movement_optimizer/gui` | missing |     1 | 0 / 1 / 0                             |
| `src/rate_of_closure`        | missing |     4 | 0 / 4 / 0                             |
| `src/rate_of_closure/ui`     | missing |     7 | 1 / 6 / 0                             |
| `src/rate_of_closure/web`    | missing |    15 | 1 / 14 / 0                            |
| `tests/rate_of_closure`      | missing |     3 | 0 / 3 / 0                             |

Missing files (30):

- `docs/rate_of_closure/LAUNCH_MONITOR_PLAYER_ANALYTICS.md`
- `docs/rate_of_closure/NEURAL_MODEL_LAB.md`
- `src/movement_optimizer/gui/motion_helpers.py`
- `src/rate_of_closure/launch_monitor_data.py`
- `src/rate_of_closure/launch_monitor_player_metrics.py`
- `src/rate_of_closure/neural_model.py`
- `src/rate_of_closure/neural_training.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_analysis_mixin.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_player_controls.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_plot_widget.py`
- `src/rate_of_closure/ui/pyqt6/launch_monitor_table_preview.py`
- `src/rate_of_closure/ui/pyqt6/neural_model_outputs.py`
- `src/rate_of_closure/ui/pyqt6/neural_training_controls.py`
- `src/rate_of_closure/web/src/components/LaunchMonitorCharts.test.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorCharts.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorImportedResults.test.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorImportedResults.tsx`
- `src/rate_of_closure/web/src/components/LaunchMonitorPlayerInsights.tsx`
- `src/rate_of_closure/web/src/components/NeuralModelCharts.tsx`
- `src/rate_of_closure/web/src/model/launchMonitorImportedResults.test.ts`
- `src/rate_of_closure/web/src/model/launchMonitorImportedResults.ts`
- `src/rate_of_closure/web/src/model/launchMonitorPlayerAnalytics.test.ts`
- `src/rate_of_closure/web/src/model/launchMonitorPlayerAnalytics.ts`
- `src/rate_of_closure/web/src/model/neuralModelBundle.test.ts`
- `src/rate_of_closure/web/src/model/neuralModelBundle.ts`
- `src/rate_of_closure/web/src/model/neuralTrainingRequest.test.ts`
- `src/rate_of_closure/web/src/model/neuralTrainingRequest.ts`
- `tests/rate_of_closure/test_launch_monitor_player_metrics.py`
- `tests/rate_of_closure/test_neural_model_contract.py`
- `tests/rate_of_closure/test_neural_model_lab_tab.py`

## #4436 - feat(rate-of-closure): add Sasho rotational AoA option

- State: `MERGED` | head `agent/sasho-face-center-rotational-aoa` (`87719a2663de`) | base `codex/4142-localized-react-execution` (on origin: True) | merged 2026-08-13T17:02:35Z | closed 2026-08-13T17:02:35Z
- URL: https://github.com/D-sorganization/Tools/pull/4436
- Diff ref: `origin/agent/sasho-face-center-rotational-aoa` | diffstat: 1202 files changed, 167955 insertions(+), 2582 deletions(-) | added 880, modified 322, deleted 0
- Recommendation: keep for review: missing product/test groups src/rate_of_closure/web

| group                        | class    | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | -------- | ----: | ------------------------------------- |
| `.codex-worktrees`           | obsolete |     6 | 0 / 0 / 6                             |
| `src/movement_optimizer/gui` | missing  |     1 | 0 / 1 / 0                             |
| `src/rate_of_closure/web`    | missing  |     1 | 0 / 1 / 0                             |

Missing files (2):

- `src/movement_optimizer/gui/motion_helpers.py`
- `src/rate_of_closure/web/src/model/__fixtures__/sasho_face_center_rotation_golden_v1.json`

## #4449 - CONS-A3: P1AM plant historian + professional SCADA foundation (supersedes #4065, #4091)

- State: `CLOSED` | head `consolidated/p1am-platform-2026-08-13` (`7fba01f5c561`) | base `main` (on origin: True) | merged None | closed 2026-08-20T09:01:27Z
- URL: https://github.com/D-sorganization/Tools/pull/4449
- Diff ref: `origin/consolidated/p1am-platform-2026-08-13` | diffstat: 140 files changed, 16678 insertions(+), 226 deletions(-) | added 105, modified 35, deleted 0
- Recommendation: drop (only non-product groups are missing)

| group                              | class   | files | landed-elsewhere / missing / obsolete |
| ---------------------------------- | ------- | ----: | ------------------------------------- |
| `dcs_scada.db`                     | missing |     1 | 0 / 1 / 0                             |
| `docs/adr`                         | missing |     1 | 0 / 1 / 0                             |
| `src/p1am_control_system/backend`  | missing |    84 | 0 / 84 / 0                            |
| `src/p1am_control_system/deploy`   | missing |     9 | 0 / 9 / 0                             |
| `src/p1am_control_system/frontend` | missing |    10 | 0 / 10 / 0                            |

Missing files (105):

- `dcs_scada.db`
- `docs/adr/ADR-007-plant-historian-timescaledb.md`
- `src/p1am_control_system/backend/advisory_router.py`
- `src/p1am_control_system/backend/advisory_workspace.py`
- `src/p1am_control_system/backend/alarm_lifecycle.py`
- `src/p1am_control_system/backend/alarm_router.py`
- `src/p1am_control_system/backend/alarm_service.py`
- `src/p1am_control_system/backend/asset_health.py`
- `src/p1am_control_system/backend/audit_log.py`
- `src/p1am_control_system/backend/audit_middleware.py`
- `src/p1am_control_system/backend/audit_router.py`
- `src/p1am_control_system/backend/availability.py`
- `src/p1am_control_system/backend/configuration_repository.py`
- `src/p1am_control_system/backend/configuration_router.py`
- `src/p1am_control_system/backend/configuration_workflow.py`
- `src/p1am_control_system/backend/connector_plugins.py`
- `src/p1am_control_system/backend/enum_compat.py`
- `src/p1am_control_system/backend/evidence_package.py`
- `src/p1am_control_system/backend/historian_shipper.py`
- `src/p1am_control_system/backend/historian_sink.py`
- `src/p1am_control_system/backend/historian_wiring.py`
- `src/p1am_control_system/backend/identity.py`
- `src/p1am_control_system/backend/identity_config.py`
- `src/p1am_control_system/backend/identity_router.py`
- `src/p1am_control_system/backend/notification_policy.py`
- `src/p1am_control_system/backend/operations_router.py`
- `src/p1am_control_system/backend/operator_router.py`
- `src/p1am_control_system/backend/process_overview.py`
- `src/p1am_control_system/backend/product_router.py`
- `src/p1am_control_system/backend/protection_management.py`
- `src/p1am_control_system/backend/recovery_package.py`
- `src/p1am_control_system/backend/representative_product.py`
- `src/p1am_control_system/backend/saved_investigation.py`
- `src/p1am_control_system/backend/scenario_evidence.py`
- `src/p1am_control_system/backend/scenario_router.py`
- `src/p1am_control_system/backend/shift_log.py`
- `src/p1am_control_system/backend/shift_log_repository.py`
- `src/p1am_control_system/backend/signal_quality.py`
- `src/p1am_control_system/backend/synthetic_procedure.py`
- `src/p1am_control_system/backend/system_health.py`
- `src/p1am_control_system/backend/system_router.py`
- `src/p1am_control_system/backend/tests/_route_inventory.py`
- `src/p1am_control_system/backend/tests/test_advisory_router.py`
- `src/p1am_control_system/backend/tests/test_advisory_workspace.py`
- `src/p1am_control_system/backend/tests/test_alarm_lifecycle.py`
- `src/p1am_control_system/backend/tests/test_alarm_router.py`
- `src/p1am_control_system/backend/tests/test_alarm_service.py`
- `src/p1am_control_system/backend/tests/test_asset_health.py`
- `src/p1am_control_system/backend/tests/test_audit_log.py`
- `src/p1am_control_system/backend/tests/test_audit_middleware.py`
- `src/p1am_control_system/backend/tests/test_audit_router.py`
- `src/p1am_control_system/backend/tests/test_availability.py`
- `src/p1am_control_system/backend/tests/test_configuration_repository.py`
- `src/p1am_control_system/backend/tests/test_configuration_router.py`
- `src/p1am_control_system/backend/tests/test_configuration_workflow.py`
- `src/p1am_control_system/backend/tests/test_connector_plugins.py`
- `src/p1am_control_system/backend/tests/test_historian_shipper.py`
- `src/p1am_control_system/backend/tests/test_historian_sink.py`
- `src/p1am_control_system/backend/tests/test_historian_wiring.py`
- `src/p1am_control_system/backend/tests/test_identity.py`
- ... and 45 more

## #4466 - Rate of Closure remainder: club builder, impact tensor, flight, multi-view workspace, ground and web companion (consolidates 43 PRs)

- State: `CLOSED` | head `consolidated/rate-closure-remainder-2026-08-13` (`9dd48d4bd27d`) | base `main` (on origin: True) | merged None | closed 2026-08-20T09:01:30Z
- URL: https://github.com/D-sorganization/Tools/pull/4466
- Diff ref: `origin/consolidated/rate-closure-remainder-2026-08-13` | diffstat: 1385 files changed, 258896 insertions(+), 451 deletions(-) | added 1344, modified 41, deleted 0
- Recommendation: keep for review: missing product/test groups src/rate_of_closure, src/rate_of_closure/web, tests/ops, tests/rate_of_closure, tests/shared

| group                        | class   | files | landed-elsewhere / missing / obsolete |
| ---------------------------- | ------- | ----: | ------------------------------------- |
| `.github/workflows`          | missing |     1 | 0 / 1 / 0                             |
| `scripts`                    | missing |     4 | 0 / 4 / 0                             |
| `src/movement_optimizer/gui` | missing |     1 | 0 / 1 / 0                             |
| `src/rate_of_closure`        | missing |     5 | 1 / 4 / 0                             |
| `src/rate_of_closure/web`    | missing |    20 | 0 / 20 / 0                            |
| `tests/ops`                  | missing |     1 | 0 / 1 / 0                             |
| `tests/rate_of_closure`      | missing |    19 | 0 / 19 / 0                            |
| `tests/shared`               | missing |     1 | 0 / 1 / 0                             |

Missing files (51):

- `.github/workflows/rate-of-closure-windows-state-security.yml`
- `scripts/four_surface_capability.py`
- `scripts/generate_regional_ground_authority_fixtures.py`
- `scripts/qualify_windows_authority_state_install.py`
- `scripts/rate_campaign_manifest.py`
- `src/movement_optimizer/gui/motion_helpers.py`
- `src/rate_of_closure/_runtime_manifest_reason.py`
- `src/rate_of_closure/four_surface_capability.py`
- `src/rate_of_closure/four_surface_declarations.py`
- `src/rate_of_closure/launch_web_dev.py`
- `src/rate_of_closure/web/src/App.workspaceState.test.tsx`
- `src/rate_of_closure/web/src/components/ChipForgivenessPanel.test.tsx`
- `src/rate_of_closure/web/src/components/ChipForgivenessPanel.tsx`
- `src/rate_of_closure/web/src/components/ClubCanvasCamera.test.tsx`
- `src/rate_of_closure/web/src/components/GroundPlayback3D.test.tsx`
- `src/rate_of_closure/web/src/components/GroundPlayback3D.tsx`
- `src/rate_of_closure/web/src/components/GroundPlaybackLoadedResult.tsx`
- `src/rate_of_closure/web/src/components/GroundPlaybackPanel.test.tsx`
- `src/rate_of_closure/web/src/components/GroundPlaybackPanel.tsx`
- `src/rate_of_closure/web/src/components/RegionalGroundExecutionFileCommands.tsx`
- `src/rate_of_closure/web/src/components/RegionalGroundVariationFileCommands.tsx`
- `src/rate_of_closure/web/src/components/SimulationPhysicsStatus.tsx`
- `src/rate_of_closure/web/src/components/SynchronizedSimulationViews.tsx`
- `src/rate_of_closure/web/src/components/ViewCompositorApp.test.tsx`
- `src/rate_of_closure/web/src/model/chipForgivenessEnsemble.test.ts`
- `src/rate_of_closure/web/src/model/clubAssemblySimulationAdapter.test.ts`
- `src/rate_of_closure/web/src/model/workspaceVariationSession.test.ts`
- `src/rate_of_closure/web/src/workers/windStrategy.worker.ts`
- `src/rate_of_closure/web/tests/e2e/camera-controls.pw.ts`
- `src/rate_of_closure/web/tests/e2e/mobile-tools-menu.pw.ts`
- `tests/ops/test_rate_of_closure_windows_state_workflow.py`
- `tests/rate_of_closure/test_camera_controls_gui.py`
- `tests/rate_of_closure/test_campaign_release_manifest.py`
- `tests/rate_of_closure/test_capability_gui.py`
- `tests/rate_of_closure/test_chip_forgiveness_runner.py`
- `tests/rate_of_closure/test_club_assembly_simulation_adapter.py`
- `tests/rate_of_closure/test_flight_execution_profiles.py`
- `tests/rate_of_closure/test_four_surface_capability.py`
- `tests/rate_of_closure/test_ground_playback_gui.py`
- `tests/rate_of_closure/test_regional_ground_execution_job.py`
- `tests/rate_of_closure/test_regional_ground_execution_workspace.py`
- `tests/rate_of_closure/test_regional_ground_job_preparation.py`
- `tests/rate_of_closure/test_regional_ground_production_runner.py`
- `tests/rate_of_closure/test_regional_ground_result_golden.py`
- `tests/rate_of_closure/test_regional_ground_variation.py`
- `tests/rate_of_closure/test_regional_ground_variation_execution.py`
- `tests/rate_of_closure/test_runtime_manifest.py`
- `tests/rate_of_closure/test_view_compositor_gui.py`
- `tests/rate_of_closure/test_wind_strategy_panel.py`
- `tests/rate_of_closure/test_workspace_session.py`
- `tests/shared/python/golf_club/test_cad_validation.py`

## Totals

- PRs audited: 11 (reachable 11, unreachable 0)
- Files absent from main: 269 (landed-elsewhere 7, missing 244, obsolete 18)
- Groups: 40 (landed-elsewhere 0, missing 37, obsolete 3)

## Golf App Gap Audit & Epic Checklist Reconciliation (Tools #4921)

Reconciled as of: 2026-09-05  
Audit Issue: Tools #4921 (Fleet Readiness Program Phase 0/1)

### Executive Summary

PRs recorded as "merged" across golf epics (#4169, #4173, #4174, #4209, #4212, #4217, #4233, #4246, #4436) had actually merged into stacked `feat/*`/`codex/*` carriers that were folded into #4466 and closed unmerged on 2026-08-20. Forensics across 355 remote stack branches identified substantive lost slices alongside substantial re-landings. All golf program states in `docs/release/rate_of_closure_campaign.v1.json` are reconciled to `main`: zero programs remain in `implemented_on_feature_stack`.

### Delivered versus Missing Slices

| Slice                               | Branches / PRs                                                                                         | Status on `main`                                                                                                                                                                     | Verdict & Action                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| **Inverse Flight Solver**           | #4195, #4196 (`feat/4195-inverse-flight-solver`, `feat/4196-impact-solution-families`)                 | Fully on `main` under `src/rate_of_closure/` and `src/shared/python/` (0 files missing beyond noise)                                                                                 | Delivered. Reconciled in #4191.                                                                     |
| **Wind Physics & Uncertainty**      | #4198, #4199 (`feat/4198-wind-physics`, `feat/4199-wind-uncertainty`, `feat/4199-wind-scalar-adapter`) | Fully on `main` (0 files missing beyond noise)                                                                                                                                       | Delivered. Reconciled in #4191.                                                                     |
| **Wind Workflow UI**                | #4199 (`feat/4199-wind-workflow`)                                                                      | Missing: `windStrategy.worker.ts`, `test_wind_strategy_panel.py`, manifest tooling                                                                                                   | Missing. Tracked in re-slice **#4960**.                                                             |
| **Camera Controls & Preferences**   | #4218, #4571 (`feat/4218-camera-preference-persistence`, `feat/4284-*`)                                | Missing: `camera-controls.pw.ts`, `ClubCanvasCamera.test.tsx`, `test_camera_controls_gui.py`, `SynchronizedSimulationViews.tsx`, `ViewCompositorApp`                                 | Missing. Tracked in re-slice **#4961**.                                                             |
| **Screw-Axis Analytics**            | #4108, #4169 (`feat/screw-axis-analysis`)                                                              | Substantially on `main`: `src/rate_of_closure/simulation/screw_analysis.py`, `ui/pyqt6/screw_overlay.py`, web `screwPresentation.ts`, `tests/rate_of_closure/test_screw_analysis.py` | Delivered. Reconciled in #4108 / #4169.                                                             |
| **Impact-Interval Dynamics**        | #4130 / PR #4133 (`feat/impact-interval-dynamics`)                                                     | Re-landed to `main` via PR #4945: `src/shared/python/swing_sim/impact_interval/` (solver, contact law, types, tests), `docs/physics/IMPACT_INTERVAL_DYNAMICS.md`                     | Delivered via PR **#4945**. Reconciled in #4130.                                                    |
| **Ground / Bounce / Regional**      | #4267, #4268–#4285 (~30 branches)                                                                      | Physics core and execution authorities fully on `main`; shared stack-base residue belongs to wind/capability slice                                                                   | Delivered on `main`; UI/manifest residue tracked in **#4960**.                                      |
| **Multi-view Compositor**           | #4225 family (`feat/4225-multiview-compositor`, `-persistence`)                                        | Multi-view workspace landed via newer re-implementations on `main`; branch-specific paths superseded                                                                                 | Verified and dropped as superseded.                                                                 |
| **Shared Golf Club Builder**        | #4146                                                                                                  | Core builder modules on `main` (`src/shared/python/golf_club/`); C5 image fitting tracked under RM #1506                                                                             | Core delivered; C5 tracked separately.                                                              |
| **LM Player Platform / Neural Lab** | #4212, #4233, #4246                                                                                    | All 19 LM files and 12 v1 Neural Lab files superseded by ADR-0046 canonical layer (`src/shared/python/launch_monitor/`) and v2 models                                                | Superseded. 0 re-land, 85 obsolete, 3 needs-owner (per `closed_stack_gap_audit_decisions.v1.json`). |

### Reconciled Golf Epic Checklists

| Epic              | Title                                              | Delivery Stage           | Status on `main`                                                                               | Action / Follow-up                         |
| ----------------- | -------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **#4103**         | Swing-Impact-Ball-Flight Simulation Platform       | `implemented_unverified` | Re-scoped as golf-app readiness epic; absorbs #4135, #4218, #4571, #4237, #4238                | Release gate owned by #4922                |
| **#4120**         | Investigation and Variation Suite                  | `implemented_unverified` | Durable ensemble, variation session, noise response, and Morris sensitivity analysis on `main` | Overlaps #4142; gate owned by #4922        |
| **#4125**         | Realistic Clubs, Swing Kinetics, Putting, Showcase | `implemented_unverified` | Swing kinetics, putting green model, stroke interchange, and club dynamics on `main`           | Gate owned by #4922                        |
| **#4130**         | Impact-Interval Club Dynamics                      | `implemented_unverified` | Core six-DOF impact dynamics re-landed via PR #4945                                            | Re-landed in PR #4945; gate owned by #4922 |
| **#4142**         | Ensemble Variation, Quiet Zones, Attribution       | `implemented_unverified` | Durable ensemble authorities and variation tabs on `main` (29 landed, 2 partial, 0 missing)    | Open for R14.6 (#4433) and R15.3 golden    |
| **#4146**         | Shared Golf Club Builder                           | `implemented_unverified` | Core builder modules on `main` (`src/shared/python/golf_club/`)                                | Core delivered; C5 tracked under RM #1506  |
| **#4158**         | Wedge Delivery Kinematics & Chip-Shot Explorer     | `implemented_unverified` | Wedge delivery kinematics and chip-shot models on `main` (6 landed, 2 partial)                 | Open for #4162 metrics, #4165 parity       |
| **#4180 / #4181** | LM Convention Registry & Comparability             | `implemented_unverified` | Folded to convention layer; registry and comparability models on `main` (3 landed, 2 partial)  | Open for convention refinements            |
| **#4189**         | Comprehensive 3D D-Plane Calculation               | `implemented_unverified` | 3D D-plane models and visual calculations on `main` (1 landed, 0 partial, 0 missing)           | Delivered on `main`                        |
| **#4191**         | Shot Design, Wind Strategy, Inverse Flight         | `implemented_unverified` | Inverse solver and wind physics on `main` (9 landed, 1 partial); UI panel in #4960             | Open for #4201 / #4922 gate and #4960 UI   |
| **#4201**         | Cross-Interface Validation & Release Evidence      | `specified_only`         | Release gating framework and evidence definitions specified                                    | Gate owned by #4922                        |
| **#4218**         | Modern Toolstrip, Persistence, Workspace           | `implemented_unverified` | Toolstrip, layout, and workspace on `main` (6 landed, 16 partial, 3 missing)                   | Camera persistence tracked in #4961        |
| **#4234**         | Production-Grade Visual Design & Error UX          | `implemented_unverified` | Visual design, theme integration, layout, and error UX on `main`; children in #4103            | Absorbed into #4103; gate in #4922         |
| **#4260**         | Golf Impact & Flight Four-Surface Parity           | `implemented_unverified` | Four-surface capability matrix `docs/release/four_surface_capability.v1.json` on `main`        | Conformance run owned by #4920/#4922       |
| **#4267**         | Qualified Landing, Bounce, Roll, Ground Modeling   | `implemented_unverified` | Physics core and execution authorities on `main` (11 landed, 12 partial, 2 missing)            | Open for playback parity and #4960         |
| **#4433**         | Visual-First Tab Visibility                        | `implemented_unverified` | Core tab implementations on `main`; audit automation manifest open                             | V4 audit automation tracked                |
| **#4571**         | Camera Controls & Presets                          | `implemented_unverified` | Camera contract on `main`; controls, presets, and snap-tracking open                           | Tracked in re-slice #4961                  |
