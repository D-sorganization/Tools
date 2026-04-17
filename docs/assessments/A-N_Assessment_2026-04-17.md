# A-N Assessment - Tools - 2026-04-17

Run time: 2026-04-17T08:01:19.6221680Z UTC
Sync status: synced
Sync notes: Already up to date.
From https://github.com/D-sorganization/Tools
 * branch              codex/tools-issue-2086-api-key-cors -> FETCH_HEAD

Overall grade: C (70/100)

## Coverage Notes
- Reviewed tracked first-party files from git ls-files, excluding cache, build, vendor, virtualenv, temp, and generated output directories.
- Reviewed 2627 tracked files, including 1842 code files, 650 test files, 41 CI files, 76 config/build files, and 494 docs/onboarding files.
- This is a read-only static assessment of committed files. TDD history and confirmed Law of Demeter semantics require commit-history review and deeper call-graph analysis; this report distinguishes those limits from confirmed file evidence.

## Category Grades
### A. Architecture and Boundaries: C (72/100)
Assesses source organization and boundary clarity from tracked first-party layout.
- Evidence: `2627 tracked first-party files`
- Evidence: `1906 files under source-like directories`

### B. Build and Dependency Management: B (84/100)
Assesses committed build, dependency, and tool configuration.
- Evidence: `Chaotic_Pendulum/requirements.txt`
- Evidence: `Makefile`
- Evidence: `pyproject.toml`
- Evidence: `requirements-lock.txt`
- Evidence: `requirements.txt`
- Evidence: `rust_core/tools-core/pyproject.toml`
- Evidence: `src/data_processing/data_processor/ruff.toml`
- Evidence: `src/data_processing/data_processor/web/package-lock.json`
- Evidence: `src/data_processing/data_processor/web/package.json`
- Evidence: `src/data_processing/data_processor/web/tsconfig.json`

### C. Configuration and Environment Hygiene: C (78/100)
Checks whether runtime and developer configuration is explicit.
- Evidence: `Chaotic_Pendulum/requirements.txt`
- Evidence: `Makefile`
- Evidence: `pyproject.toml`
- Evidence: `requirements-lock.txt`
- Evidence: `requirements.txt`
- Evidence: `rust_core/tools-core/pyproject.toml`
- Evidence: `src/data_processing/data_processor/ruff.toml`
- Evidence: `src/data_processing/data_processor/web/package-lock.json`
- Evidence: `src/data_processing/data_processor/web/package.json`
- Evidence: `src/data_processing/data_processor/web/tsconfig.json`

### D. Contracts, Types, and Domain Modeling: B (82/100)
Design by Contract evidence includes validation, assertions, typed models, explicit raised errors, and invariants.
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/config.py`
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/physics.py`
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/renderer.py`
- Evidence: `Chaotic_Pendulum/tests/test_physics.py`
- Evidence: `_bootstrap.py`
- Evidence: `conftest.py`
- Evidence: `launch.py`
- Evidence: `launch_signal_toolkit.py`
- Evidence: `matlab/+rotation_converter/RigidTransform.m`
- Evidence: `migrate_print_to_logging.py`

### E. Reliability and Error Handling: C (76/100)
Reliability is graded from test presence plus explicit validation/error-handling signals.
- Evidence: `.agent/skills/tests/SKILL.md`
- Evidence: `.claude/skills/tests/SKILL.md`
- Evidence: `Chaotic_Pendulum/tests/test_physics.py`
- Evidence: `docs/assessments/Assessment_C_Test_Coverage.md`
- Evidence: `docs/assessments/issues/ISSUE_TEST_COVERAGE.md`
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/config.py`
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/physics.py`
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/renderer.py`
- Evidence: `Chaotic_Pendulum/tests/test_physics.py`
- Evidence: `_bootstrap.py`

### F. Function, Module Size, and SRP: F (55/100)
Evaluates function size, script/module size, and single responsibility using static size signals.
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/renderer.py (733 lines)`
- Evidence: `rust_core/math-primitives/src/geometry.rs (585 lines)`
- Evidence: `rust_core/math-primitives/src/py_bindings.rs (504 lines)`
- Evidence: `rust_core/math-primitives/src/types.rs (520 lines)`
- Evidence: `rust_core/tools-core/src/ball_flight.rs (1054 lines)`
- Evidence: `scripts/mypy_autofix_agent.py (724 lines)`
- Evidence: `src/asteroid_jumper/renderer.py (551 lines)`
- Evidence: `matlab/test_rotation_converter.m (coarse avg 84 lines/definition)`
- Evidence: `scripts/generate_assessments.py (coarse avg 315 lines/definition)`
- Evidence: `scripts/quality-check.py (coarse avg 92 lines/definition)`

### G. Testing and TDD Posture: B (82/100)
TDD history cannot be confirmed statically; grade reflects committed automated test posture.
- Evidence: `.agent/skills/tests/SKILL.md`
- Evidence: `.claude/skills/tests/SKILL.md`
- Evidence: `Chaotic_Pendulum/tests/test_physics.py`
- Evidence: `docs/assessments/Assessment_C_Test_Coverage.md`
- Evidence: `docs/assessments/issues/ISSUE_TEST_COVERAGE.md`
- Evidence: `docs/development/TEST_COVERAGE_ANALYSIS.md`
- Evidence: `docs/development/TEST_IMPROVEMENTS_SUMMARY.md`
- Evidence: `matlab/test_rotation_converter.m`
- Evidence: `scripts/check_minimum_test_contract.py`
- Evidence: `src/asteroid_jumper/tests/__init__.py`
- Evidence: `src/asteroid_jumper/tests/test_smoke.py`
- Evidence: `src/c3d_viewer/tests/__init__.py`

### H. CI/CD and Automation: C (78/100)
Checks for tracked CI/CD workflow files.
- Evidence: `.github/workflows/Bot-CI-Trigger.yml`
- Evidence: `.github/workflows/Code-Metrics.yml`
- Evidence: `.github/workflows/Comment-to-Issue-Converter.yml`
- Evidence: `.github/workflows/Jules-Archivist.yml`
- Evidence: `.github/workflows/Jules-Auto-Assign-Issues.yml`
- Evidence: `.github/workflows/Jules-Auto-Rebase.yml`
- Evidence: `.github/workflows/Jules-Auto-Refactor.yml`
- Evidence: `.github/workflows/Jules-Auto-Repair.yml`
- Evidence: `.github/workflows/Jules-Cleaner.yml`
- Evidence: `.github/workflows/Jules-Code-Quality-Fixer.yml`

### I. Security and Secret Hygiene: F (35/100)
Secret scan is regex-based; findings require manual confirmation.
- Evidence: `src/media_processing/video_processor/apps/web/lib/__tests__/csrf.test.ts`
- Evidence: `tests/test_integration_test_helpers.py`

### J. Documentation and Onboarding: B (82/100)
Checks docs, README, onboarding, and release documents.
- Evidence: `.Jules/README.md`
- Evidence: `.Jules/palette.md`
- Evidence: `.agent/skills/issues-10-sequential/SKILL.md`
- Evidence: `.agent/skills/issues-5-combined/SKILL.md`
- Evidence: `.agent/skills/lint/SKILL.md`
- Evidence: `.agent/skills/tests/SKILL.md`
- Evidence: `.agent/skills/update-issues/SKILL.md`
- Evidence: `.agent/workflows/fix_summary.md`
- Evidence: `.agent/workflows/issues-10-sequential.md`
- Evidence: `.agent/workflows/issues-5-combined.md`
- Evidence: `.agent/workflows/lint.md`
- Evidence: `.agent/workflows/tests.md`

### K. Maintainability, DRY, and Duplication: F (55/100)
DRY is assessed through duplicate filename clusters and TODO/FIXME density as static heuristics.
- Evidence: `api appears in 4 files`
- Evidence: `app appears in 16 files`
- Evidence: `build appears in 5 files`
- Evidence: `cli appears in 4 files`
- Evidence: `config appears in 4 files`
- Evidence: `scripts/analyze_completist_data.py`
- Evidence: `scripts/generate_assessments.py`
- Evidence: `scripts/generate_comprehensive_assessment.py`
- Evidence: `scripts/generate_fresh_assessments.py`
- Evidence: `scripts/legacy_tools/code_quality_check.py`

### L. API Surface and Law of Demeter: F (58/100)
Law of Demeter is approximated with deep member-chain hints; confirmed violations require semantic review.
- Evidence: `Chaotic_Pendulum/chaotic_pendulum/renderer.py`
- Evidence: `UnifiedToolsLauncher.py`
- Evidence: `launch_signal_toolkit.py`
- Evidence: `rust_core/tools-core/src/ball_flight.rs`
- Evidence: `scripts/generate_comprehensive_assessment.py`
- Evidence: `scripts/generate_theme_screenshots.py`
- Evidence: `src/asteroid_jumper/controller.py`
- Evidence: `src/asteroid_jumper/controls_panel.py`
- Evidence: `src/asteroid_jumper/main_window.py`
- Evidence: `src/asteroid_jumper/metrics_panel.py`

### M. Observability and Operability: C (74/100)
Checks for logging, metrics, monitoring, and operational artifacts.
- Evidence: `.github/workflows/Code-Metrics.yml`
- Evidence: `docs/assessments/Assessment_L_Logging.md`
- Evidence: `docs/assessments/issues/ISSUE_LOGGING_SPLIT.md`
- Evidence: `migrate_print_to_logging.py`
- Evidence: `scripts/convert_print_to_logging.py`
- Evidence: `src/asteroid_jumper/metrics_panel.py`
- Evidence: `src/data_processing/data_processor/python/data_processor/logging_config.py`
- Evidence: `src/media_processing/video_processor/apps/web/components/golf/MetricsPanel.tsx`
- Evidence: `src/media_processing/video_processor/apps/web/lib/__tests__/logger.test.ts`
- Evidence: `src/media_processing/video_processor/apps/web/lib/logger.ts`

### N. Governance, Licensing, and Release Hygiene: C (74/100)
Checks ownership, release, contribution, security, and license metadata.
- Evidence: `.github/CODEOWNERS`
- Evidence: `.github/agents/security-agent.md`
- Evidence: `CHANGELOG.md`
- Evidence: `CONTRIBUTING.md`
- Evidence: `LICENSE`
- Evidence: `SECURITY.md`
- Evidence: `docs/assessments/Assessment_F_Security.md`
- Evidence: `docs/assessments/issues/ISSUE_SECURITY_DATA_LEAKAGE.md`
- Evidence: `docs/assessments/issues/Issue_F_Security.md`
- Evidence: `docs/changelogs/2026-01-31-fleet-updates.md`

## Explicit Engineering Practice Review
- TDD: Automated tests are present, but red-green-refactor history is not confirmable from static files.
- DRY: Duplicate responsibility clusters require review: api appears in 4 files; app appears in 16 files; build appears in 5 files; cli appears in 4 files; config appears in 4 files
- Design by Contract: Validation/contract signals were found in tracked code.
- Law of Demeter: Deep member-chain hints were found and should be semantically reviewed.
- Function size and SRP: Large modules or coarse long-definition signals were found.

## Key Risks
- Large modules/scripts reduce maintainability and SRP clarity.
- Potential hard-coded secret patterns require manual security review.
- Repeated filename clusters suggest possible duplicated responsibilities.
- Deep member-chain usage may indicate Law of Demeter pressure points.

## Prioritized Remediation Recommendations
1. Split the largest modules by responsibility and add characterization tests before refactoring.
2. Review duplicate filename/responsibility clusters and extract shared helpers only where behavior is truly repeated.
3. Review deep member chains and introduce boundary methods where object graph traversal leaks across modules.

## Actionable Issue Candidates
### Split oversized modules by responsibility
- Severity: medium
- Problem: Oversized files found: Chaotic_Pendulum/chaotic_pendulum/renderer.py (733 lines); rust_core/math-primitives/src/geometry.rs (585 lines); rust_core/math-primitives/src/py_bindings.rs (504 lines); rust_core/math-primitives/src/types.rs (520 lines); rust_core/tools-core/src/ball_flight.rs (1054 lines); scripts/mypy_autofix_agent.py (724 lines); src/asteroid_jumper/renderer.py (551 lines); src/c3d_viewer/python/c3d_viewer/ui/pyqt6/main_window.py (750 lines); src/data_processing/data_processor/python/benchmarks/performance_benchmark.py (564 lines); src/data_processing/data_processor/python/data_processor/core/__init__.py (616 lines); src/data_processing/data_processor/python/data_processor/core/augmentation_transforms.py (765 lines); src/data_processing/data_processor/python/data_processor/core/cross_correlation.py (1056 lines); src/data_processing/data_processor/python/data_processor/core/dataset_manager.py (611 lines); src/data_processing/data_processor/python/data_processor/core/feature_extractor.py (772 lines); src/data_processing/data_processor/python/data_processor/core/kalman_filter.py (821 lines); src/data_processing/data_processor/python/data_processor/core/nn_script_exporter_renderers.py (584 lines); src/data_processing/data_processor/python/data_processor/core/nn_trainer.py (840 lines); src/data_processing/data_processor/python/data_processor/core/outlier_detection.py (757 lines); src/data_processing/data_processor/python/data_processor/core/pca_analysis.py (621 lines); src/data_processing/data_processor/python/data_processor/core/script_generator.py (608 lines); src/data_processing/data_processor/python/data_processor/core/signal_processing.py (808 lines); src/data_processing/data_processor/python/data_processor/core/spectral_analysis.py (805 lines); src/data_processing/data_processor/python/data_processor/core/state_space.py (1006 lines); src/data_processing/data_processor/python/data_processor/core/surface_plot.py (654 lines); src/data_processing/data_processor/python/data_processor/core/time_series_decomposition.py (679 lines); src/data_processing/data_processor/python/data_processor/core/uncertainty_quantification.py (998 lines); src/data_processing/data_processor/python/data_processor/core/undo_redo.py (504 lines); src/data_processing/data_processor/python/data_processor/core/wavelet_denoising.py (594 lines); src/data_processing/data_processor/python/data_processor/gui_refactored.py (801 lines); src/data_processing/data_processor/python/data_processor/high_performance_loader.py (590 lines); src/data_processing/data_processor/python/data_processor/ui/folder_tool_tab.py (678 lines); src/data_processing/data_processor/python/data_processor/ui/format_converter_tab.py (730 lines); src/data_processing/data_processor/python/data_processor/ui/pyqt6/main_window.py (733 lines); src/data_processing/data_processor/python/data_processor/ui/pyqt6/main_window_tabs.py (754 lines); src/data_processing/data_processor/python/data_processor/ui/pyqt6/statistical_widgets.py (548 lines); src/data_processing/data_processor/python/data_processor/ui/pyqt6/visualization_widgets.py (509 lines); src/data_processing/data_processor/python/data_processor/ui/pyqt6/widgets.py (545 lines); src/data_processing/data_processor/python/tests/test_advanced_analysis.py (873 lines); src/data_processing/data_processor/python/tests/test_signal_processing_core.py (596 lines); src/data_processing/data_processor/python/tests/test_statistical_analysis.py (888 lines); src/data_processing/data_processor/web/src/components/AnalyticsSuite.tsx (834 lines); src/data_processing/data_processor/web/src/hooks/useDataProcessor.ts (1031 lines); src/document_processing/pdf_renamer/src/pdf_renamer/gui.py (979 lines); src/financial_calculator/python/financial_calculator/ui/pyqt6/main_window.py (604 lines); src/flow_rate_converter/python/flow_rate_converter/ui/pyqt6/main_window.py (551 lines); src/folder_packer_pro/ui_tabs.py (649 lines); src/folder_tool/folder_tool_ui.py (748 lines); src/function_generator/python/function_generator/ui/pyqt6/main_window.py (630 lines); src/function_generator/web/src/components/FunctionGenerator.tsx (946 lines); src/humanoid_builder_gui/python/humanoid_builder_gui/ui/pyqt6/main_window.py (829 lines); src/inertia_calculator/python/inertia_calculator/ui/pyqt6/main_window.py (611 lines); src/lower_body_model/simulator.py (771 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/CONVOLUTION_REVERB_EXAMPLES.m (632 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/ENHANCEMENT_EXAMPLES.m (567 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/AdvancedAudioProcessor.m (1012 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/AntiAliasingTools.m (779 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/AudioEditor.m (823 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/AudioEffects.m (572 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/ConvolutionReverb.m (940 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/MixerCoreEnhanced.m (901 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/MusicProductionTools.m (1265 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/core/WaveletProcessor.m (685 lines); src/media_processing/audio_processor/matlab/audio_signal_processor/gui/MainWindow.m (3522 lines); src/media_processing/video_processor/apps/web/components/video/__tests__/VideoUploader.test.tsx (546 lines); src/media_processing/video_processor/apps/web/lib/__tests__/csrf.test.ts (717 lines); src/media_processing/video_processor/apps/web/lib/golf/persistence.ts (503 lines); src/media_processing/video_processor/apps/web/lib/golf/phaseDetector.ts (551 lines); src/media_processing/video_processor/apps/web/lib/golf/reportGenerator.ts (594 lines); src/media_processing/video_processor/apps/web/lib/golf/swingAnalyzer.ts (772 lines); src/multi_param_analysis/python/multi_param_analysis/ui/pyqt6/main_window.py (706 lines); src/ode_solver/python/ode_solver/ui/pyqt6/main_window.py (560 lines); src/ode_solver/web/src/components/ODESolverCalculator.tsx (523 lines); src/optimizer_gui/python/optimizer_gui/ui/pyqt6/main_window.py (784 lines); src/pendulum_simulator/double_pendulum_colab.ipynb (776 lines); src/pendulum_simulator/pendulum-core/src/cmaes.rs (532 lines); src/pendulum_simulator/pendulum-core/src/golfer.rs (532 lines); src/pendulum_simulator/pendulum-core/src/lib.rs (1619 lines); src/pendulum_simulator/pendulum-web/src/App.tsx (691 lines); src/pendulum_simulator/pendulum-web/src/physics.ts (522 lines); src/pendulum_simulator/pendulum-web/src/physics_golfer.ts (524 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/analysis_tab.py (778 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/base_pendulum_widget.py (830 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget.py (661 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget_base.py (585 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget_golfer.py (594 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/equations_data.py (1105 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/golfer_pendulum_widget.py (784 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/main_window.py (711 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/optimization_widget.py (840 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py (920 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/pendulum_widget.py (927 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/perturbation_panel.py (549 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/simulation_panel.py (940 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/swing_comparison_dialog.py (536 lines); src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py (876 lines); src/pendulum_simulator/src/double_pendulum_golf/native_backend.py (762 lines); src/pendulum_simulator/src/double_pendulum_golf/physics.py (761 lines); src/pendulum_simulator/src/double_pendulum_golf/physics_golfer_jax.py (691 lines); src/pendulum_simulator/src/double_pendulum_golf/physics_triple.py (649 lines); src/pendulum_simulator/tests/test_friction_triple.py (568 lines); src/pendulum_simulator/tests/test_golfer_dynamics_extended.py (598 lines); src/pendulum_simulator/tests/test_jacobians_golfer.py (593 lines); src/pendulum_simulator/tests/test_native_backend.py (546 lines); src/pendulum_simulator/tests/test_physics_golfer_jax.py (504 lines); src/pressure_drop_calculator/python/pressure_drop_calculator/ui/pyqt6/main_window.py (509 lines); src/python/src/help/help_content.py (837 lines); src/python/src/help/help_system.py (973 lines); src/python/src/utils/integration_test_helpers.py (872 lines); src/python/src/utils/test_utils.py (914 lines); src/python/tests/test_folders_tool.py (677 lines); src/rotation_converter/core.py (505 lines); src/rotation_converter/rigid_transform.py (777 lines); src/rotation_converter/screw_visualization.py (533 lines); src/rrt_path_planner/matlab/src/gui/starWarsPathPlannerGUI.m (927 lines); src/rrt_path_planner/python/src/star_wars_rrt.py (920 lines); src/shared/python/calc_backend/tests/test_calc_backend.py (1052 lines); src/shared/python/calc_backend/tests/test_calc_backend_gaps.py (506 lines); src/shared/python/contracts.py (650 lines); src/shared/python/data_processing/processor.py (541 lines); src/shared/python/data_processing/tests/test_processor.py (536 lines); src/shared/python/gui_launcher/launcher.py (733 lines); src/shared/python/humanoid_character_builder/core/anthropometry.py (629 lines); src/shared/python/humanoid_character_builder/core/segment_definitions.py (759 lines); src/shared/python/humanoid_character_builder/generators/mesh_generator_makehuman.py (574 lines); src/shared/python/humanoid_character_builder/generators/mesh_generator_smplx.py (528 lines); src/shared/python/humanoid_character_builder/generators/urdf_generator.py (762 lines); src/shared/python/humanoid_character_builder/interfaces/api.py (705 lines); src/shared/python/humanoid_character_builder/mesh/collision_generator.py (762 lines); src/shared/python/humanoid_character_builder/mesh/inertia_calculator.py (634 lines); src/shared/python/humanoid_character_builder/mesh/mesh_processor.py (813 lines); src/shared/python/model_generation/api/rest_api_routes.py (1061 lines); src/shared/python/model_generation/builders/manual_builder.py (584 lines); src/shared/python/model_generation/builders/parametric_builder.py (665 lines); src/shared/python/model_generation/builders/urdf_writer.py (696 lines); src/shared/python/model_generation/cli/main.py (827 lines); src/shared/python/model_generation/converters/mjcf_converter.py (649 lines); src/shared/python/model_generation/converters/simscape/mdl_parser.py (611 lines); src/shared/python/model_generation/converters/simscape/simscape_converter.py (823 lines); src/shared/python/model_generation/converters/urdf_parser.py (567 lines); src/shared/python/model_generation/core/physics_validation.py (570 lines); src/shared/python/model_generation/core/types.py (714 lines); src/shared/python/model_generation/core/validation.py (507 lines); src/shared/python/model_generation/editor/editor_modifications.py (707 lines); src/shared/python/model_generation/editor/frankenstein_editor.py (708 lines); src/shared/python/model_generation/editor/text_editor.py (1039 lines); src/shared/python/model_generation/explorer/model_explorer.py (717 lines); src/shared/python/model_generation/inertia/calculator.py (621 lines); src/shared/python/model_generation/library/model_library.py (880 lines); src/shared/python/model_generation/tests/test_unified_loader.py (703 lines); src/shared/python/plot_engine/matplotlib_renderer.py (578 lines); src/shared/python/plot_theme/tests/test_plot_theme.py (726 lines); src/shared/python/plot_theme/themes.py (596 lines); src/shared/python/programmatic_pid/cli.py (661 lines); src/shared/python/signal_toolkit/calculus.py (624 lines); src/shared/python/signal_toolkit/core.py (686 lines); src/shared/python/signal_toolkit/filters.py (817 lines); src/shared/python/signal_toolkit/fitting.py (840 lines); src/shared/python/signal_toolkit/io.py (704 lines); src/shared/python/signal_toolkit/limits.py (513 lines); src/shared/python/signal_toolkit/noise.py (556 lines); src/shared/python/signal_toolkit/polynomial_generator.py (649 lines); src/shared/python/signal_toolkit/series.py (747 lines); src/shared/python/signal_toolkit/tests/test_series.py (764 lines); src/shared/python/signal_toolkit/tests/test_signal_toolkit.py (801 lines); src/shared/python/signal_toolkit/tests/test_signal_toolkit_extended.py (669 lines); src/shared/python/signal_toolkit/widget_processing.py (804 lines); src/shared/python/signal_toolkit/widget_ui.py (962 lines); src/shared/python/theme/colors.py (563 lines); src/shared/python/theme/dialogs/theme_manager_dialog.py (507 lines); src/shared/python/theme/stylesheets.py (615 lines); src/shared/python/theme/theme_manager.py (566 lines); src/shared/python/upstream_drift_tools/calculators/conversion/flow_rate_converter.py (701 lines); src/shared/python/upstream_drift_tools/calculators/conversion/service.py (894 lines); src/shared/python/upstream_drift_tools/calculators/conversion/tables.py (619 lines); src/shared/python/upstream_drift_tools/calculators/electrical/electrical_model.py (513 lines); src/shared/python/upstream_drift_tools/calculators/thermo/steam_engine.py (911 lines); src/shared/python/upstream_drift_tools/data_processing/core.py (647 lines); src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py (918 lines); src/shared/python/upstream_drift_tools/process_calculators/acid_gas_dewpoint_calculator.py (929 lines); src/shared/python/upstream_drift_tools/process_calculators/constants.py (616 lines); src/shared/python/upstream_drift_tools/process_calculators/optimization.py (548 lines); src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/utils/gas_properties.py (948 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/References/psa_stage_removal_sensitivity.ipynb (936 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_analysis.ipynb (1181 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_analysis_colab.ipynb (711 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_webapp.py (694 lines); src/shared/python/upstream_drift_tools/process_calculators/scrubber_calculator.py (803 lines); src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py (790 lines); src/shared/python/upstream_drift_tools/process_calculators/syngas_water_calculator.py (735 lines); src/shared/python/upstream_drift_tools/process_calculators/wgs_reactor_calculator.py (734 lines); src/shared/python/upstream_drift_tools/tests/calculators/conversion/test_conversion_service.py (657 lines); src/shared/python/upstream_drift_tools/tests/process_calculators/test_psa_model.py (522 lines); src/shared/python/upstream_drift_tools/tests/test_data_processor_engine.py (710 lines); src/shared/python/upstream_drift_tools/ui/mixins/calculator_state_mixin.py (805 lines); src/shared/python/upstream_drift_tools/ui/widgets/data_processor_widget.py (605 lines); src/shared/python/upstream_drift_tools/ui/widgets/unit_converter_widget.py (566 lines); src/shared/python/upstream_drift_tools/utils/state_manager.py (538 lines); src/shared/python/upstream_drift_tools/utils/unit_constants.py (576 lines); src/solar_system_model/solar_system/core/celestial_body.py (768 lines); src/solar_system_model/solar_system/data/historical_events.py (922 lines); src/solar_system_model/solar_system/physics/orbital_mechanics.py (590 lines); src/solar_system_model/solar_system/physics/trajectory_planner.py (791 lines); src/solar_system_model/solar_system/ui/widgets.py (1046 lines); src/solar_system_model/solar_system/visualization/camera.py (560 lines); src/solar_system_model/solar_system/visualization/renderer.py (966 lines); src/solar_system_model/solar_system/visualization/scene.py (585 lines); src/solar_system_model/solar_system/visualization/ui_renderer.py (817 lines); src/steam_engine_calculator/python/steam_engine_calculator/ui/pyqt6/main_window.py (728 lines); src/steam_engine_calculator/web/src/components/SteamEngineCalculator.tsx (524 lines); src/tools/matlab_quality_utils.py (600 lines); src/tools/mypy_autofix_agent.py (724 lines); src/urdf_builder_gui/tests/test_urdf_builder_gui.py (644 lines); src/vessel_drafter/python/vessel_drafter/gui/vessel_drafter_window.py (570 lines); src/vessel_drafter/python/vessel_drafter/preview/vessel_drafter_scene.py (770 lines); src/web_applications/calculator/calculator.py (838 lines); src/web_applications/unit_converter/converter.py (638 lines); src/web_applications/unit_converter/unit-converter-app/app.js (1072 lines); src/web_applications/unit_converter/unit-converter-app/converter.js (1049 lines); tests/architecture/test_layer_boundaries.py (510 lines); tests/conftest.py (545 lines); tests/data_processing/data_processor/test_neural_network.py (675 lines); tests/rotation_converter/test_modern_robotics.py (708 lines); tests/rotation_converter/test_rigid_transform.py (1077 lines); tests/rotation_converter/test_rotation_core.py (540 lines); tests/test_integration_test_helpers.py (522 lines); tests/test_review_fixes_2026_03_09.py (563 lines); tests/tools/test_matlab_quality_utils.py (594 lines)
- Evidence: Category F lists files over 500 lines or coarse long-definition signals.
- Impact: Large modules obscure ownership, complicate review, and weaken SRP.
- Proposed fix: Add characterization tests, then split cohesive responsibilities into smaller modules.
- Acceptance criteria: Largest files are reduced or justified; extracted modules have focused tests.
- Expectations: SRP, function size, module size, maintainability

### Review duplicated responsibility clusters
- Severity: medium
- Problem: Repeated filename clusters found: api appears in 4 files; app appears in 16 files; build appears in 5 files; cli appears in 4 files; config appears in 4 files
- Evidence: Category K duplicate-name clustering found repeated responsibility names.
- Impact: Potential duplicated logic increases maintenance cost and drift risk.
- Proposed fix: Review clusters, remove accidental duplication, and extract shared helpers where behavior is truly common.
- Acceptance criteria: Documented review of clusters; duplicated implementations are consolidated or justified.
- Expectations: DRY, maintainability, SRP

### Investigate potential hard-coded secret patterns
- Severity: high
- Problem: Potential secret-like assignments found in: src/media_processing/video_processor/apps/web/lib/__tests__/csrf.test.ts; tests/test_integration_test_helpers.py
- Evidence: Category I regex scan matched secret-like assignments.
- Impact: Hard-coded secrets can expose credentials and create security incidents.
- Proposed fix: Manually verify findings, rotate any exposed credentials, and move secrets to environment or secret management.
- Acceptance criteria: Secret scan is clean or findings are documented false positives; exposed credentials are rotated.
- Expectations: security, reliability

### Review deep object traversal hotspots
- Severity: medium
- Problem: Deep member-chain hints found in: Chaotic_Pendulum/chaotic_pendulum/renderer.py; UnifiedToolsLauncher.py; launch_signal_toolkit.py; rust_core/tools-core/src/ball_flight.rs; scripts/generate_comprehensive_assessment.py; scripts/generate_theme_screenshots.py; src/asteroid_jumper/controller.py; src/asteroid_jumper/controls_panel.py
- Evidence: Category L found repeated chains with three or more member hops.
- Impact: Law of Demeter pressure can make APIs brittle and increase coupling.
- Proposed fix: Review hotspots and introduce boundary methods or DTOs where callers traverse object graphs.
- Acceptance criteria: Hotspots are documented, simplified, or justified; tests cover any API boundary changes.
- Expectations: Law of Demeter, SRP, maintainability

