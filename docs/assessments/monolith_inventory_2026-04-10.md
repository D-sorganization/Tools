# Monolith Inventory — 2026-04-10

**Related issue:** Tools #1997 — Establish 500 LOC CI gate.

## The 500 LOC Rule

Every `.py` and `.rs` file under `src/` must stay at or below **500 lines of
code**. New files above the budget fail CI (`check_file_size_budget.py`). The
goal is to keep modules cohesive, reviewable, and testable, and to prevent the
regrowth of monoliths after a decomposition pass.

### Baseline / grandfathering

The 183 files listed below are grandfathered in the baseline
(`scripts/monolith_baseline.txt`) so CI does not block while decomposition is
in progress. The baseline is a plain newline-separated list of repo-relative
paths (POSIX-style slashes). A file is ignored by the gate if and only if its
normalized path appears in the baseline.

The correct workflow is to **remove** entries from the baseline as files are
refactored below 500 LOC — not to add new entries. A new monolith should be
rejected in review; if there is a legitimate reason to add one, add the path
to `scripts/monolith_baseline.txt` in the same PR and link an issue with a
decomposition plan.

### How to run the gate locally

```bash
python3 scripts/check_file_size_budget.py \
    --max-loc 500 \
    --baseline-file scripts/monolith_baseline.txt
```

Use `--changed-only` on a feature branch to scan only the files that differ
from `origin/staging`:

```bash
python3 scripts/check_file_size_budget.py \
    --max-loc 500 \
    --changed-only \
    --baseline-file scripts/monolith_baseline.txt
```

## Burn-down plan

Decompose the top 10 monoliths per iteration. Prefer:

1. Extracting pure calc kernels into `src/shared/python/` or `src/shared/tools/`.
2. Splitting PyQt6 UI files into tab/panel modules plus a thin controller.
3. Breaking Rust source into sub-modules by concern (math, io, ffi).
4. Splitting large tests along feature seams and extracting shared fixtures.

## Inventory (sorted by LOC, descending)

| File | LOC | Suggested split strategy |
|------|-----|--------------------------|
| `src/rotation_converter/modern_robotics.py` | 2101 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/rotation_converter/ui/pyqt6/main_window.py` | 1233 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py` | 1218 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/equations_popup.py` | 1155 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py` | 1090 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/calc_backend/tests/test_calc_backend.py` | 1088 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_gui.py` | 1042 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/core/anova.py` | 1027 | Decompose pipeline stages into composable functions in a sub-package |
| `src/data_processing/data_processor/python/data_processor/core/cross_correlation.py` | 1014 | Decompose pipeline stages into composable functions in a sub-package |
| `src/data_processing/data_processor/python/data_processor/core/time_series_decomposition.py` | 992 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/model_generation/api/rest_api_routes.py` | 980 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/vectorized_filter_engine.py` | 975 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/python/src/help/help_system.py` | 975 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/core/uncertainty_quantification.py` | 972 | Decompose pipeline stages into composable functions in a sub-package |
| `src/data_processing/data_processor/python/data_processor/core/state_space.py` | 970 | Decompose pipeline stages into composable functions in a sub-package |
| `src/document_processing/pdf_renamer/src/pdf_renamer/gui.py` | 965 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/web_applications/calculator/calculator.py` | 962 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/signal_toolkit/widget_ui.py` | 957 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/solar_system_model/solar_system/visualization/renderer.py` | 951 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/simulation_panel.py` | 934 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/utils/gas_properties.py` | 928 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/upstream_drift_tools/process_calculators/acid_gas_dewpoint_calculator.py` | 920 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/rrt_path_planner/python/src/star_wars_rrt.py` | 915 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/python/src/utils/test_utils.py` | 909 | Split tests by feature area; extract shared fixtures into conftest |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py` | 906 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/pendulum_widget.py` | 905 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py` | 904 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/calculators/thermo/steam_engine.py` | 898 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/tests/test_statistical_analysis.py` | 879 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/programmatic_pid/cli.py` | 872 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/optimizer_gui/python/optimizer_gui/ui/pyqt6/main_window.py` | 867 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/python/src/utils/integration_test_helpers.py` | 867 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/model_generation/library/model_library.py` | 865 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/core/kalman_filter.py` | 863 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/pressure_drop_interface.py` | 845 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/humanoid_builder_gui/python/humanoid_builder_gui/ui/pyqt6/main_window.py` | 832 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/python/src/help/help_content.py` | 832 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/signal_toolkit/fitting.py` | 826 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/model_generation/cli/main.py` | 822 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/tests/test_advanced_analysis.py` | 821 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/upstream_drift_tools/process_calculators/scrubber_calculator.py` | 821 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/core/nn_trainer.py` | 810 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/humanoid_character_builder/mesh/mesh_processor.py` | 802 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/model_generation/converters/simscape/simscape_converter.py` | 800 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/signal_toolkit/filters.py` | 800 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/signal_toolkit/widget_processing.py` | 797 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/signal_toolkit/tests/test_signal_toolkit.py` | 796 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/upstream_drift_tools/ui/mixins/calculator_state_mixin.py` | 795 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/data_processing/data_processor/python/data_processor/gui_refactored.py` | 794 | Decompose pipeline stages into composable functions in a sub-package |
| `src/solar_system_model/solar_system/visualization/ui_renderer.py` | 794 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/core/signal_processing.py` | 786 | Decompose pipeline stages into composable functions in a sub-package |
| `src/pendulum_simulator/src/double_pendulum_golf/physics_golfer_jax.py` | 786 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/steam_engine_calculator/python/steam_engine_calculator/ui/pyqt6/main_window.py` | 784 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/data_processing/data_processor/python/data_processor/core/spectral_analysis.py` | 783 | Decompose pipeline stages into composable functions in a sub-package |
| `src/solar_system_model/solar_system/physics/trajectory_planner.py` | 772 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/analysis_tab.py` | 767 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/rotation_converter/rigid_transform.py` | 767 | Decompose pipeline stages into composable functions in a sub-package |
| `src/multi_param_analysis/python/multi_param_analysis/ui/pyqt6/main_window.py` | 761 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/optimization_widget.py` | 759 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/signal_toolkit/tests/test_series.py` | 759 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/upstream_drift_tools/process_calculators/wgs_reactor_calculator.py` | 757 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/humanoid_character_builder/core/segment_definitions.py` | 754 | Extract helper classes/functions into side modules; keep core as facade |
| `src/vessel_drafter/python/vessel_drafter/preview/vessel_drafter_scene.py` | 753 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/ui/pyqt6/main_window_tabs.py` | 749 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/golfer_pendulum_widget.py` | 749 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/solar_system_model/solar_system/core/celestial_body.py` | 748 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/humanoid_character_builder/mesh/collision_generator.py` | 746 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/core/feature_extractor.py` | 744 | Decompose pipeline stages into composable functions in a sub-package |
| `src/c3d_viewer/python/c3d_viewer/ui/pyqt6/main_window.py` | 742 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/folder_tool/folder_tool_ui.py` | 740 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/humanoid_character_builder/generators/urdf_generator.py` | 740 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/pendulum_simulator/src/double_pendulum_golf/native_backend.py` | 737 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/core/nn_script_exporter.py` | 735 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/signal_toolkit/series.py` | 734 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/core/augmentation_transforms.py` | 733 | Decompose pipeline stages into composable functions in a sub-package |
| `src/data_processing/data_processor/python/data_processor/core/outlier_detection.py` | 730 | Decompose pipeline stages into composable functions in a sub-package |
| `src/data_processing/data_processor/python/data_processor/ui/pyqt6/main_window.py` | 726 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/plot_theme/tests/test_plot_theme.py` | 721 | Split tests by feature area; extract shared fixtures into conftest |
| `src/pendulum_simulator/src/double_pendulum_golf/physics.py` | 720 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/process_calculators/syngas_water_calculator.py` | 712 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/ui/format_converter_tab.py` | 708 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/model_generation/explorer/model_explorer.py` | 708 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/upstream_drift_tools/tests/test_data_processor_engine.py` | 705 | Split tests by feature area; extract shared fixtures into conftest |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget.py` | 698 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/model_generation/core/types.py` | 696 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/financial_calculator/python/financial_calculator/ui/pyqt6/main_window.py` | 695 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/model_generation/editor/editor_modifications.py` | 693 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/humanoid_character_builder/interfaces/api.py` | 690 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/signal_toolkit/io.py` | 690 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/model_generation/editor/frankenstein_editor.py` | 689 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/upstream_drift_tools/calculators/conversion/flow_rate_converter.py` | 688 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_webapp.py` | 686 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/model_generation/builders/urdf_writer.py` | 678 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/python/tests/test_folders_tool.py` | 672 | Split tests by feature area; extract shared fixtures into conftest |
| `src/inertia_calculator/python/inertia_calculator/ui/pyqt6/main_window.py` | 669 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/tests/test_analytical_jacobians.py` | 667 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/signal_toolkit/tests/test_signal_toolkit_extended.py` | 664 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/signal_toolkit/core.py` | 663 | Extract helper classes/functions into side modules; keep core as facade |
| `src/data_processing/data_processor/python/data_processor/ui/folder_tool_tab.py` | 657 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/gui_launcher/launcher.py` | 657 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/base_pendulum_widget.py` | 652 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/upstream_drift_tools/tests/calculators/conversion/test_conversion_service.py` | 652 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/model_generation/builders/parametric_builder.py` | 651 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/contracts.py` | 644 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/data_processing/core.py` | 641 | Extract helper classes/functions into side modules; keep core as facade |
| `src/shared/python/model_generation/converters/mjcf_converter.py` | 637 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/function_generator/python/function_generator/ui/pyqt6/main_window.py` | 636 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/data_processing/data_processor/python/data_processor/core/surface_plot.py` | 635 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/signal_toolkit/polynomial_generator.py` | 632 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/folder_packer_pro/ui_tabs.py` | 629 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/web_applications/unit_converter/converter.py` | 624 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/humanoid_character_builder/mesh/inertia_calculator.py` | 622 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget_golfer.py` | 617 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/rotation_converter/_mr_dynamics.py` | 617 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/humanoid_character_builder/core/anthropometry.py` | 617 | Extract helper classes/functions into side modules; keep core as facade |
| `src/pendulum_simulator/src/double_pendulum_golf/physics_triple.py` | 616 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/calculators/conversion/tables.py` | 614 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/upstream_drift_tools/process_calculators/constants.py` | 611 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/model_generation/inertia/calculator.py` | 610 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/theme/stylesheets.py` | 610 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/core/pca_analysis.py` | 606 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/signal_toolkit/calculus.py` | 604 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/rotation_converter/_mr_rotation_matrices.py` | 603 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/ui/widgets/data_processor_widget.py` | 603 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/main_window.py` | 601 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/model_generation/converters/simscape/mdl_parser.py` | 600 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/core/dataset_manager.py` | 598 | Decompose pipeline stages into composable functions in a sub-package |
| `src/pendulum_simulator/tests/test_golfer_dynamics_extended.py` | 597 | Split tests by feature area; extract shared fixtures into conftest |
| `src/data_processing/data_processor/python/tests/test_signal_processing_core.py` | 595 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/model_generation/tests/test_unified_loader.py` | 595 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/plot_theme/themes.py` | 594 | Separate data prep from rendering; extract reusable plotting primitives |
| `src/pendulum_simulator/tests/test_jacobians_golfer.py` | 593 | Split tests by feature area; extract shared fixtures into conftest |
| `src/tools/matlab_quality_utils.py` | 591 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/core/script_generator.py` | 590 | Decompose pipeline stages into composable functions in a sub-package |
| `src/urdf_builder_gui/tests/test_urdf_builder_gui.py` | 585 | Split tests by feature area; extract shared fixtures into conftest |
| `src/ode_solver/python/ode_solver/ui/pyqt6/main_window.py` | 583 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/solar_system_model/solar_system/physics/orbital_mechanics.py` | 581 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/data_processing/data_processor/python/data_processor/high_performance_loader.py` | 580 | Decompose pipeline stages into composable functions in a sub-package |
| `src/data_processing/data_processor/python/data_processor/core/wavelet_denoising.py` | 579 | Decompose pipeline stages into composable functions in a sub-package |
| `src/solar_system_model/solar_system/visualization/scene.py` | 576 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/upstream_drift_tools/utils/unit_constants.py` | 575 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/model_generation/builders/manual_builder.py` | 572 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/plot_engine/matplotlib_renderer.py` | 568 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/vessel_drafter/python/vessel_drafter/gui/vessel_drafter_window.py` | 568 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/tests/test_friction_triple.py` | 567 | Split tests by feature area; extract shared fixtures into conftest |
| `src/asteroid_jumper/renderer.py` | 566 | Separate data prep from rendering; extract reusable plotting primitives |
| `src/shared/python/model_generation/core/physics_validation.py` | 561 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/theme/colors.py` | 561 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/benchmarks/performance_benchmark.py` | 559 | Decompose pipeline stages into composable functions in a sub-package |
| `src/shared/python/theme/theme_manager.py` | 555 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/ui/widgets/unit_converter_widget.py` | 553 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/model_generation/converters/urdf_parser.py` | 550 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/model_generation/editor/text_editor.py` | 548 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/flow_rate_converter/python/flow_rate_converter/ui/pyqt6/main_window.py` | 545 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/tests/test_native_backend.py` | 545 | Split tests by feature area; extract shared fixtures into conftest |
| `src/solar_system_model/solar_system/visualization/camera.py` | 545 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/steam_engine_calculator/tests/test_steam_engine_calculator_gui.py` | 543 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/upstream_drift_tools/process_calculators/optimization.py` | 541 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/signal_toolkit/noise.py` | 538 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/ui/pyqt6/statistical_widgets.py` | 535 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/perturbation_panel.py` | 535 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/upstream_drift_tools/utils/state_manager.py` | 535 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/ui/pyqt6/widgets.py` | 532 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/upstream_drift_tools/calculators/conversion/service.py` | 532 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/pendulum_simulator/pendulum-core/src/cmaes.rs` | 531 | Split Rust module into sub-modules by concern (math, io, ffi) |
| `src/pendulum_simulator/pendulum-core/src/golfer.rs` | 531 | Split Rust module into sub-modules by concern (math, io, ffi) |
| `src/shared/python/data_processing/processor.py` | 528 | Decompose pipeline stages into composable functions in a sub-package |
| `src/rotation_converter/screw_visualization.py` | 527 | Separate data prep from rendering; extract reusable plotting primitives |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/swing_comparison_dialog.py` | 524 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/pendulum_simulator/pendulum-core/python/physics_native.py` | 520 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/engine/_flow_calculations.py` | 520 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |
| `src/shared/python/upstream_drift_tools/tests/process_calculators/test_psa_model.py` | 519 | Split tests by feature area; extract shared fixtures into conftest |
| `src/shared/python/data_processing/tests/test_processor.py` | 518 | Split tests by feature area; extract shared fixtures into conftest |
| `src/pressure_drop_calculator/python/pressure_drop_calculator/ui/pyqt6/main_window.py` | 517 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/rotation_converter/_mr_kinematics.py` | 509 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/data_processing/data_processor/python/data_processor/ui/pyqt6/visualization_widgets.py` | 505 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/calc_backend/tests/test_calc_backend_gaps.py` | 505 | Split tests by feature area; extract shared fixtures into conftest |
| `src/rotation_converter/core.py` | 504 | Extract helper classes/functions into side modules; keep core as facade |
| `src/shared/python/theme/dialogs/theme_manager_dialog.py` | 504 | Split UI: extract tabs/panels, move business logic to controller/service |
| `src/shared/python/signal_toolkit/limits.py` | 502 | Identify natural seams (classes, sections), extract to sibling modules |
| `src/shared/python/upstream_drift_tools/calculators/electrical/electrical_model.py` | 502 | Extract pure calc kernels into shared/python helpers; keep orchestration thin |

_Total: 181 files above 500 LOC._
