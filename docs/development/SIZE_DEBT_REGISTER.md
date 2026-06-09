<!--
  GENERATED FILE - do not edit by hand.
  Regenerate with:
    python scripts/generate_size_debt_register.py --write
  Source of truth for issue #3261 (retire monoliths / structural debt).
-->

# Size debt register

Tracks every source file under `src/` at or above **800 lines**. The
repository standard caps files at 400 lines; files at/above 800 lines are
structural debt and files at/above 1000 lines are **CRITICAL**.

This register is the ranked work queue for issue #3261. Refactor one file per
PR (responsibility-preserving extraction, behaviour pinned by characterization
tests), then regenerate this file so the count ratchets down. The register is
intentionally non-blocking: it informs prioritisation, it does not gate CI.


- Files at/above 800 LOC: **71**
- CRITICAL (at/above 1000 LOC): **21**

| Rank | LOC | Class | File |
| ---- | --- | ----- | ---- |
| 1 | 3521 | CRITICAL | `src/media_processing/audio_processor/matlab/audio_signal_processor/gui/MainWindow.m` |
| 2 | 2219 | CRITICAL | `src/p1am_control_system/frontend/src/App.tsx` |
| 3 | 2089 | CRITICAL | `src/rotation_converter/modern_robotics.py` |
| 4 | 1387 | CRITICAL | `src/shared/python/chat/_chat_dock_widget_qt.py` |
| 5 | 1374 | CRITICAL | `src/p1am_control_system/backend/main.py` |
| 6 | 1296 | CRITICAL | `src/rotation_converter/ui/pyqt6/main_window.py` |
| 7 | 1264 | CRITICAL | `src/media_processing/audio_processor/matlab/audio_signal_processor/core/MusicProductionTools.m` |
| 8 | 1140 | CRITICAL | `src/web_applications/unit_converter/unit-converter-app/app.js` |
| 9 | 1133 | CRITICAL | `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py` |
| 10 | 1105 | CRITICAL | `src/pendulum_simulator/src/double_pendulum_golf/gui/equations_data.py` |
| 11 | 1091 | CRITICAL | `src/shared/python/calc_backend/tests/test_calc_backend.py` |
| 12 | 1088 | CRITICAL | `src/pendulum_simulator/src/double_pendulum_golf/gui/base_pendulum_widget.py` |
| 13 | 1073 | CRITICAL | `src/data_processing/data_processor/web/src/hooks/useDataProcessor.ts` |
| 14 | 1058 | CRITICAL | `src/function_generator/web/src/components/FunctionGenerator.tsx` |
| 15 | 1055 | CRITICAL | `src/shared/python/sidekick/process_calculators/psa_package/psa_gui.py` |
| 16 | 1048 | CRITICAL | `src/web_applications/unit_converter/unit-converter-app/converter.js` |
| 17 | 1043 | CRITICAL | `src/solar_system_model/solar_system/ui/widgets.py` |
| 18 | 1017 | CRITICAL | `src/shared/python/model_generation/api/rest_api_routes.py` |
| 19 | 1013 | CRITICAL | `src/data_processing/data_processor/python/data_processor/core/cross_correlation.py` |
| 20 | 1011 | CRITICAL | `src/media_processing/audio_processor/matlab/audio_signal_processor/core/AdvancedAudioProcessor.m` |
| 21 | 1003 | CRITICAL | `src/data_processing/data_processor/python/data_processor/core/state_space.py` |
| 22 | 996 | HIGH | `src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py` |
| 23 | 995 | HIGH | `src/data_processing/data_processor/python/data_processor/core/uncertainty_quantification.py` |
| 24 | 986 | HIGH | `src/data_processing/data_processor/web/src/components/AnalyticsSuite.tsx` |
| 25 | 983 | HIGH | `src/pendulum_simulator/src/double_pendulum_golf/gui/pendulum_widget.py` |
| 26 | 980 | HIGH | `src/document_processing/pdf_renamer/src/pdf_renamer/gui.py` |
| 27 | 976 | HIGH | `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/utils/gas_properties.py` |
| 28 | 975 | HIGH | `src/python/src/help/help_system.py` |
| 29 | 969 | HIGH | `src/web_applications/calculator/calculator.py` |
| 30 | 966 | HIGH | `src/solar_system_model/solar_system/visualization/renderer.py` |
| 31 | 963 | HIGH | `src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py` |
| 32 | 960 | HIGH | `src/shared/python/signal_toolkit/widget_ui.py` |
| 33 | 958 | HIGH | `src/shared/python/sidekick/calculators/thermo/steam_engine.py` |
| 34 | 939 | HIGH | `src/media_processing/audio_processor/matlab/audio_signal_processor/core/ConvolutionReverb.m` |
| 35 | 939 | HIGH | `src/shared/python/sidekick/lab/bio/c3d_reader.py` |
| 36 | 937 | HIGH | `src/pendulum_simulator/src/double_pendulum_golf/gui/simulation_panel.py` |
| 37 | 927 | HIGH | `src/rrt_path_planner/matlab/src/gui/starWarsPathPlannerGUI.m` |
| 38 | 918 | HIGH | `src/rrt_path_planner/python/src/star_wars_rrt.py` |
| 39 | 912 | HIGH | `src/python/src/utils/test_utils.py` |
| 40 | 911 | HIGH | `src/pendulum_simulator/src/double_pendulum_golf/physics_golfer_jax.py` |
| 41 | 901 | HIGH | `src/shared/python/model_generation/library/model_library.py` |
| 42 | 900 | HIGH | `src/media_processing/audio_processor/matlab/audio_signal_processor/core/MixerCoreEnhanced.m` |
| 43 | 885 | HIGH | `src/data_processing/data_processor/python/tests/test_statistical_analysis.py` |
| 44 | 884 | HIGH | `src/shared/python/sidekick/process_calculators/syngas_compression_calculator.py` |
| 45 | 883 | HIGH | `src/data_processing/data_processor/python/data_processor/core/kalman_filter.py` |
| 46 | 878 | HIGH | `src/shared/python/programmatic_pid/cli.py` |
| 47 | 875 | HIGH | `src/optimizer_gui/python/optimizer_gui/ui/pyqt6/main_window.py` |
| 48 | 870 | HIGH | `src/data_processing/data_processor/python/tests/test_advanced_analysis.py` |
| 49 | 869 | HIGH | `src/python/src/utils/integration_test_helpers.py` |
| 50 | 863 | HIGH | `src/shared/python/sidekick/ui/tools_sidebar/os_terminal.py` |
| 51 | 853 | HIGH | `src/shared/python/sidekick/process_calculators/scrubber_calculator.py` |
| 52 | 850 | HIGH | `src/shared/python/model_generation/cli/main.py` |
| 53 | 844 | HIGH | `src/solar_system_model/solar_system/data/space_events_data.py` |
| 54 | 843 | HIGH | `src/pendulum_simulator/src/double_pendulum_golf/gui/optimization_widget.py` |
| 55 | 841 | HIGH | `src/shared/python/sidekick/ui/mixins/calculator_state_mixin.py` |
| 56 | 840 | HIGH | `src/humanoid_builder_gui/python/humanoid_builder_gui/ui/pyqt6/main_window.py` |
| 57 | 840 | HIGH | `src/pendulum_simulator/src/double_pendulum_golf/gui/analysis_tab.py` |
| 58 | 840 | HIGH | `src/shared/python/ai/gui/assistant_panel.py` |
| 59 | 840 | HIGH | `src/shared/python/signal_toolkit/fitting.py` |
| 60 | 837 | HIGH | `src/data_processing/data_processor/python/data_processor/core/nn_trainer.py` |
| 61 | 835 | HIGH | `src/python/src/help/help_content.py` |
| 62 | 831 | HIGH | `src/shared/python/signal_toolkit/widget_processing.py` |
| 63 | 828 | HIGH | `src/shared/python/sidekick/process_calculators/pressure_drop_calculator/pressure_drop_interface.py` |
| 64 | 823 | HIGH | `src/data_processing/data_processor/python/data_processor/core/signal_processing.py` |
| 65 | 822 | HIGH | `src/media_processing/audio_processor/matlab/audio_signal_processor/core/AudioEditor.m` |
| 66 | 820 | HIGH | `src/pendulum_simulator/src/double_pendulum_golf/gui/golfer_pendulum_widget.py` |
| 67 | 820 | HIGH | `src/shared/python/model_generation/converters/simscape/simscape_converter.py` |
| 68 | 818 | HIGH | `src/shared/python/sidekick/process_calculators/constants.py` |
| 69 | 814 | HIGH | `src/solar_system_model/solar_system/visualization/ui_renderer.py` |
| 70 | 811 | HIGH | `src/shared/python/humanoid_character_builder/mesh/mesh_processor.py` |
| 71 | 802 | HIGH | `src/data_processing/data_processor/python/data_processor/core/spectral_analysis.py` |
