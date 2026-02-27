# Tools Repository Inventory (Current Reality)

- Date: 2026-02-27
- Source of truth: tool_surface_contract.json, launcher files, and repository structure
- Total manifest tools: 29
- Missing tool README files: 0
- Web surface gaps (manifest says web, implementation missing): 0
- Note: test counts below are based on filename matches under tests/

## Tool Surface Matrix

| Tool ID | Category | PyQt6 | Web (Manifest) | Web (Implemented) | Tests (name match) | README |
| --- | --- | --- | --- | --- | --- | --- |
| acid_gas_dewpoint | Process Simulation | Yes | Yes | Yes | 2 | Yes |
| baghouse_calculator | Process Simulation | Yes | Yes | Yes | 0 | Yes |
| c3d_viewer | Biomechanics | Yes | No | N/A | 0 | Yes |
| data_processor | Data Processing | Yes | Yes | Yes | 6 | Yes |
| electrode_advisor | Process Simulation | Yes | Yes | Yes | 0 | Yes |
| financial_calculator | Process Simulation | Yes | Yes | Yes | 2 | Yes |
| flare_calculator | Process Simulation | Yes | Yes | Yes | 0 | Yes |
| flow_rate_converter | Utilities | Yes | No | Yes (untracked) | 2 | Yes |
| folder_packer_pro | Development Tools | Yes | No | N/A | 0 | Yes |
| folder_tool | Development Tools | Yes | No | N/A | 2 | Yes |
| function_generator | Signal Processing | Yes | Yes | Yes | 0 | Yes |
| humanoid_builder_gui | Robotics | Yes | No | N/A | 0 | Yes |
| inertia_calculator | Robotics | Yes | No | N/A | 0 | Yes |
| multi_param_analysis | Analysis | Yes | No | N/A | 2 | Yes |
| ode_solver | Mathematics | Yes | No | Yes (untracked) | 2 | Yes |
| optimizer_gui | Optimization | Yes | No | N/A | 0 | Yes |
| pdf_renamer | Development Tools | Yes | No | N/A | 0 | Yes |
| pressure_drop_calculator | Process Simulation | Yes | Yes | Yes | 0 | Yes |
| psa_package | Process Simulation | Yes | No | Yes (untracked) | 0 | Yes |
| rotation_converter | Robotics | Yes | Yes | Yes | 4 | Yes |
| scrubber_calculator | Process Simulation | Yes | Yes | Yes | 0 | Yes |
| signal_processing_studio | Signal Processing | Yes | No | N/A | 0 | Yes |
| steam_engine_calculator | Thermodynamics | Yes | No | Yes (untracked) | 0 | Yes |
| syngas_compression | Process Simulation | Yes | Yes | Yes | 2 | Yes |
| syngas_water_calculator | Process Simulation | Yes | No | Yes (untracked) | 2 | Yes |
| thermal_profile_predictor | Process Simulation | Yes | No | Yes (untracked) | 0 | Yes |
| trc_vessel_designer | Process Simulation | Yes | Yes | Yes | 0 | Yes |
| urdf_builder_gui | Robotics | Yes | No | N/A | 0 | Yes |
| wgs_reactor | Process Simulation | Yes | Yes | Yes | 0 | Yes |

## Repository-Wide Functional Components

- Launchers: UnifiedToolsLauncher.py, Launcher.py, launch.py, and plugin registry under src/python/src/core/
- Shared libraries: src/shared/python/* (theme, signal toolkit, model generation, humanoid, launcher, thermo/conversion helpers)
- Web applications: src/web_applications/* (calculator, unit converter, urdf viewer)
- Domain stacks: src/document_processing/*, src/media_processing/*, src/data_processing/*
- Verification utilities: src/verification/*
- Automation and governance workflows: .github/workflows/*

## Implementation Gap Summary

1. README coverage gaps identified: 0
2. Web implementation gaps identified: 0
3. Tools with no name-matched tests: 19
4. Workflow tracking document updated in this change set.
