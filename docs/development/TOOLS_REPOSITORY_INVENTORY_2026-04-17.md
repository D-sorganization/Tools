# Tools Repository Inventory

- Date: 2026-04-17
- Scope: Issue #2091 narrow manifest/layout slice
- Source of truth: `gui_registration.py`, `launch_pyqt6.py`, `launch_web.py`, `launch_gui.py`, `tools.json`, and `tool_surface_contract.json`
- Manifest contract tools after this change: 21
- Launcher-backed tool directories after this change: 21
- Launcher-backed orphan directories after this change: 0

## Registered Launcher-Backed Tools

| Tool ID | Directory | Category | PyQt6 | Web | Legacy GUI |
| --- | --- | --- | --- | --- | --- |
| `c3d_viewer` | `src/c3d_viewer` | Biomechanics | Yes | No | No |
| `data_processor` | `src/data_processing/data_processor` | Data Processing | Yes | Yes | No |
| `financial_calculator` | `src/financial_calculator` | Process Simulation | Yes | Yes | No |
| `flow_rate_converter` | `src/flow_rate_converter` | Utilities | Yes | No | No |
| `folder_packer_pro` | `src/folder_packer_pro` | Development Tools | Yes | No | No |
| `folder_tool` | `src/folder_tool` | Development Tools | Yes | No | No |
| `function_generator` | `src/function_generator` | Signal Processing | Yes | Yes | No |
| `humanoid_builder_gui` | `src/humanoid_builder_gui` | Robotics | Yes | No | No |
| `inertia_calculator` | `src/inertia_calculator` | Robotics | Yes | No | No |
| `lower_body_model` | `src/lower_body_model` | Biomechanics | Yes | No | No |
| `multi_param_analysis` | `src/multi_param_analysis` | Analysis | Yes | No | No |
| `ode_solver` | `src/ode_solver` | Mathematics | Yes | No | No |
| `optimizer_gui` | `src/optimizer_gui` | Optimization | Yes | No | No |
| `pdf_renamer` | `src/document_processing/pdf_renamer` | Development Tools | Yes | No | No |
| `pid_generator` | `src/pid_generator` | Engineering Drafting | Yes | No | No |
| `pressure_drop_calculator` | `src/pressure_drop_calculator` | Process Simulation | Yes | Yes | No |
| `rotation_converter` | `src/rotation_converter` | Robotics | Yes | Yes | No |
| `signal_processing_studio` | `src/signal_processing_studio` | Signal Processing | Yes | No | No |
| `steam_engine_calculator` | `src/steam_engine_calculator` | Thermodynamics | Yes | No | No |
| `urdf_builder_gui` | `src/urdf_builder_gui` | Robotics | Yes | No | No |
| `vessel_drafter` | `src/vessel_drafter` | Engineering Drafting | Yes | No | No |

## Decisions In This Slice

| Path | Decision | Reason |
| --- | --- | --- |
| `src/lower_body_model` | Registered in manifests | It already exposes `launch_pyqt6.py`, has focused tests, and was the only launcher-backed tool directory without `gui_registration.py`. |
| `scripts/check_tools_manifest_layout.py` | Added CI guard | New launcher-backed tools now fail CI until they add a sibling `gui_registration.py`, preventing the same manifest blind spot from returning. |

## Deferred Layout Inventory

The following top-level `src/` directories remain outside this narrow launcher-backed manifest contract. They are either grouping directories, shared libraries, embedded apps, or candidate future tool registrations that need separate owner review before deletion or layout migration:

| Path | Current role | Follow-up decision needed |
| --- | --- | --- |
| `src/asteroid_jumper` | Importable/game-like package with tests, no launcher script | Decide whether to add a supported launcher or document it as library/demo code. |
| `src/data_processing` | Wrapper namespace for `data_processor` | Keep as grouping namespace unless more tools are added. |
| `src/document_processing` | Wrapper namespace for `pdf_renamer` | Keep as grouping namespace unless more document tools are added. |
| `src/folder_tool_pro` | Directory with README but no launcher-backed manifest surface | Decide whether it is dead code or should be registered. |
| `src/matlab` | MATLAB examples/tests | Keep out of Python launcher manifests. |
| `src/media_processing` | Media-processing namespace and web/video assets | Split into explicit tool registrations in a separate migration. |
| `src/pendulum_simulator` | Custom nested provider/package layout | Keep deferred because it already has a provider manifest and needs coordinated layout migration. |
| `src/project_packer` | Tests/README without launcher-backed manifest surface | Decide whether to add a launcher or retire. |
| `src/python` | Legacy shared launcher/utilities package | Keep out of tool manifests. |
| `src/rrt_path_planner` | Mixed Python/MATLAB path planner assets | Decide whether to register a launcher or keep as examples. |
| `src/shared` | Shared libraries and themes | Keep out of tool manifests. |
| `src/solar_system_model` | Package/tests without current launcher-backed manifest surface | Decide whether to add a supported launcher or document as demo code. |
| `src/tools` | Internal tooling/utilities | Keep out of user-facing launcher manifests. |
| `src/verification` | Verification support code | Keep out of tool manifests. |
| `src/web_applications` | Standalone web app grouping namespace | Split into explicit web registrations in a separate migration. |

This PR uses `Refs #2091` because the broader issue also requests canonical `src/<tool>/python/<tool>/` migrations and dead-code deletion decisions across these deferred directories.
