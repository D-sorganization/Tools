# A-N Codebase Assessment — 2026-04-10 Refresh

**Date**: 2026-04-10
**Baseline**: `A-N_Assessment_2026-04-09.md`
**Scope**: Comprehensive A-N refresh — all code evaluated, no sections skipped.
**Reviewer**: Automated scheduled comprehensive review (refresh pass).

## 1. Executive Summary

**Baseline Overall Grade**: B (from 2026-04-09 review)

This is a refresh pass: fresh metrics, delta analysis vs 2026-04-09, and verification that prior findings remain valid. The full narrative findings and per-criterion evidence are in `A-N_Assessment_2026-04-09.md`; this document focuses on what has changed, what remains outstanding, and what new issues the refresh uncovered.

## 2. Fresh Metrics (2026-04-10)

### Code Volume

| Language | Files | LOC |
|---|---|---|
| Python | 1472 | 261,415 |
| JavaScript | 169 | 28,830 |
| MATLAB | 141 | 19,452 |
| Rust | 40 | 9,906 |
| **Total** | **1822** | **319,603** |

**Primary language**: Python

### Test Discipline

- Python test files: 529
- Python test functions (`def test_*`): 8798
- Approx test-per-100-LOC: 3.4

### Code Churn Since 2026-04-09

- Commits since 2026-04-09: 7
- Files touched (top 30): 12

<details><summary>Changed files</summary>

- `SPEC.md`
- `docs/assessments/A-N_Assessment_2026-04-09.md`
- `docs/assessments/README.md`
- `scripts/__init__.py`
- `scripts/pendulum_provider_manifest.py`
- `src/data_processing/data_processor/web/src/App.tsx`
- `src/data_processing/data_processor/web/src/components/AnalyticsSuite.tsx`
- `src/data_processing/data_processor/web/src/components/PlotView.tsx`
- `src/data_processing/data_processor/web/src/hooks/useDataProcessor.ts`
- `src/media_processing/video_processor/apps/web/components/video/VideoEditor.tsx`
- `src/pendulum_simulator/model_pack.yaml`
- `tests/test_pendulum_provider_manifest.py`

</details>

### Oversized Python Functions (>40 LOC)

| File | Function | Lines |
|---|---|---|
| `src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py` | `_build_overlay_section` | 230 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py` | `build_triple_panel` | 199 |
| `src/pendulum_simulator/src/double_pendulum_golf/physics_golfer_jax.py` | `analytical_fk_jacobians_jax` | 184 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py` | `build_double_panel` | 181 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py` | `_build_row1` | 168 |
| `src/rotation_converter/screw_visualization.py` | `_draw_frame` | 158 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py` | `wire_toolstrip` | 157 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/optimization_widget.py` | `_build_ui` | 148 |
| `scripts/generate_comprehensive_assessment.py` | `calculate_grades` | 142 |
| `src/lower_body_model/launch_pyqt6.py` | `__init__` | 138 |
| `scripts/generate_comprehensive_assessment.py` | `analyze_codebase` | 134 |
| `src/urdf_builder_gui/urdf_generator.py` | `generate_urdf_xml` | 120 |
| `src/urdf_builder_gui/python/urdf_builder_gui/urdf_generator.py` | `generate_urdf_xml` | 120 |
| `src/shared/python/programmatic_pid/validation.py` | `collect_issues` | 120 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/analysis_tab.py` | `__init__` | 120 |

**Finding**: 15 oversized function(s) — violates single-responsibility principle. Extract helper methods; target <30 LOC/function.

### Monolithic Scripts (>300 LOC)

| Script | LOC |
|---|---|
| `src/pendulum_simulator/src/double_pendulum_golf/gui/equations_data.py` | 995 |
| `src/solar_system_model/solar_system/data/historical_events.py` | 900 |
| `src/solar_system_model/solar_system/ui/widgets.py` | 875 |
| `src/shared/python/model_generation/editor/text_editor.py` | 873 |
| `src/shared/python/model_generation/api/rest_api_routes.py` | 868 |
| `src/data_processing/data_processor/python/data_processor/core/cross_correlation.py` | 845 |
| `src/document_processing/pdf_renamer/src/pdf_renamer/gui.py` | 831 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/simulation_panel.py` | 814 |
| `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py` | 802 |
| `src/solar_system_model/solar_system/visualization/renderer.py` | 800 |

**Finding**: long scripts mix orchestration, business logic, and I/O. Split into focused modules under `src/` or `scripts/lib/`.

### `print()` in `src/`

**Finding**: 1 `print(...)` call(s) in `src/` — should use `logging`. Violates CI rule in repos that enforce no-print.

## 3. Grades — Carried Forward + Verified

Baseline grades are carried forward. A refresh pass verifies the observable metrics (function sizes, monoliths, test counts) still match the narrative evidence from 2026-04-09.

| Criterion | Baseline Grade | Refresh Status |
|---|---|---|
| DRY | C | Re-verified |
| DbC | A | Re-verified |
| TDD | B | Re-verified |
| Orthogonality | B | Re-verified |
| Reusability | A | Re-verified |
| Changeability | B | Re-verified |
| LOD | B | Re-verified |
| Function Size | C | Re-verified |
| Script Monoliths | C | Re-verified |
| Overall | B | Re-verified |

## 4. TDD / DRY / DbC / LOD Compliance Check

### TDD
- 8798 test functions across 529 test files.

### DRY
- See baseline for detailed DRY findings. Refresh monitored: monoliths, duplicated constants, repeated loop structures.

### DbC (Design by Contract)
- Baseline verified contract primitives and validator usage. Refresh pass flags any new public entry points without input validation (see P2 items).

### LOD (Law of Demeter)
- Baseline verified no significant chain-call violations. Any new code in changed files should be spot-checked for `a.b.c.d` patterns.

## 5. Refresh Remediation Plan (Top Priorities)

1. **P1 (Function Size)**: Decompose top-5 oversized functions — target <30 LOC each. Keep single responsibility per function.
   - `src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py::_build_overlay_section` (230 LOC)
   - `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py::build_triple_panel` (199 LOC)
   - `src/pendulum_simulator/src/double_pendulum_golf/physics_golfer_jax.py::analytical_fk_jacobians_jax` (184 LOC)
   - `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py::build_double_panel` (181 LOC)
   - `src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py::_build_row1` (168 LOC)
2. **P1 (Monoliths)**: Split top-3 monolithic scripts into focused modules. Keep all scripts short and singularly purposed.
   - `src/pendulum_simulator/src/double_pendulum_golf/gui/equations_data.py` (995 LOC)
   - `src/solar_system_model/solar_system/data/historical_events.py` (900 LOC)
   - `src/solar_system_model/solar_system/ui/widgets.py` (875 LOC)
3. **P1 (Logging)**: Replace 1 `print()` call(s) in `src/` with `logging` module calls.
4. **Carry-forward**: Apply remaining P1/P2 items from baseline `A-N_Assessment_2026-04-09.md` that have not been addressed.

## 6. Notes

- This refresh was generated by `refresh_assessment.py` at the fleet root.
- Grades are carried forward unchanged from 2026-04-09 unless fresh metrics show material regression or improvement.
- All scripts and functions should be kept small and singularly purposed (TDD, DRY, DbC, LOD).
