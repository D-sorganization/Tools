# Monolithic Files Audit

_Generated for issue #2152 — "Split oversized modules by responsibility"._
_Date: 2026-04-19. Fleet triage wave 25._

## Purpose

This audit identifies Python modules in `src/` that exceed the 500 LOC
"large file" threshold and proposes safe, responsibility-aligned splits.
Files above 1000 LOC are listed but explicitly **deferred** from this
wave because they typically contain domain-heavy UI, physics, or solver
logic that needs characterisation tests before extraction.

## Method

```bash
find src/ -name "*.py" -exec wc -l {} \; | sort -rn | head -20
```

Results are bucketed by risk, each bucket given a proposed split pattern
derived from common seams (data vs. behaviour, IO vs. compute, widget vs.
model, etc.).

## Top 20 Oversized Python Files

| LOC  | Path                                                                                                                         | Bucket       |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ------------ |
| 1104 | `src/pendulum_simulator/src/double_pendulum_golf/gui/equations_data.py`                                                       | defer (>1k)  |
| 1060 | `src/shared/python/model_generation/api/rest_api_routes.py`                                                                   | defer (>1k)  |
| 1055 | `src/data_processing/data_processor/python/data_processor/core/cross_correlation.py`                                          | defer (>1k)  |
| 1051 | `src/shared/python/calc_backend/tests/test_calc_backend.py`                                                                   | defer (>1k)  |
| 1045 | `src/solar_system_model/solar_system/ui/widgets.py`                                                                           | defer (>1k)  |
| 1038 | `src/shared/python/model_generation/editor/text_editor.py`                                                                    | defer (>1k)  |
| 1005 | `src/data_processing/data_processor/python/data_processor/core/state_space.py`                                                | defer (>1k)  |
| 997  | `src/data_processing/data_processor/python/data_processor/core/uncertainty_quantification.py`                                 | split-later  |
| 978  | `src/document_processing/pdf_renamer/src/pdf_renamer/gui.py`                                                                  | UI — defer   |
| 972  | `src/python/src/help/help_system.py`                                                                                          | split-later  |
| 965  | `src/solar_system_model/solar_system/visualization/renderer.py`                                                               | UI — defer   |
| 961  | `src/shared/python/signal_toolkit/widget_ui.py`                                                                               | UI — defer   |
| 947  | `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/utils/gas_properties.py`                 | split-later  |
| 939  | `src/pendulum_simulator/src/double_pendulum_golf/gui/simulation_panel.py`                                                     | UI — defer   |
| 928  | `src/shared/python/upstream_drift_tools/process_calculators/acid_gas_dewpoint_calculator.py`                                  | physics-core |
| 926  | `src/pendulum_simulator/src/double_pendulum_golf/gui/pendulum_widget.py`                                                      | UI — defer   |
| 921  | `src/solar_system_model/solar_system/data/historical_events.py`                                                               | **split ✔**   |
| 919  | `src/rrt_path_planner/python/src/star_wars_rrt.py`                                                                            | split-later  |
| 919  | `src/pendulum_simulator/src/double_pendulum_golf/gui/panel_builders.py`                                                       | UI — defer   |
| 917  | `src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py`                                                                | split-later  |

## Proposed Responsibility Splits (non-UI, non-core-physics, <1000 LOC)

The table below only includes modules where a safe seam exists between
**data / constants / tables** and **behaviour**, or between **IO** and
**pure compute**. Files involving Qt/PyQt widgets, rendering canvases,
or tightly coupled physics solvers are deferred to sprints with domain
reviewers present.

| File                                                                 | LOC  | Suggested seam                                                                                          |
| -------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------- |
| `solar_system/data/historical_events.py`                             | 921  | ✅ **done this wave** — data → `space_events_data.py`; queries stay in `historical_events.py`.            |
| `python/src/help/help_system.py`                                     | 972  | Extract the static help content table into `help_content_<topic>.py` modules behind a registry.         |
| `process_calculators/pressure_drop_calculator/utils/gas_properties.py` | 947 | Separate gas-property **tables/constants** from the **mixing-rule functions**.                          |
| `rrt_path_planner/python/src/star_wars_rrt.py`                       | 919  | Extract map/obstacle fixtures into a data module; keep the RRT algorithm class focused.                 |
| `upstream_drift_tools/lab/bio/c3d_reader.py`                         | 917  | Split file-format parsing (header/parameters/frames) into three cooperating modules behind a facade.    |
| `data_processor/core/uncertainty_quantification.py`                  | 997  | Split MC sampling, Bayesian updating, and sensitivity analysis into three files under a subpackage.     |
| `process_calculators/acid_gas_dewpoint_calculator.py`                | 928  | Extract Antoine/virial coefficient tables to a constants module before touching the solver.             |
| `help/help_content.py`                                               | 837  | Already topic-tagged; trivially splittable by topic into a `help_content/` package.                     |
| `shared/python/humanoid_character_builder/core/segment_definitions.py` | 759 | Segment *data* vs. segment *validation/lookup*.                                                         |

## Categorisation of All Files >500 LOC

- **UI / widgets / renderers (defer, need manual regression)**: ~45 files.
  Qt `main_window.py`, `panel_builders.py`, `*_widget.py`, solar-system
  and pendulum renderers/scenes. These need screenshot or interaction
  tests before splitting.
- **Core physics / solvers (defer, need domain review)**: `physics*.py`,
  `kalman_filter.py`, `state_space.py`, `cross_correlation.py`,
  `signal_processing.py`, `spectral_analysis.py`, `ball_flight.rs`,
  `pendulum-core/src/lib.rs`, `nn_trainer.py`, `feature_extractor.py`.
- **Test modules >500 LOC**: left as-is — test length rarely harms SRP.
  If split, do so by fixture grouping, not function count.
- **Data/tables/constants**: lowest risk. Good candidates for the next
  few waves; pattern is always "pure data → `_data.py`, behaviour stays".
- **Notebooks / MATLAB / TS/TSX / JS / Rust**: out of scope for this
  Python-focused audit.

## Safe Split Completed This Wave

### `solar_system/data/historical_events.py` (921 → 106 LOC)

- New module: `solar_system/data/space_events_data.py` (843 LOC, pure data).
- `historical_events.py` now contains only the query API (`get_events_for_date`,
  `get_events_by_year`, `get_events_by_category`) and re-exports
  `SPACE_EVENTS` for backward compatibility.
- No public import path changed. Existing callers
  (`ui/widgets.py`, `tests/test_parametrized.py`,
  `visualization/scene.py`, etc.) continue to work unmodified.

### Why this file, not a larger one?

- **No UI coupling**: zero Qt, zero rendering dependencies.
- **No physics coupling**: no solvers, no numerical state.
- **Clean seam**: the file is already "one big list + three small
  functions"; the split is a pure **extract-constant** refactor.
- **<1000 LOC guard-rail respected**: 921 LOC.
- **Callers verified**: `grep -r historical_events` showed all imports
  are via the public names we re-export.

## Next Waves

Recommended order for follow-up triage waves, roughly increasing in risk:

1. `help_content.py` → `help_content/` package (trivial, per-topic files).
2. `pressure_drop_calculator/utils/gas_properties.py` → constants vs. mixing-rules.
3. `star_wars_rrt.py` → pull map fixtures out of the algorithm.
4. `c3d_reader.py` → header/parameters/frames submodules.
5. `uncertainty_quantification.py` → MC / Bayes / sensitivity.

Each of these should follow the same pattern used here: extract the
lowest-risk payload, keep the original import path working via
re-export, verify ruff + mypy + at least a smoke import test.

## Acceptance Criteria (issue #2152)

- [x] Audit of oversized modules produced (this document).
- [x] Proposed splits per module.
- [x] One safe split performed (`historical_events.py`).
- [x] Ruff + mypy clean on affected files.
- [x] No behaviour change; no public import removed.
