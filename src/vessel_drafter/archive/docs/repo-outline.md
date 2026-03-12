# Repo Outline

`Programmatic-Drafting` is organized around stable geometry inputs and
reproducible export paths.

## Top-level structure

- `src/programmatic_drafting/models`
  - Typed defaults mirrored from application UIs and drafting assumptions.
- `src/programmatic_drafting/projects`
  - Per-project builders and reusable profile helpers that assemble solids for
    export.
- `src/programmatic_drafting/exporters`
  - STEP and manifest export helpers.
- `src/programmatic_drafting/preview`
  - Reusable preview models for GUI and lightweight inspection tooling.
- `src/programmatic_drafting/gui`
  - PyQt6 vessel-drafter UI shells, preview renderers, and port-editing
    widgets over validated geometry models.
- `generated`
  - Checked-in reference exports for inspection and regression review.
- `tests`
  - Layout math and export smoke tests.

## How to add a new drawing project

1. Add a typed source-of-truth model under `models/`.
2. Add a project builder under `projects/`.
3. Export a checked-in reference STEP file under `generated/`.
4. If the project is interactive, expose a preview or editor under `gui/`.
5. Add tests covering default geometry and artifact generation.
6. Add a short project note under `docs/projects/`.

## Intended scope

- Layout validation
- Reproducible STEP exports
- Mechanical envelope and placement studies
- Interactive drafting previews and parameter editing
- FreeCAD inspection workflows

## Out of scope for now

- Meshing or FEA pipelines
- Parametric assemblies driven by external databases
