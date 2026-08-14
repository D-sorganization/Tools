# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Current-state only; history lives in git. Last updated: 2026-08-14.

## Where This Tool Is Headed

Impact/ball-flight explorer with two first-class clients — a PyQt6 desktop app
(`ui/pyqt6/`) and a React web app (`web/`) — over one shared Python physics core
(`src/shared/python/swing_sim/`, `simulation/`, `variation/`, `club/`). The web app
also runs against a same-origin Python companion (`web_companion/`) with an isolated
private authority child (`web_authority/`) so browser code never executes physics.

Epics in flight: #4103 (swing→impact→flight platform), #4104 (P7 wasm-pack core —
**not** delivered yet; the React app still uses the TypeScript twins), #4142/#4433
(variation/Morris screening), #4205/#4260/#4267/#4377 (ground study and release
qualification).

## Recent Activity (grounding — `git log --oneline -20 -- src/rate_of_closure`)

The 43 open PRs across five families (club builder/impact tensor, flight/wind/wedge,
multi-view workspace, ground, web companion) were folded into one change and their
originals closed as superseded. That consolidation resolved several cross-family
defects worth knowing about, because each was invisible while the families were
separate branches:

- The PyQt and React module registries had drifted apart (10 canonical ids vs 12
  views), so no workspace file could round-trip. They are now one 12-entry set;
  the PyQt-only ground-study tab is declared unshared in
  `ui/pyqt6/main_window_file_commands.py` and is restored in place on import.
- The qualified club-assembly binding (#4111/#4341) was not wired through the
  simulation on either client; it now flows through `SimulationConfig` →
  `simulation/pipeline.py` → `SimulationRun.club_assembly_usage`, with the React
  twin matching in `web/src/model/simulation.ts`.
- Ground playback existed twice (two independent implementations). One survives,
  extended with the other's comparison overlay and event markers.

## Must-Read Architecture Pointers

- **Module registry is a cross-client contract.** `application/workspace_session.py`
  `CANONICAL_MODULE_IDS` must stay identical to `web/src/model/viewPreferences.ts`
  `PRIMARY_VIEWS`, in the same order. The React `validatedModules` guard requires
  every one of its view ids; Python rejects any set mismatch. Adding a view means
  editing both, plus `ui/pyqt6/navigation_state.py` `DEFAULT_TAB_IDS` and the
  PyQt→canonical map. A PyQt-only tab must be added to `_UNSHARED_PYQT_TAB_IDS`
  instead, or the document contract breaks.
- **The regional-ground controller owns the live variation plan** because the
  execution request port reads it. `web/src/components/PrimaryWorkspacePanel.tsx`
  bridges it into the controlled workspace API and mirrors changes back into the
  app-level snapshot, so saved workspaces record the real study.
- **Matplotlib views must tolerate zero-size layout.** The multi-view compositor can
  lay a hosted view out at zero height; `ui/pyqt6/figure_canvas.py` keeps the figure
  at a one-pixel floor so transforms stay invertible.
- **Release evidence is derived, not hand-written.** `docs/release/four_surface_capability.v1.json`
  is validated against declarations derived from `docs/release/rate_of_closure_campaign.v1.json`
  (`four_surface_declarations.py`). Editing the campaign manifest means regenerating
  the capability manifest and its inventory counts. Carriers record
  `evidence_commit_sha` — never a self-referential head sha.
- **Anything imported at module level by a collected test must be in
  `requirements.txt`**, not only a pyproject extra. `web_companion/runtime.py` and
  `web_authority/child.py` import `uvicorn`, and `web_authority/job_store.py`
  imports `filelock`; both are declared there for exactly this reason. A missing
  package is a *collection* error and fails the whole lane before any test runs.

## Gate Commands (this tool)

```bash
python3 -m pytest tests/rate_of_closure -m "not slow and not e2e"
python3 -m pytest tests/shared/python/golf_club src/shared/python/swing_sim
python3 -m ruff check src/rate_of_closure && python3 -m ruff format --check src/rate_of_closure
MYPYPATH=src python3 -m mypy --ignore-missing-imports --follow-imports=skip src/rate_of_closure
# web mirror:
cd src/rate_of_closure/web && npm ci && npm run type-check && npm run lint && npx vitest run && npm run build
```

On Windows, `MYPYPATH` needs `;` separators — a `:` makes mypy silently check nothing.
Install PyQt6 and matplotlib in whatever venv you type-check with, or mypy resolves
them to `Any` and reports a false clean.

## Do-Not List

- Do not use `datetime.UTC`. The matrix still runs Python 3.10; use `timezone.utc`
  with `# noqa: UP017` **on the same physical line**, or a `sys.version_info` guard.
- Do not claim `Closes #4104`. The P7 wasm-pack core is absent
  (`git grep -l wasm` under `web/` is empty); use `Part of`.
- Do not add a canonical module id to one client only.
- Do not run physics in browser code, or let the companion publish the authority
  bearer token or child port to browser-visible state.
- Do not reformat with a ruff other than the CI pin (0.14.10) — a newer build
  reflows files and fails Format Check.

## Roadmap (ordered)

1. Land the consolidation and let `tests (3.11)` adjudicate the merged tree.
2. Deliver the #4104 wasm-pack core so the React app stops relying on TypeScript
   twins for physics, then retire the parity twins it replaces.
3. Fix #3973 (NaN clears active alarms) — untouched by any open PR.
4. Re-enable the 162 embedded test files CI does not collect (#3975); the pendulum
   and `swing_sim` embedded suites currently gate nothing.
5. Close out ground-study qualification (#4205/#4267/#4377) with real evidence
   rather than declaration-only capability records.
