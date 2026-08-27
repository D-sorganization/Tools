# AGENT_HANDOFF — Rate_of_Closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-27
> **Current state only**, capped at 150 lines; history lives in git and in [`docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md`](../../docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md).
> Do not append dated entries — that is how it reached 2,205 lines.

## What This Tool Is Now

A swing → impact → ball-flight simulator with parity **PyQt6 and React
surfaces**, plus a Rust fast path; it began as a closure-rate calculator.

PyQt6 entry: `rate_of_closure.ui.pyqt6.main_window:RateOfClosureMainWindow`,
titled "Rate of Closure Impact Explorer" as UpstreamDrift's manifest advertises.
React mirror: `web/` (Vite/TS, ~1,570 tests across 193 files).

Physics lives in `src/shared/python/swing_sim/` — outside this tool, so
UpstreamDrift reaches it through `vendor/ud-tools`: `impact/` (contact law,
gear effect; #4130 extends `SpringDamperImpactModel` rather than duplicating
it), `flight/`, `ground/`, `variation/`, `solver/`. Club physics for the
fitting epics is the sibling `shared/python/golf_club/`.

## Delivery Pattern

PR #4466 predates this package and cannot merge. Twenty-two slices landed as
#4517–#4547; #4571 owns the remaining camera reimplementation. Diff every
candidate against current main and preserve main-owned tests.

## Active Epics — Golf Epics Merged

Launch-monitor epic #4583 has merged Release A with explicit-identity projects,
bounded private-authority loading, canonical dataset/covariation clients, and
source-backed expected-strokes/longitudinal analysis. No private rows or paths
enter project files and no baseline data is bundled. Release B remains open:
do not claim vendor emulation or paired-device validation without real paired
observations.

#4584/#4599 merged source-backed strokes-gained v2 into both clients with exact
strata, uncertainty, exclusions, and grouping attestations. #4600 owns the
post-merge PyQt reference; #4602/#4608/#4610/#4613 own isolated rendered gates.

Club Fitting #4549, Heavy Hit #4562, and packaging #4579 are complete and
merged. Their physics lives shared-first in `shared/python/{golf_club,swing_sim}`;
see the two contracts under `docs/specs/`.

#4142 remains Python-authoritative; PyQt6 and React do not reimplement physics.
R10.3, R10.4, R11.1, and R11.3 are protected-merged through
`4ddec9175814451fdc3d1a94b45f1190e7503bca`. The complete-trial authority and
`swing-trace-time-linear-contiguous/v1` preserve stable trial/point/frame
identity, missing intervals, outcomes, failure semantics, and approximate
impact-marker error. The inherited 3-source by 4-adapter matrix has two verified
double-pendulum cells and ten explicitly unavailable cells.

R12.3 protected-squash-merged via PR #4782 as `a1b00db14`. R13.3 PR #4784
is held at exact head `100a50e58`: required checks pass, but its optional Rust
and fleet gates cannot run while OGLaptop's WSL VHDX is corrupt. Do not create
redundant reruns or merge through the evidence gap. #4791/R13.5 is active from
current `main`. `morris-target-selection` v1 binds kind/name/unit/point/time/
frame and exposes all-input or one-source views without analysis execution;
PyQt and React share a state/impact/shot parity fixture. Selected rows retain
global ranks and typed denominators. The branch ledger is 28 verified / 3
partial before R13.3 reconciliation. These remain model-scenario screening
views, not causal anatomy, human validation, or coaching authority.

PR #4705 maps all 31 #4433 obligations; trusted run `32689177846` proves only
the initial React/PyQt visibility, accessibility, performance, and baseline
tier. PR #4733 merged V0.1 with purpose, prerequisites, and reciprocal
counterparts; PR #4736 merged strict TypeScript-reader parity as `34a809d9`.
PR #4738 merged V5.2's fail-closed changed-path governance as `4b4aec421`.
The audit is 8 verified / 23 partial; seven blockers and two human actions remain.

### Adding a Tab: The Four-Manifest Lockstep (Read Before Starting C6/C7/H4)

A new tab is **not** just a widget. Four packaged manifests declare the tab set,
cross-checked by **order-strict tuple equality** on `(surface, tab_id)` against
`visualization_tabs.v1.json`. Adding to three of four — or to all four in a
different order — fails with a message that does not name the offending file.

All four live in `src/rate_of_closure/`: `visualization_tabs.v1.json` is the
authority (20 entries = 10 `pyqt` + 10 `react`); `visualization_accessibility.v1.json`,
`visualization_performance.v1.json` and `visual_baselines.v1.json` must match it
entry-for-entry, in order.

- Surface strings are **`pyqt` and `react`** — _not_ `pyqt6`, despite the
  package being `ui/pyqt6`.
- PyQt6 registration: build the widget in `ui/pyqt6/main_window.py` (~line 127,
  beside `self._plots_tab = PlotsTab()`), then add a `PrimaryTabSpec`
  `(module_id, widget, label)`; `create_primary_tabs` in `main_window_layout.py`
  stores `module_id` via `setTabData`, and it must equal the manifest `tab_id`.
- `visualization_tabs` demands a `primary_visual_locator` that resolves at
  runtime — `pyqt_visualization_tab_probe.py` drives the real widget, so a
  locator for a not-yet-rendered canvas fails there.
- Gates (all in `tests/rate_of_closure/`): `test_visualization_tab_manifest.py`,
  `test_visualization_accessibility.py`, `test_visualization_performance_manifest.py`,
  `test_visual_baseline_compare.py`, `test_pyqt_visualization_tab_visibility.py`.

## Must-Read Architecture Pointers

1. `src/rate_of_closure/README.md` — frame and unit conventions, run/build.
2. `src/shared/python/swing_sim/impact/` — the contact-force law #4130 extends.
3. `web/src/model/__fixtures__/` — golden fixtures pinning Python↔TS parity.
   Changing one is a contract change on **both** sides; land together.
4. `rust_core/swing-core/` — pendulum EOM + plane projection, pyo3 + wasm.
5. `src/shared/python/golf_club/AGENT_HANDOFF.md` — fitting/heavy-hit physics.

## Gate Commands (This Tool)

```bash
python3 -m pytest tests/rate_of_closure src/shared/python/swing_sim -n auto --timeout=300
cd src/rate_of_closure/web && npm run test && npm run build && npx tsc --noEmit && npx eslint .
cargo test -p swing-core
python3 -m ruff check src/rate_of_closure src/shared/python/swing_sim
```

## Do-Not List

- **Do not append a dated entry to this file.** See the header.
- **Do not merge #4466, #4446 or #4447 with a strategy flag.** See above.
- **Do not take a branch file wholesale without diffing it against `main`.**
  Symbol comparison misses function-body and data-only changes. Read the diff.
- Do not exceed 500 LOC per file in `rate_of_closure`, `swing_sim` or
  `swing-core` — split along a real seam, as `ui/pyqt6`'s mixins do.
- Do not eagerly import `assembly_binding`, `engineering_sidecar` or
  `simulation_adapter` from `club/__init__.py`, nor `shared.python.golf_club`
  at module scope — both reach SciPy via `swing_sim.variation → solver →
flight`, breaking the Morris UI import contract. Use the lazy-export map.

## Known Local-Environment Traps

- **Reproduce CI's mypy exactly.** Pass every changed file to one Python 3.12
  invocation with `MYPYPATH=src:src/python/src`; tests are excluded.
  `MYPYPATH='src;src/python/src' py -3.12 -m mypy --ignore-missing-imports --follow-imports=skip <changed non-test files>`
- **`tools_core` capability is two-tier** — a wheel can expose
  `simulate_trajectory` yet lack the tee-aware full-state API; guard on the
  specific capability. `test_club_view_camera.py`'s cadence test asserts
  wall-clock time and is flaky under load.
- PowerShell `Set-Content` and `pathlib.write_text` can rewrite LF as CRLF;
  preserve newlines explicitly.
- **`detect_secrets scan` writes native separators** — on Windows run it
  _before_ normalising the baseline to forward slashes, never after
  (`tests/ops/test_detect_secrets_baseline.py` rejects backslash keys).
- **A purely additive file can still be wrong.** `PrimaryViewTabs.test.tsx` and
  `TorqueProfilePanel.test.tsx` add lines, delete none, and both fail against
  `main`'s components. Never delete an `origin/main` file to pass a check.
