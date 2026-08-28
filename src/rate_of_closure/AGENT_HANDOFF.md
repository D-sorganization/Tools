# AGENT_HANDOFF — Rate_of_Closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-28
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

## Active Epics — Golf Epics Merged

Launch-monitor #4583 Release A merged (explicit-identity projects, bounded
private-authority loading, source-backed expected-strokes). No private rows or
baseline data are bundled. Release B open — **do not claim vendor emulation or
paired-device validation without real paired observations.** #4584/#4599 merged
strokes-gained v2 into both clients; #4600 owns the PyQt reference,
#4602/#4608/#4610/#4613 the isolated rendered gates.

Club Fitting #4549, Heavy Hit #4562, and packaging #4579 are complete; physics
is shared-first in `shared/python/{golf_club,swing_sim}`. Putting #4800 has
P1-P5, P7, and P9; `putting_result/2` supersedes v1 without silent migration,
and Python remains the sole Monte-Carlo authority — the React twins mirror the
outcome vocabulary and the deterministic single-putt evaluation only, never a
second sampler. The React Putting tab runs one chokepoint
(`evaluatePuttWithTrajectory`) and presents that record. P6 (Qt tab) and P8
(shared 3D playback) remain; putt playback waits on P8's
`playbackTransport.ts`/`TimedSample` seam and must not fork it.
Clubhead-realism #4799 has G1-G3 and G5: leading-edge lean, offset hosels, real blade soles,
and toe-view acceptance gates over the **public** `parametric_head_mesh` (pinned per-club
tables + a center-pivot counterfactual that reddens all 16 clubs if the lean is reverted).
All 16 cross-runtime club gates are protected. G4 remains.

#4142 remains Python-authoritative; PyQt6 and React do not reimplement physics.
R10.3, R10.4, R11.1, and R11.3 are protected-merged through
`4ddec9175814451fdc3d1a94b45f1190e7503bca`. The complete-trial authority and
`swing-trace-time-linear-contiguous/v1` preserve stable trial/point/frame
identity, missing intervals, outcomes, failure semantics, and approximate
impact-marker error. The inherited 3-source by 4-adapter matrix has two verified
double-pendulum cells and ten explicitly unavailable cells.

R12.3 protected-squash-merged via PR #4782 as `a1b00db14`; R13.3 PR #4784
protected-squash-merged as `d6c8a0a67`. Its paired-attribution schema binds one
independently estimable source and optional exact locus to state, impact, and
shot scalars with ten unsupported cells preserved. R13.5 protected-squash-
merged through PR #4794 as `35853199b`; `morris-target-selection` v1 binds
kind/name/unit/point/time/frame and exact all-input or selected-source views.
R14.3 issue #4792 is replayed from that protected base. Its
governed matrix proves PyQt/React parity from authoring through export,
including `all_together`, `individual`, and `both` execution, dataset-free
individual results, aggregate progress, cancellation, and 1--4096 durable
chunks. The ledger is 30 verified / 1 partial. All remain model-scenario
screening views, not global main effects, causal anatomy, governed human
validation, or coaching authority. Hosted visual review found and fixed an
unlabeled PyQt policy selector; exact Linux candidate `1f3f6ca7` is proposed
without widening its calibrated renderer envelope.

PR #4705 maps all 31 #4433 obligations; trusted run `32689177846` proves only
the initial React/PyQt visibility, accessibility, performance, and baseline
tier. PR #4733 merged V0.1 with purpose, prerequisites, and reciprocal
counterparts; PR #4736 merged strict TypeScript-reader parity as `34a809d9`.
PR #4738 merged V5.2's fail-closed changed-path governance as `4b4aec421`.
The audit is 8 verified / 23 partial; seven blockers and two human actions remain.
PRs #4835/#4837 are protected-merged through `d7a95e2a4`; the fifth manifest expands all 20 tabs
over registered states/reference cases and binds scientific/nonvisual context.
PR #4838's extension checklist and immutable consumer map move the audit to 10 verified / 21 partial.
Executed render, performance, decimation, approved-image, and human gaps remain; no pixel tolerance was loosened.

### Adding a Tab: The Five-Manifest Lockstep (Read Before Starting C6/C7/H4)

A new tab is **not** just a widget. Five packaged manifests declare the tab set,
cross-checked by **order-strict tuple equality** on `(surface, tab_id)` against
`visualization_tabs.v1.json`. Adding to four of five — or to all five in a
different order — fails with a message that does not name the offending file.

All five live in `src/rate_of_closure/`: `visualization_tabs.v1.json` is the
authority (20 entries = 10 `pyqt` + 10 `react`); `visualization_accessibility.v1.json`,
`visualization_performance.v1.json`, `visual_baselines.v1.json`, and
`visualization_acceptance.v1.json` must match it entry-for-entry, in order.

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
  invocation with `MYPYPATH=src:src/python/src`; tests are excluded. The flag is
  **`--follow-imports=silent`**, matching `ci-standard.yml`; `=skip` crashes
  mypy 1.13 (`unresolved placeholder type`) on files already on `main`, so a
  `skip` run reports failures that CI does not have.
  `MYPYPATH='src;src/python/src' py -3.12 -m mypy --ignore-missing-imports --follow-imports=silent <changed non-test files>`
- **`tools_core` capability is two-tier** — a wheel can expose
  `simulate_trajectory` yet lack the tee-aware full-state API; guard on the
  specific capability. `test_club_view_camera.py`'s cadence test budgets
  process-CPU time; it can still trip when sibling suites saturate the box.
- PowerShell `Set-Content` and `pathlib.write_text` can rewrite LF as CRLF;
  preserve newlines explicitly.
- **`detect_secrets scan` writes native separators** — on Windows run it
  _before_ normalising the baseline to forward slashes, never after
  (`tests/ops/test_detect_secrets_baseline.py` rejects backslash keys).
- **A purely additive file can still be wrong.** `PrimaryViewTabs.test.tsx` and
  `TorqueProfilePanel.test.tsx` add lines, delete none, and both fail against
  `main`'s components. Never delete an `origin/main` file to pass a check.
