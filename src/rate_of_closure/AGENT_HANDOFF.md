# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-20

> **Current state only**, capped at 150 lines; history lives in git and in
> [`docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md`](../../docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md).
> Do not append dated entries — that is how it reached 2,205 lines.

## What This Tool Is Now

A swing → impact → ball-flight simulator with **PyQt6 and React surfaces held
at parity**, plus a Rust fast path. It began as a single closure-rate calculator
(twist model: GC-path vs impact-point-path gap, °/ft).

PyQt6 entry: `rate_of_closure.ui.pyqt6.main_window:RateOfClosureMainWindow`,
titled "Rate of Closure Impact Explorer" — that exact string is what
UpstreamDrift's launcher manifest advertises. React mirror: `web/` (Vite/TS,
~1,570 tests across 193 files).

Physics lives in `src/shared/python/swing_sim/` — outside this tool, so
UpstreamDrift reaches it through `vendor/ud-tools`: `impact/` (contact law,
gear effect; #4130 extends `SpringDamperImpactModel` rather than duplicating
it), `flight/`, `ground/`, `variation/`, `solver/`. Club physics for the
fitting epics is the sibling `shared/python/golf_club/`.

## Delivery Pattern, and the One Thing Left From #4466

Long stacked PRs are dead here: #4119 closed unmerged and took its stack with
it. Content lands as **slices rebuilt onto current `main`**.

PR #4466 cannot be merged: its merge-base predates this package, leaving 281
both-added conflicts. Twenty-two current-main slices landed as #4517–#4547.
Only the camera cluster remains, owned by #4571. It is a reimplementation:
wiring the mixin passes 20 camera tests but regresses three main-owned tests and
requires about 20 more UI files. Do not merge/slice #4466 or close it before
#4571. The old branch is not uniformly newer; diff every candidate and require
main's existing suite to pass before adding new tests.

## Active Epics — Golf Epics Merged

Launch-monitor epic #4583 has merged consolidated Release A. PyQt6 and
React now share explicit-identity projects, arbitrary-variable analysis,
dispersion/target-error, attested session summaries, persistence/export, and a
safe capability-driven Neural Model Lab. The desktop client can load all
261,666 manifest-verified private-authority rows from an explicitly authorized
local root while its plot stays bounded. UpstreamDrift v2 is the canonical
contract; embedded calculations are labeled compatibility/offline. The current
#4277 slice adds identity-attested pooled, player-centered, between-player,
per-player and random-effects covariation plus exploratory all-pairs scans to
both clients, with unit-labelled plots and complete backing exports. The next
performance slice adds hash-verified, user-authorized expected-strokes baseline
artifacts and attested longitudinal player/population inference in both clients;
no baseline data is bundled. Do not claim vendor emulation or paired-device
validation: Release B remains open and has no real paired observations.

#4584/#4599 merged UpstreamDrift's canonical source-backed strokes-gained v2
endpoint into both clients with exact lie/context/target strata, optional
benchmark uncertainty, structured exclusions, and explicitly attested grouped
summaries. A blank authority URL retains a labelled local compatibility path;
no baseline or inferred identity is bundled. #4600 owns the inspected
post-merge PyQt visual-reference approval. #4602 isolates the trusted React
functional, Axe, and protected timing phases while retaining unchanged budgets.

Club Fitting #4549, Heavy Hit #4562, and packaging #4579 are complete and
merged. Their physics lives shared-first in `shared/python/{golf_club,swing_sim}`;
see the two contracts under `docs/specs/`.

### Adding a tab: the four-manifest lockstep (read before starting C6/C7/H4)

A new tab is **not** just a widget. Four packaged manifests declare the tab set,
cross-checked by **order-strict tuple equality** on `(surface, tab_id)` against
`visualization_tabs.v1.json`. Adding to three of four — or to all four in a
different order — fails with a message that does not name the offending file.

All four live in `src/rate_of_closure/`: `visualization_tabs.v1.json` is the
authority (18 entries = 9 `pyqt` + 9 `react`); `visualization_accessibility.v1.json`,
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
5. `src/shared/python/golf_club/AGENT_HANDOFF.md` — the fitting/heavy-hit
   physics surface the GUI children bind to.

## Gate Commands (this tool)

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

- **Reproduce CI's mypy exactly or it disagrees in both directions.** CI passes
  _every_ changed file to **one** invocation with `MYPYPATH=src:src/python/src`.
  Per-file runs degrade those imports to `Any`, inventing `no-any-return`
  findings CI lacks and hiding the `redundant-cast` ones it has — #4531 failed
  `quality-gate` on exactly that. Use **3.12**; mypy 1.13 errors internally on
  3.13 for multi-file sets. `tests/` is excluded from mypy entirely.
  `MYPYPATH='src;src/python/src' py -3.12 -m mypy --ignore-missing-imports --follow-imports=skip <changed non-test files>`
- **`tools_core` capability is two-tier** — a wheel can expose
  `simulate_trajectory` yet lack the tee-aware full-state API; guard on the
  specific capability. `test_club_view_camera.py`'s cadence test asserts
  wall-clock time and is flaky under load.
- PowerShell `Set-Content` and Python `pathlib.write_text` rewrite LF files as
  CRLF (an 834-line phantom diff from a 4-line edit); use
  `io.open(..., newline="")`.
- **`detect_secrets scan` writes native separators** — on Windows run it
  _before_ normalising the baseline to forward slashes, never after
  (`tests/ops/test_detect_secrets_baseline.py` rejects backslash keys).
- **A purely additive file can still be wrong.** `PrimaryViewTabs.test.tsx` and
  `TorqueProfilePanel.test.tsx` add lines, delete none, and both fail against
  `main`'s components. Never delete an `origin/main` file to pass a check.
