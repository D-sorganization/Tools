# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-18

> **Current state only.** `CLAUDE.md` caps this at 150 lines; history lives in
> git and in
> [`docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md`](../../docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md).
> Do not append dated entries — that is how it reached 2,205 lines.

## What This Tool Is Now

A swing → impact → ball-flight simulator with **PyQt6 and React surfaces held
at parity**, plus a Rust fast path. It began as a single closure-rate calculator
(twist model: GC-path vs impact-point-path gap, °/ft).

PyQt6 entry point: `rate_of_closure.ui.pyqt6.main_window:RateOfClosureMainWindow`,
titled "Rate of Closure Impact Explorer" — that exact string is what
UpstreamDrift's launcher manifest advertises. The React mirror lives in `web/`
(Vite/TS, ~1,570 tests across 193 files).

Physics lives in `src/shared/python/swing_sim/`, outside this tool so
UpstreamDrift reaches it through `vendor/ud-tools`:

`impact/` (contact law, gear effect — epic #4130 extends `SpringDamperImpactModel`
rather than duplicating it), `flight/` (ball-flight models, capability evaluator,
ground transfer, wind strategy), `ground/` (skid/roll/bounce, surface profiles),
`variation/` (Monte Carlo, dispersion, Morris) and `solver/`.

## Delivery Pattern — Read This Before Touching #4466

Long stacked PRs are dead here: #4119 closed unmerged and took its stack with it.

Content now lands as **consolidations and slices rebuilt onto current `main`**.

`consolidated/rate-closure-remainder-2026-08-13` (PR #4466) **cannot be merged
by any strategy.** After #4473 squash-merged, the merge-base collapses to a
commit predating `src/rate_of_closure/`, so all 281 overlapping files are
both-added conflicts with no common ancestor. Both directions were measured and
both produce a broken tree: `-X theirs` → 47 failed / 40 errors, `-X ours` → 19
collection errors.

It is landing as feature slices instead — 20 merged as of 2026-08-18, from
#4517 through #4545. Each one's reasoning is in its SPEC.md §12 row (1.17.x);
read those before re-deriving a decision.

**The branch is not uniformly newer than `main`.** Repeatedly, a file taken
wholesale would have reverted shipped work — `flight/tests/test_wind.py`
(restores a 1e-12 tolerance that fails on Linux, #4513), `web/src/model/flight.ts`
(a net reduction), `plotting/render.py` (failed 13 tests), 16 `variation` files.
Diff every shared file against `main` before copying.

**But "it deletes lines" is not automatically disqualifying.** The flight
integrator (#4542) and `capabilityOptimizer` (#4545) both delete only their own
helpers being reshaped, and both were taken only after `main`'s entire existing
suite passed against the swapped-in version, before any new tests. Meet that
standard or leave the file alone.

## Remaining From #4466

52 files as of 2026-08-18, and they are **not** blocked on effort — they are
blocked on one thing: the camera-controls cluster.

| area | left | blocker |
| --- | --- | --- |
| `tests/rate_of_closure` | 19 | assert APIs `main`'s modules lack |
| `web/src` | 18 | ground playback 3D, chip forgiveness, view compositor |
| other | 15 | `web/tests` remnants, unrelated trees |

**The camera cluster is a reimplementation, not a migration — measured, not
assumed.** Wiring `CameraViewportMixin` into `simulation_view` and `flight_view`
passes 20 of 20 camera GUI tests but regresses three `main`-owned ones
(named-camera azimuth, legend placement, the accessibility control-count pin),
and reproducing the branch's Face-On behaviour needs ~20 more `ui/pyqt6` files
that **delete** shipped work — `flight_explorer_run.py` (−324),
`flight_view_bundle.py` (−200), `club_view_render.py` (−185),
`flight_view_inspector.py` (−157). Make it its own epic; do not slice it.

`swing_sim` is complete. `application`, `web_authority`, `ui/pyqt6`,
`web_companion`, `web_distribution` and the React `model`/component/hook layers
have all landed.

## Active Epic: Club Fitting Tester (#4549)

Contract `docs/specs/CLUB_FITTING_TESTER.md`; children #4550–#4556. **Shared-first**:
physics/wires in `shared/python/{golf_club,swing_sim}`; tool-local = UI binding only.
Order: C1 ✓ → C2 shaft deltas → C3 document → C5 interchange → C4 engine → C6/C7 GUIs.

## Must-Read Architecture Pointers

1. `src/rate_of_closure/README.md` — frame and unit conventions, run/build.
2. `src/shared/python/swing_sim/impact/` — the contact-force law #4130 extends.
3. `web/src/model/__fixtures__/` — shared golden fixtures pinning Python↔TS
   parity. Changing one is a contract change on **both** sides; land together.
4. `rust_core/swing-core/` — pendulum EOM and plane projection, pyo3 + wasm.

## Gate Commands (this tool)

```bash
python3 -m pytest tests/rate_of_closure src/shared/python/swing_sim -n auto --timeout=300
cd src/rate_of_closure/web && npm run test && npm run build && npx tsc --noEmit && npx eslint .
cargo test -p swing-core
python3 -m ruff check src/rate_of_closure src/shared/python/swing_sim
python3 scripts/check_test_assertions.py --changed-files <file-list>
```

## Do-Not List

- **Do not append a dated entry to this file.** See the header.
- **Do not merge #4466, #4446 or #4447 with a strategy flag.** See above.
- **Do not take a branch file wholesale without diffing it against `main`.**
  Symbol comparison is necessary but not sufficient — it misses function-body
  changes (the `test_wind.py` tolerance) and data-only content. Read the diff.
- Do not exceed 500 LOC per file in `rate_of_closure`, `swing_sim` or
  `swing-core` — split along a real seam, as the two execution-workspace mixins
  in `ui/pyqt6` do.
- Do not eagerly import `assembly_binding`, `engineering_sidecar` or
  `simulation_adapter` from `club/__init__.py` — they reach SciPy via
  `golf_club → swing_sim.variation → solver → flight`, breaking the Morris UI
  import contract. Use the lazy-export map already there.

## Known Local-Environment Traps

- **Reproduce CI's mypy exactly, or it will disagree with you in both
  directions.** CI passes *every* changed file to **one** invocation with
  `MYPYPATH=src:src/python/src`, so imports inside the changed set resolve.
  Checking files one at a time degrades those same imports to `Any`: it invents
  `no-any-return` findings CI does not have, and hides the `redundant-cast`
  findings it does. #4531 failed `quality-gate` on exactly that. Use Python
  **3.12** — mypy 1.13 raises an internal error on 3.13 for multi-file sets, so a
  3.13 batch run tells you nothing:
  `MYPYPATH='src;src/python/src' py -3.12 -m mypy --ignore-missing-imports --follow-imports=skip <every changed non-test file>`
  CI excludes `tests/` from mypy entirely, so findings there do not matter.
- **`tools_core` capability is two-tier** — a wheel can expose `simulate_trajectory`
  yet lack the tee-aware full-state API; guard on the specific capability.
- `test_club_view_camera.py`'s cadence test asserts wall-clock time; flaky under load.
- PowerShell `Set-Content` and Python `pathlib.write_text` both rewrite LF files
  as CRLF (an 834-line phantom diff from a 4-line edit). Use
  `io.open(..., newline="")` and match the file's existing endings.
- **`detect_secrets scan` writes native separators**, so on Windows it must be
  run *before* normalising the baseline to forward slashes, never after.
  `tests/ops/test_detect_secrets_baseline.py` rejects backslash keys, and that
  suite is changed-file scoped, so the debt stays invisible until a PR touches
  `tests/ops/`.
- **A file that is purely additive by line count can still be wrong.**
  `PrimaryViewTabs.test.tsx` and `TorqueProfilePanel.test.tsx` both add lines and
  delete none, and both fail against `main`'s unchanged components. Never delete
  a file present on `origin/main` to make a check pass — restore `main`'s
  version instead.
