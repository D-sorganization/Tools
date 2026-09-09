# AGENT_HANDOFF — Rate_of_Closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-09-09
> **Current state only**, capped at 150 lines; history lives in git and `docs/agent_handoff_archive/`.

## What This Tool Is Now

A swing → impact → ball-flight simulator with parity **PyQt6 and React surfaces**, plus a Rust fast path (`swing-core`); it began as a closure-rate calculator.

PyQt6 entry: `rate_of_closure.ui.pyqt6.main_window:RateOfClosureMainWindow`.
React mirror: `web/` (Vite/TS, ~1,570 tests across 193 files).
Physics lives in `src/shared/python/swing_sim/` and `src/shared/python/golf_club/`.

## Test Isolation #5114

- Candidate: fix/5114-qt-deferred-cleanup, based on 184e453db. Deliver queued native Qt deletions after fixture finalizers while retaining pytest-qt exception capture; no application is created for non-GUI cases.
- TDD: widget/timer regression fails before the hook; two regressions pass afterward (17.10 s Windows). Missing PyQt6/pytest-qt collection skips are verified separately. Full Linux patch fails: 2,880 passed, 29 skipped, three worker losses and a 900-second cap. Fourteen serial profiled GUI controls pass; biased sampling does not explain full-suite losses. #5114 stays open.
- GitHub CLI/connector require reauthentication; publication and lease renewal are unavailable. Current #5114 lease expires 2026-09-09T09:55:42Z. Recheck ownership once access returns.
- Evidence, failed controls and preserved environments: docs/development/impact-acoustics/RATE_QT_ISOLATION.md. Keep existing test deadlines and renderer references.

## Active Epics — Golf Epics Merged

- Launch-monitor #4583 Release A merged; Release B open (vendor emulation requires real paired data). #4584/#4599 merged strokes-gained v2.
- ADR-0046 Stage 2 is classified; no modules retire yet. 20 symbols from 10 modules are pinned by UpstreamDrift drift gates.
- Source-backed SG excludes and audits; it does not raise (ADR-0048 G1-D3). Both runtimes assert `input_row_count == included_row_count + total_excluded`.
- Club Fitting #4549, Heavy Hit #4562, packaging #4579, Putting #4800 (P1-P9), and Clubhead-realism #4799 are complete. Python remains sole Monte-Carlo authority.
- #4142 variation and sensitivity: R10-R14 merged; R14.6/calibrated-renderer PRs #4835/#4837 on main through `d7a95e2a4`.

### Adding a Tab: The Five-Manifest Lockstep

Five packaged manifests in `src/rate_of_closure/` declare the tab set, cross-checked by **order-strict tuple equality** on `(surface, tab_id)`: `visualization_tabs.v1.json` is authority (20 entries = 10 `pyqt` + 10 `react`), matched by `visualization_accessibility`, `visualization_performance`, `visual_baselines`, and `visualization_acceptance`.

- Surface strings are `pyqt` and `react`, not `pyqt6`.
- PyQt6 registration: build widget in `ui/pyqt6/main_window.py`, add `PrimaryTabSpec(module_id, widget, label)`.
- Visual baselines are captured on Linux fleet runners (`RATE_VISUAL_BASELINE_CANDIDATE_DIR`).
- Gates: `tests/rate_of_closure/test_visualization_*_manifest`, `test_visual_baseline_compare.py`, `test_pyqt_visualization_tab_visibility.py`.

## Renderer prerequisite #4844

- Worktree `C:/Users/diete/Repositories/Tools-impact-render`; branch `fix/4844-consistent-pyqt-renderer`; commit `SELF`; PR #5090 open.
- PR and trusted PyQt capture use one digest-pinned Ubuntu 24.04 container; trusted remains on fleet runners. Exact font versions replace alternate-version acceptance; Qt runtime/SIP pins are checked.
- Published `df4101f28`: two Linux captures pass 73 browser and 23 PyQt tests; all ten PyQt PNGs are byte-identical. The reviewed 20-image proposal preserves tolerances; its provenance test fails before and all 60 local contracts pass after refresh. Both captured sets pass the comparison CLI. Normal hooks and fresh CI remain; #5087 closed unmerged.
- Continue from `docs/development/rate-pyqt-renderer-4844.md`. Do not accept another host stack under the same image identity.

## Must-Read Architecture Pointers

1. `src/rate_of_closure/README.md` — frame and unit conventions, run/build.
2. `src/shared/python/swing_sim/impact/` — contact-force law #4130 extends.
3. `web/src/model/__fixtures__/` — golden fixtures pinning Python↔TS parity.
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

- **Do not exceed 150 lines in this file.**
- **Do not append dated entries to this file.** Put history in commit messages.
- Do not exceed 500 LOC per file in `rate_of_closure`, `swing_sim`, or `swing-core`.
- Do not eagerly import `assembly_binding`, `engineering_sidecar`, or `simulation_adapter` from `club/__init__.py`.
- Reproduce CI's mypy exactly with `--follow-imports=silent` and `MYPYPATH='src;src/python/src'`.
- Do not rebind PyQt visual baselines without verifying candidate artifacts.
