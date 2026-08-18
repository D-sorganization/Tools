# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-17

> **Current state only.** `CLAUDE.md` caps handoff docs at 150 lines and keeps
> history in git. The 140 dated entries this file had accumulated through
> 2026-08-15 are preserved verbatim in
> [`docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md`](../../docs/agent_handoff_archive/2026-08_rate_of_closure_handoff_log.md).
> Do not append dated entries here again — that is how it reached 2,205 lines.

## What This Tool Is Now

A swing → impact → ball-flight simulator with **PyQt6 and React surfaces held
at parity**, plus a Rust fast path. It began as a single closure-rate
calculator (twist model: GC-path vs impact-point-path gap, °/ft).

The PyQt6 entry point is `rate_of_closure.ui.pyqt6.main_window`, class
`RateOfClosureMainWindow`, titled "Rate of Closure Impact Explorer" — that exact
string is what UpstreamDrift's launcher manifest advertises. The React mirror
lives in `web/` (Vite/TS, ~1,095 tests across 132 files).

Physics lives in `src/shared/python/swing_sim/`, deliberately outside this tool
so UpstreamDrift reaches it through `vendor/ud-tools`:

- `swing_sim/impact/` — impact model, gear effect, `SpringDamperImpactModel`
  (Kelvin-Voigt contact force history). Epic #4130 extends this rather than
  duplicating it.
- `swing_sim/flight/` — literature ball-flight models, capability observation
  and evaluator, ground transfer, regional ground pipeline, wind strategy.
- `swing_sim/ground/` — skid/roll/bounce, surface profiles, regional execution.
- `swing_sim/variation/` — Monte Carlo, dispersion, Morris sensitivity.
- `swing_sim/solver/` — goal-driven multi-start solver, target regions.

## Delivery Pattern — Read This Before Touching #4466

Long stacked PRs are dead here. PR #4119, which everything once stacked on, was
**closed unmerged on 2026-08-14**; #4124 and #4129 read "merged" but merged only
into their own stacked bases and died with it. Until 2026-08-16
`src/rate_of_closure/` on `main` held exactly one file — its own handoff doc.

Content now lands as **consolidations and slices rebuilt onto current `main`**.

`consolidated/rate-closure-remainder-2026-08-13` (PR #4466) **cannot be merged
by any strategy.** After #4473 squash-merged, the shared merge-base collapses to
a commit predating `src/rate_of_closure/`, so every overlapping file is a
both-added conflict with no common ancestor — 281 of them, none trivially
identical. Both directions produce a broken tree, measured:

| strategy | result |
| --- | --- |
| `-X theirs` (prefer main) | 47 failed / 40 errors — net-new files call APIs main's versions lack |
| `-X ours` (prefer branch) | 19 collection errors — breaks main's newer morris/variation work |

It is landing as feature slices instead: #4517 `swing_sim.ground`, #4518
`swing_sim.flight` + React parity, #4519 club/plotting, #4520 variation.

**The branch is not uniformly newer than `main`.** Four times a file taken
wholesale would have reverted shipped work — `flight/tests/test_wind.py` (would
restore a 1e-12 parity tolerance that fails on Linux, see #4513),
`web/src/model/flight.ts` (+31/−107, a net reduction), `plotting/render.py`
(predates the point-inspector work; failed 13 tests), and 16 of `variation`'s
files (`ensemble_chunks.py` −360, `_ensemble_parser.py` −340). Check each file
against `main` before copying; a strategy flag will silently revert.

## Remaining From #4466

| area | net-new | notes |
| --- | --- | --- |
| `simulation/` | 5 | ground playback; needs `club` (#4519) |
| `application/` + `web_authority/` | 34 + 15 | **import each other** — must land together |
| `ui/pyqt6/` | 40 | 80 files also modified |
| `web/` | 215 | 229 files also modified |

`swing_sim` is complete: `variation` needs nothing (`main` is a superset),
`impact` has no symbol differences, `solver` has one branch-only test.

## Must-Read Architecture Pointers

1. `src/rate_of_closure/README.md` — frame and unit conventions, dossier
   sourcing, run/build instructions.
2. `src/shared/python/swing_sim/impact/` — the contact-force law epic #4130
   extends.
3. `src/rate_of_closure/web/src/model/__fixtures__/` — the shared golden
   fixtures pinning Python↔TypeScript parity. Changing one is a contract change
   on **both** sides; land them together.
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
  Symbol-level comparison is necessary but not sufficient — it misses changes
  inside function bodies (the `test_wind.py` tolerance) and data-only content
  (catalog entries). Count entries for catalogs; read the diff for the rest.
- Do not change a golden fixture on one side only — Python and TypeScript both
  consume them, and CI runs the two suites separately.
- Do not duplicate the Kelvin-Voigt contact-force law; #4130 shares
  `SpringDamperImpactModel`.
- Do not exceed 500 LOC per file in `rate_of_closure`, `swing_sim` or
  `swing-core` — sub-package instead.
- Do not eagerly import `assembly_binding`, `engineering_sidecar` or
  `simulation_adapter` from `club/__init__.py`. They reach
  `shared.python.golf_club → swing_sim.variation → solver → flight → scipy`,
  which drags SciPy into `club.types` and breaks the Morris UI import contract.
  Use the lazy-export map already there.

## Known Local-Environment Traps

- **mypy 1.13 crashes on Python 3.13** (`unresolved placeholder type None`) for
  multi-file sets here. Check files individually; CI runs 3.12 and reports
  normally. Both `no-any-return` findings in #4520 were caught this way.
- **`tools_core` capability is two-tier.** A wheel can expose
  `simulate_trajectory` while lacking the tee-aware full-state API. Guard on the
  specific capability, not just `is_rust_available()`.
- Rewriting `SPEC.md` with PowerShell `Set-Content` normalises it to CRLF and
  turns the next merge into a whole-file conflict. Edit it in place.
