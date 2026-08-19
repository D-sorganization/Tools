# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-18

> **Current state only**, capped at 150 lines by `CLAUDE.md`; history lives in
> git and in [`docs/agent_handoff_archive/2026-08_tools_root_handoff_log.md`](docs/agent_handoff_archive/2026-08_tools_root_handoff_log.md).
> Do not append dated entries here again.

## Where This Repo Is Headed

Tools is the D-sorganization fleet's shared engineering-tools monorepo (45+
tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). The current
center of gravity is `src/rate_of_closure`, grown from a closure-rate
calculator into a swing → impact → ball-flight simulation platform. Since
early August the delivery pattern has shifted from long stacked PRs to
**scoped consolidations rebuilt directly onto current `main`**.

| Epic  | Status (one line)                                                                                                                                                                                                                         |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #4103 | Swing-Impact-Ball-Flight platform. Open. Stack PR #4119 was **closed, not merged**; its content landed as 22 slices (#4517-#4547). Remaining: the camera cluster (needs its own epic, see below) and Phase 7 (WASM web parity, Pages CI). |
| #4120 | Investigation & Variation Suite. Open. PR #4124 **merged**.                                                                                                                                                                               |
| #4125 | Realistic clubs / kinetics / putting / showcase. Open. PR #4129 **merged**. H5 (public release-management repo) still not started.                                                                                                        |
| #4130 | Impact-interval club dynamics. Open, foundation only — F1 formulation doc not started, no PR.                                                                                                                                             |
| #4142 | Ensemble variation, quiet zones, sensitivity attribution. Open. Visualization/authority slice landed via **#4473**.                                                                                                                       |
| #4146 | Shared Club Builder. Open. First slice #4147 **closed**; assembly physics contracts landed in #4157.                                                                                                                                      |
| #4433 | Visual-first tab visibility and visualization-led UX. Open. Landed via **#4473**.                                                                                                                                                         |
| #4549 | Club Fitting Tester (OEM-grade). Open. **C1-C5 merged in #4557** - mesh inertia tensor, shaft delivery deltas, OEM fitting document, delivery interchange, counterfactual engine. **C6/C7 (GUI tabs) remain.**                            |
| #4562 | Heavy Hit - hand/body coupling at impact. Open. **H1-H3 merged in #4568**; H5 pin bump is UD #8767. Headline: physiological hands change driver ball speed **<1%**. **H4 (GUI) remains.**                                                 |

Per-tool detail: `src/rate_of_closure/AGENT_HANDOFF.md` (current, refreshed by
#4473), `src/pendulum_simulator/AGENT_HANDOFF.md`,
`src/rotation_converter/AGENT_HANDOFF.md`.

## Open PR Situation — Read Before Filing Anything

**16 PRs are open, none are drafts.** The golf queue is empty: everything from
the #4549 and #4562 epics has merged. What remains open is unrelated to the
golf platform except #4466.

| PR    | Scope                                                                      |
| ----- | -------------------------------------------------------------------------- |
| #4466 | Rate of Closure remainder — **content-complete except the camera cluster** |
| #4449 | P1AM plant historian + SCADA foundation (supersedes #4065, #4091)          |
| #4447 | Variation / Morris sensitivity suite (consolidates 34 drafts)              |
| #4446 | Ground study + rate-of-closure suites (supersedes #4409/#4410)             |

**#4466 still cannot be merged by any strategy** — after #4473 squash-merged,
its merge-base collapses to a commit predating `src/rate_of_closure/`, making
all 281 overlapping files both-added conflicts with no common ancestor. It was
landed as 22 slices instead (#4517–#4547). **Only the camera-controls cluster
is left, and it is a reimplementation, not a migration**: wiring
`CameraViewportMixin` in passes 20/20 camera GUI tests but regresses three
`main`-owned ones, and matching the branch's Face-On behaviour needs ~20 more
`ui/pyqt6` files that _delete_ shipped work. **#4571 is that epic; do not slice it, and
do not close #4466 until #4571 lands.**

The 39 `codex/4142-*` / `codex/4433-*` drafts were closed on 2026-08-16 as
superseded by merged #4473, and their work is on `main` (Morris chain under
`application/morris/`, the #4433 inspectors under `ui/pyqt6/`). **Their
branches are intact; reopen rather than rebase** — merging one now would remove
current `main` content. #4473 does **not** supersede #4438 (merged).

Other live non-golf work: `src/shared/python` hygiene (#4507 lint
normalisation, #4509 mypy debt), CI repairs (#4454 merge-hold guard, #4469
architecture guards always-on), and eight Bolt/Palette micro-PRs.

**Known-red on `main`, already filed — do not re-diagnose:** #4561 (browser
qualification: companion readiness metadata + missing Firefox/WebKit binaries;
the workflow is scoped to Chromium meanwhile), #4569 (two `1.17.40` SPEC rows
glued onto one table line by a union merge), #4558/#4559/#4560 (cross-runtime
fixture parity gaps).

## Must-Read Architecture Pointers

1. `CLAUDE.md` — repo-wide conventions, CI gate list, cross-repo dependency
   rules (Tools is a leaf dependency; UpstreamDrift and Gasification_Model
   consume it).
2. `docs/architecture/CANONICAL_TOPOLOGY.md` — canonical repo topology policy.
3. `SPEC.md` — living specification; §12 Change Log requires a dated row for
   every PR touching `src/` (enforced by `spec-check.yml`, see gates below).
4. `docs/AGENT_HANDOFF_TEMPLATE.md` — template for a new tool's handoff doc.

## Gate Commands (repo-wide)

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check (Ruff, NOT Black)
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m contract                     # API contract tests (downstream-facing)
python3 -m pytest -m integration --timeout=60     # cross-repo integration
```

SPEC freshness (CI job `spec-freshness` in `.github/workflows/spec-check.yml`):
any PR touching `src/**`, `tests/**`, `config/**`, `pyproject.toml`,
`Cargo.toml`, `package.json`, or `requirements.txt` must also modify
`SPEC.md` in the same PR, or carry the `spec-exempt` label. This includes
`src/<tool>/AGENT_HANDOFF.md` edits, since they live under `src/`.

Note: `ruff format --check` reports four pre-existing failures under
`src/data_processing/data_processor/python/tests/` — on `main`, not yours.

## Do-Not List

- **Do not reopen-and-rebase the closed `codex/4142-*` / `codex/4433-*`
  branches.** They are superseded by merged #4473 and have diverged far enough
  that merging one would remove current `main` content.
- **Do not add per-file Python version guards.** `conftest.py` reads each
  package's declared `requires-python` and skips below it (`CLAUDE.md`).
- **Do not append a dated entry to this file.** It hit 2,708 lines that way,
  18x the limit. Put history in the commit message.
- Do not regenerate `tests/sidekick_api_baseline.json` with
  `--regenerate-api-baseline` — it blesses existing drift, not just your change.
  Hand-edit it and coordinate a downstream migration.
- Do not change public signatures in `src/shared/python/**` without coordinated
  migration issues in UpstreamDrift and Gasification_Model (`CLAUDE.md`).
- Do not edit the shared Python surface from inside a consumer's vendored copy;
  Tools is the source of truth (#4495 exists because three fixes were orphaned).
- Do not import across package boundaries — LoD is enforced.
- Do not exceed the 500-LOC file budget on new/modified files in the golf-sim
  packages (`rate_of_closure`, `swing_sim`, `swing-core`) — sub-package,
  don't grow monoliths. The repo-wide protected budget is 1,200 lines.
- Do not use `git commit --no-verify` / `--push --no-verify` to bypass hooks;
  see `CLAUDE.md` Hook bypass policy.
- Do not assume delta CI covers you. Changed-file selection hides whole-tree
  debt on the lint and type gates until a PR touches enough shared files.
- Do not hand-roll a GitHub Pages deploy workflow for `rate_of_closure/web` —
  Phase 7 of #4103 owns this.
- **Do not `git add -A` after a stash pop or merge.** Raw conflict markers were
  staged into `SPEC.md` and pushed twice this way. `grep '<<<<<<<'` and confirm
  `git diff origin/main -- SPEC.md` is exactly one insertion before pushing.
- Do not remove a stdlib-XML `# nosec` without replacing the justification —
  `quality-gate` runs bandit (B405/B314) and the repo's convention is
  nosec-with-a-written-reason, not swapping in `defusedxml`.

## Short-Term Roadmap (ordered)

1. **Finish the two golf epics — GUI surfaces only.** #4555 (C6, PyQt6 Club
   Tester tab), #4556 (C7, React parity), #4566 (H4, heavy-hit panels). All
   physics and wires are merged and shared-first; these are binding work.
   `src/rate_of_closure/AGENT_HANDOFF.md` carries the exact four-manifest
   recipe — read it first, it is the non-obvious part.
2. **Land the camera-cluster epic #4571** so #4466 can finally close.
3. Clear the `src/shared/python` hygiene pair (#4507, #4509) — these unblock
   downstream consumers.
4. Land the CI repairs (#4454, #4469) and fix the filed reds (#4561, #4569).
5. Start #4130 Phase F1 (formulation document).
6. Phase 7 of #4103: WASM swap for the web mirror + real Pages CI deploy.
7. #4125 H5: stand up the public release-management repo (cross-repo).
