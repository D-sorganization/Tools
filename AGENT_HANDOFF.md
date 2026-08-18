# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-16

> **Current state only.** `CLAUDE.md` caps handoff docs at 150 lines and keeps
> history in git. The 137 dated entries this file had accumulated through
> 2026-08-15 are preserved verbatim in
> [`docs/agent_handoff_archive/2026-08_tools_root_handoff_log.md`](docs/agent_handoff_archive/2026-08_tools_root_handoff_log.md).
> Do not append dated entries here again.

## Where This Repo Is Headed

Tools is the D-sorganization fleet's shared engineering-tools monorepo (45+
tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). The current
center of gravity is `src/rate_of_closure`, grown from a closure-rate
calculator into a swing → impact → ball-flight simulation platform. Since
early August the delivery pattern has shifted from long stacked PRs to
**scoped consolidations rebuilt directly onto current `main`**.

| Epic  | Status (one line)                                                                                                                                                             |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #4103 | Swing–Impact–Ball-Flight platform. Open. Original stack PR #4119 was **closed, not merged**; content landed via consolidations. Phase 7 (WASM web parity, Pages CI) still open. |
| #4120 | Investigation & Variation Suite. Open. PR #4124 **merged**.                                                                                                                    |
| #4125 | Realistic clubs / kinetics / putting / showcase. Open. PR #4129 **merged**. H5 (public release-management repo) still not started.                                             |
| #4130 | Impact-interval club dynamics. Open, foundation only — F1 formulation doc not started, no PR.                                                                                  |
| #4142 | Ensemble variation, quiet zones, sensitivity attribution. Open. Visualization/authority slice landed via **#4473**.                                                            |
| #4146 | Shared Club Builder. Open. First slice #4147 **closed**; assembly physics contracts landed in #4157.                                                                           |
| #4433 | Visual-first tab visibility and visualization-led UX. Open. Landed via **#4473**.                                                                                              |

Per-tool detail: `src/rate_of_closure/AGENT_HANDOFF.md` (current, refreshed by
#4473), `src/pendulum_simulator/AGENT_HANDOFF.md`,
`src/rotation_converter/AGENT_HANDOFF.md`.

## Open PR Situation — Read Before Filing Anything

**21 PRs are open and none are drafts.** The 39 `codex/4142-*` /
`codex/4433-*` drafts were closed on 2026-08-16 as superseded by merged
PR #4473 ("CONS-RATE: Complete #4142/#4433 visualization authority"), which
says so in its own description. Their work is on `main` — the Morris chain as
`src/rate_of_closure/application/morris/{host,client,contracts}.py`, the #4433
chain as `flight_sample_inspector.py`, `putting_sample_inspector.py`,
`plot_point_inspector.py`, `visual_state_frame.py`, and
`visualization_tab_audit.py`. **Their branches are intact; reopen rather than
rebase** — those branches have diverged far enough that merging one now would
remove current `main` content. #4473 does **not** supersede #4438 (merged).

The consolidations are the real queue:

| PR    | Scope                                                                       |
| ----- | --------------------------------------------------------------------------- |
| #4466 | Rate of Closure remainder — **cannot be merged as-is**, see below           |
| #4449 | P1AM plant historian + SCADA foundation (supersedes #4065, #4091)            |
| #4447 | Variation / Morris sensitivity suite (consolidates 34 drafts)                |
| #4446 | Ground study + rate-of-closure suites (supersedes #4409/#4410)               |

**#4466 is blocked by a merge-base collapse.** After #4473 squash-merged, the
shared merge-base falls back to a commit predating `src/rate_of_closure/`, so
every overlapping file is a both-added conflict with no common ancestor (281 of
them). It is being landed as standalone slices against current `main` instead —
#4517 is the first (`swing_sim.ground`). Expect the other consolidations to hit
the same wall; slice rather than fight the conflict.

Other live non-golf work: `src/shared/python` hygiene (#4495 upstreaming
orphaned consumer edits, #4507 lint normalisation, #4509 mypy debt), CI repairs
(#4454 merge-hold guard, #4469 architecture guards always-on, #4504 codemap
session skip, #4506 hook allowlist), three dependabot bumps, and five
Bolt/Palette micro-PRs.

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

Note: `ruff format --check` currently reports four pre-existing failures under
`src/data_processing/data_processor/python/tests/`. They are on `main` and are
not caused by your diff; do not silently absorb them into an unrelated PR.

## Do-Not List

- **Do not reopen-and-rebase the closed `codex/4142-*` / `codex/4433-*`
  branches.** They are superseded by merged #4473 and have diverged far enough
  that merging one would remove current `main` content.
- **Do not add per-file Python version guards.** `conftest.py` reads each
  package's declared `requires-python` and skips collection below it; see the
  Python floor section in `CLAUDE.md`. Adding `sys.version_info` checks or
  `importorskip` calls per module is the whack-a-mole this replaced.
- **Do not append a dated entry to this file.** It grew to 2,708 lines that
  way, 18x over the policy limit. Put history in the commit message.
- Do not regenerate the sidekick API baseline
  (`tests/sidekick_api_baseline.json`) with `--regenerate-api-baseline`. It
  blesses whatever drift currently exists rather than only your change. Hand-edit
  the baseline and coordinate a downstream migration.
- Do not modify public function signatures in `src/shared/python/**` without
  opening coordinated migration issues in UpstreamDrift and Gasification_Model
  (see `CLAUDE.md` Cross-Repo Dependencies).
- Do not edit the shared Python surface from inside a consumer's vendored copy;
  Tools is the source of truth. #4495 exists because three fixes were orphaned
  that way.
- Do not import across package boundaries (e.g. `signal_processing_studio`
  importing from `sidekick.process_calculators`) — LoD is enforced.
- Do not exceed the 500-LOC file budget on new/modified files in the golf-sim
  packages (`rate_of_closure`, `swing_sim`, `swing-core`) — sub-package,
  don't grow monoliths. The repo-wide protected budget is 1,200 lines.
- Do not use `git commit --no-verify` / `--push --no-verify` to bypass hooks;
  see `CLAUDE.md` Hook bypass policy.
- Do not assume delta CI covers you. Changed-file selection can hide
  accumulated whole-tree debt on both the lint and type gates; a PR that touches
  enough shared files will surface it all at once.
- Do not hand-roll a GitHub Pages deploy workflow for `rate_of_closure/web` —
  Phase 7 of #4103 owns this.

## Short-Term Roadmap (ordered)

1. Land the consolidations as **slices against current `main`**, not merges —
   #4517 is the pattern (`swing_sim.ground` out of #4466). Check each slice for
   overlap with what #4473 already merged before resolving anything.
2. Clear the `src/shared/python` hygiene trio (#4495, #4507, #4509) — these
   unblock downstream consumers.
3. Land the CI repairs (#4454, #4469, #4504, #4506).
4. Start #4130 Phase F1 (formulation document).
5. Phase 7 of #4103: WASM swap for the web mirror + real Pages CI deploy.
6. #4125 H5: stand up the public release-management repo (cross-repo).
