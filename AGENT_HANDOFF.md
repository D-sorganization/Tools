# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-23

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

| Epic  | Status (one line)                                                                                                                                                                                                                      |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #4103 | Swing-Impact-Ball-Flight platform. Open. Stack PR #4119 closed; content landed in slices. Remaining: camera cluster (#4571) and Phase 7 (WASM web parity, Pages CI).                                                                   |
| #4120 | Investigation & Variation Suite. Open. PR #4124 **merged**.                                                                                                                                                                            |
| #4125 | Realistic clubs / kinetics / putting / showcase. Open. PR #4129 **merged**. H5 (public release-management repo) still not started.                                                                                                     |
| #4130 | Impact-interval club dynamics. **COMPLETED** (F1–F4 landed in PR #4577) — 6-DOF transient package, tests, and impact wire.                                                                                                             |
| #4142 | Ensemble variation, quiet zones, sensitivity attribution. Open. #4628/#4635/#4646 merged durable R11.5 execution and visual correction; #4626 owns current-main qualification before the evidence audit and UpstreamDrift pin.         |
| #4146 | Shared Club Builder. Open. Assembly physics contracts landed in #4157.                                                                                                                                                                 |
| #4433 | Visual-first tab visibility and visualization-led UX. Open. Core authority landed via **#4473**; current-main React evidence passes, while #4626 must restore the PyQt and distribution qualifications before acceptance adjudication. |
| #4430 | Qualified rotating-base companion. **COMPLETED** via #4618/#4619; UpstreamDrift consumed the immutable provider through #8954.                                                                                                         |
| #4549 | Club Fitting Tester (OEM-grade). **COMPLETED** (#4557, #4577) — C1–C7 delivered (mesh inertia, shaft delivery, OEM doc, counterfactuals, PyQt6/React GUI tabs).                                                                        |
| #4562 | Heavy Hit - hand/body coupling at impact. **COMPLETED** (#4568, #4577) — H1–H4 delivered (coupled mechanics, MJCF/URDF/.osim import, GUI readout).                                                                                     |
| #4583 | Professional launch-monitor program. Release A and source-backed SG are merged; #4603 adds canonical dataset/covariation consumers through the ordinary protected flow. Release B physical collection remains external and open.       |

Per-tool detail: `src/rate_of_closure/AGENT_HANDOFF.md`, `src/pendulum_simulator/AGENT_HANDOFF.md`, and `src/rotation_converter/AGENT_HANDOFF.md`.

## Open PR Situation — Read Before Filing Anything

No PR is a draft. The live golf queue includes Sidekick and remaining #4142 work; query `gh pr list` rather than trusting a count here.

| PR    | Scope                                                                      |
| ----- | -------------------------------------------------------------------------- |
| #4585 | Sidekick Phase S1/S2 integration; active protected delivery                |
| #4600 | Post-merge PyQt launch-monitor visual-baseline approval                    |
| #4649 | Serialize trusted Playwright apt access; reconcile #4142/#4433 handoffs    |
| #4466 | Rate of Closure remainder — **content-complete except the camera cluster** |
| #4449 | P1AM plant historian + SCADA foundation (supersedes #4065, #4091)          |
| #4447 | Variation / Morris sensitivity suite (consolidates 34 drafts)              |
| #4446 | Ground study + rate-of-closure suites (supersedes #4409/#4410)             |

**#4466 still cannot be merged by any strategy** — its merge-base predates
`src/rate_of_closure/`, making 281 overlapping files both-added conflicts. Its
content landed as 22 slices (#4517–#4547). **Only the camera-controls cluster
remains, as a reimplementation**: wiring `CameraViewportMixin` passes 20 camera
tests but regresses three main-owned tests and needs ~20 UI files that delete
shipped work. #4571 owns this; do not slice it or close #4466 before it lands.

The 39 `codex/4142-*` / `codex/4433-*` drafts were superseded by merged #4473
(Morris under `application/morris/`, #4433 inspectors under `ui/pyqt6/`).

Other live non-golf work: `src/shared/python` hygiene (#4507 lint normalisation, #4509 mypy debt), CI repairs (#4454 merge-hold guard, #4469 architecture guards always-on), and eight Bolt/Palette micro-PRs.

**Current #4142 evidence state.** R11.5 durable execution, bounded analysis,
stable-paint capture, and the registered three-viewport correction are merged
through #4628/#4635/#4646; #4649 merged the apt mutex. #4657 merged PyQt
isolation and bounded dirty-tree diagnostics. #4658 normalized tracked release
evidence; #4659 then protected-merged exact Inter 5.3.0 visual fonts as current
remote `main` `26d144aa745e81c3fb75d5d196d484449059b491`. Trusted run
`32682451878` passed React and all 18 PyQt renders; comparison correctly stopped
at the newly versioned React environment. Direct PNG analysis disproved the
earlier apparent shell omission, which was a multi-image viewer artifact.
The PyQt candidate exposed real global-QSettings leakage and an under-specified
environment identity. Branch `fix/4626-pyqt-evidence-isolation` routes state to
fresh INI storage and records Qt/PyQt/Matplotlib/font versions. Require protected
evidence and separate source-pinned baseline approval before R10-R15/#4433 audit.
**Known-red on `main`, already filed — do not re-diagnose:** #4582 (the Phase 0
branch isolates the benchmark from inconsistent self-hosted pip), #4561 (browser
qualification: companion readiness metadata + missing Firefox/WebKit binaries;
the workflow is scoped to Chromium meanwhile), #4569 (two `1.17.40` SPEC rows
glued onto one table line by a union merge), #4558/#4559/#4560 (cross-runtime
fixture parity gaps). #4602/#4608/#4610 isolate trusted React/PyQt evidence;
#4607 removes the blocking npm-cache post-hook, and #4613 owns deterministic Qt probe teardown after complete rendered evidence generation.

## Must-Read Architecture Pointers

1. `CLAUDE.md` — conventions, CI gates, and cross-repo dependency rules; Tools is a leaf consumed by UpstreamDrift and Gasification_Model.
2. `docs/architecture/CANONICAL_TOPOLOGY.md` — canonical repo topology policy.
3. `SPEC.md` — §12 requires a dated row for every PR touching `src/`.
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

1. **Sidekick Unified Integration**: implement the S1–S5 plan in
   `docs/development/epic_sidekick_unified_impact_model_and_launcher_integration.md`.
2. Restore the isolated advisory benchmark lane through #4582.
3. **Land the camera-cluster epic #4571** so #4466 can finally close.
4. Merge #4626, require passing trusted current-main React/PyQt/distribution evidence, complete the #4142/#4433 audit, and pin the immutable Tools commit in UpstreamDrift; #4430 is complete.
5. Phase 7 of #4103: WASM swap for the web mirror + real Pages CI deploy.
6. #4125 H5: stand up the public release-management repo (cross-repo).
7. Approve #4600's inspected post-merge PyQt launch-monitor visual reference.
