# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-09.

## Issue #4300 moving-subject default-camera continuation

Branch `fix/rate-pyqt-default-camera` in worktree
`C:\Users\diete\Repositories\Tools-worktrees\pyqt-default-camera` is a normal
child of exact #4301 head `9322df75d6ad1b6ef57be02741ac972e7c6f86cf`.
The shared Python and TypeScript camera authorities now provide a DRY
share-ready moving-subject state: 2x initial zoom, bounded tracking enabled,
and Auto Fit enabled. PyQt6 Swing/Flight and React Club/Flight consume that
state; static viewports keep the neutral default. Users can disable tracking
or Auto Fit, manual orbit still suspends tracking, and re-center restores it.
No physics, trajectory, geometry, frame, or result schema changed.

RED-first PyQt and React tests prove both moving views start checked, fitted,
and tracking while retaining the snap/orbit/recenter workflow. Validation is
49 focused Python/PyQt tests, 108 React files / 653 tests, TypeScript,
zero-warning ESLint, production build, six desktop/constrained-HiDPI Chrome
Playwright cases, pinned MyPy 1.13 on four production files, Ruff, and diff
checks. Native inspection of the exact worktree shows the Swing subject at a
useful scale and in-frame during playback. Current-head manifest evidence,
draft publication, protected CI, review, and ordinary integration remain
required; #4300 and epic #4218 stay open.

## Issue #4300 constrained Tools-menu continuation

Draft PR #4301 publishes branch `fix/rate-mobile-tools-menu` from worktree
`C:\Users\diete\Repositories\Tools-worktrees\mobile-tools-menu` at immutable
evidence commit `ebd804ff24e7ce5ca58c7d1495c438ab1dcd83b5`. It targets exact camera
carrier `42753a576f42d4c43c35fd786d0748e1d03672c5` on
`feat/4284-camera-snap-tracking`. The fix
keeps all File/View/Tools popovers inside a 16 px viewport gutter through one
shared horizontal clamp, bounds menu width on smaller screens, and preserves
unchanged desktop positioning plus native `<details>/<summary>` keyboard and
accessibility semantics.

RED evidence at 520 x 900 measured the opened Tools menu right edge at
622.48 px against a 504 px contract limit. GREEN evidence is 108 Vitest files
with 653 tests, six Playwright cases across desktop and constrained 2x-DPR
projects, TypeScript checking, zero-warning ESLint, a 194-module production
build, the campaign-manifest validator, 11 manifest tests, and `git diff
--check`. The browser contract keyboard-opens Tools and verifies all four
labels, Alt+G, Alt+T, and F1 remain readable and contained. Track this bounded
child under epic #4218 and issue #4300; draft PR evidence is not protected
integration.

## Issue #4284 camera continuation

Draft PR #4298 publishes branch `feat/4284-camera-snap-tracking` with tested
camera evidence through immutable commit
`2095e748ddca2d7036bbd49a731528f5634daff9`. The current local restack normally
merges exact published #4282 carrier
`bb101cedd555d07d493aae998b46050c68660cdd` into exact camera branch parent
`7f1e14d42ffe8c23856a12fc8b0d0a8a4eeaf092`; the PR base remains
`feat/4199-wind-workflow`. It implements the
shared, UI-neutral camera command contract in Tools PyQt6 and React swing,
impact, and flight 3D views: exact Face On/Down the Line/Overhead/Isometric
snaps, opt-in bounded subject tracking, zoom-preserving Auto Fit, predictable
manual suspension, and one-action Recenter. The published evidence also adds
solver-sample frame stepping plus real-browser Playwright coverage for a
bounded playback/camera interaction matrix and a 520 x 900, 2x-DPR viewport.
UpstreamDrift consumers, native rendered cross-platform review, hosted CI,
review, and protected release remain open; do not close #4284 on local evidence.

Evidence commit `2095e748` passes 39 focused Python/PyQt camera tests, the full
107-file / 650-test React suite, four Playwright tests across desktop and
constrained 2x-DPR Chromium, TypeScript, zero-warning ESLint, the 193-module
production build, Ruff format/check, targeted mypy, campaign validation, and
diff checks. Headless desktop and 700 px camera-bar renders show no control
overlap; this Qt runtime lacks usable fonts, so native-font visual review
remains an integration gate. Browser automation is not a substitute for that
manual native review.

The prior documentation-only successor records the already-published camera
evidence commit. The campaign contract uses `evidence_commit_sha`, not a
self-referential current-head field. This local merge records its exact two
parents; its own future SHA is intentionally absent from the commit it creates.

The composed local merge candidate passes 65 focused camera, PyQt6,
compatibility, and campaign-manifest tests on Python 3.13; all 15 compatibility
contracts also pass on real CPython 3.10.20. The complete React suite remains
107 files / 650 tests, all four Playwright desktop/constrained-DPR cases pass,
and TypeScript, zero-warning ESLint, and the 193-module production build pass.
Canonical Ruff check/format passes all 28 changed Python files; pinned Python
3.12/mypy 1.13 passes 20 changed production modules; manifest/schema,
documentation-governance, and staged/working-tree diff checks pass.

Authorities are `docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md`,
`src/rate_of_closure/application/camera_commands.py`, and the cross-runtime
golden fixture under `web/src/model/__fixtures__/`. Every implementation commit
must keep this file and `src/rate_of_closure/AGENT_HANDOFF.md` current.

## Active Rate of Closure campaign

The active checkout is
`C:\Users\diete\Repositories\Tools-worktrees\camera-snap-tracking` on draft
PR #4298. Its local restack incorporates exact published #4282 carrier
`bb101cedd555d07d493aae998b46050c68660cdd`, whose base remains
`feat/4199-wind-scalar-adapter` and which already incorporates exact corrected
#4281 parent `958770049f0124dac0426a6dd62fd4edbf437e7a`. It composes
four reviewed slices:
strict capability parsing and signed decimal entry; complete capability
diagnostics/result exports and quantitative React scatter scales; package-safe
static-web release entrypoints; and the strict `rate-of-closure-campaign/v1`
release-evidence authority. The normal restack merge also carries the parent's
Python 3.10 compatibility, variation-export, and scalar-ensemble corrections.
No branch was rebased, retargeted, force-pushed, or published by this local
restack; hosted CI remains due.

Canonical files:

- `docs/release/rate_of_closure_campaign.v1.json` — normalized campaign state;
- `scripts/rate_campaign_manifest.py` — schema and contradiction validator;
- `docs/release/RATE_OF_CLOSURE_CAMPAIGN_MANIFEST.md` — maintenance procedure;
- `src/rate_of_closure/AGENT_HANDOFF.md` — current Rate-specific handoff;
- `docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md` — historical detail.

The manifest deliberately reports the campaign as **not released**. It
distinguishes specified scope, feature-stack implementation, protected parent
merge, and exact `main` release. A local pass, a feature-branch merge, and a
default-branch release are never interchangeable.

## Current carrier and release state

Capability PRs #4294, #4289, and #4283 were merged top-down into
`feat/4199-wind-workflow`; their feature parents were not protected. PR #4282
is the current open carrier on base `feat/4199-wind-scalar-adapter`. Its exact
published head `bb101cedd555d07d493aae998b46050c68660cdd` incorporates corrected
parent `958770049f0124dac0426a6dd62fd4edbf437e7a`; #4298 now incorporates that
published carrier through a normal local merge.

The previous exact #4282 head `3186a265b1` built and loaded `swing_core`, but
its Python 3.11 parity job failed before collection because a cached
`pytest-qt` plugin loaded without PyQt6. Commit `18fe89201` disables third-party
pytest plugin auto-loading only in that Rust parity step. Its focused workflow
contract test passes; a fresh hosted parity run is still required.

Combined local evidence on the composed continuation is 828 Rate Python/PyQt
tests and 104 React files / 642 tests, plus TypeScript, zero-warning ESLint,
and the 188-module Vite production build. The deterministic manifest,
generated-schema JSON, Ruff, targeted mypy, and nine manifest/parity contracts
also pass on implementation head `2c1a77baa`.

Hosted quality-gate run `31340032608` reached mypy 1.13 and exposed
CI-context-only `no-any-return` findings in the Pydantic manifest loader and
Qt elapsed-timer adapter because the delta lane uses `--follow-imports=skip`.
Both boundaries now narrow their return values explicitly. The exact Python
3.12/mypy 1.13 delta is clean across 54 files; Ruff passes and 62 focused
regression tests plus eight campaign-manifest tests pass (only pre-existing
optional-plugin config warnings).

Corrected-parent propagation evidence is 62 focused wind/scalar/variation and
compatibility tests on both Python 3.11 and real CPython 3.10.20, plus 8 React
files / 35 tests, TypeScript, and focused zero-warning ESLint. The Python 3.10
run exposed one child-owned direct `enum.StrEnum` import in capability
observations; it now uses the shared runtime compatibility contract and is
included in the source-level regression. Ruff check/format passes 15 focused
files, pinned mypy 1.13 passes 10 production modules, and all nine campaign
manifest/parity contracts pass.

The direct web launcher dynamically loads the root bootstrap through
`importlib` instead of mutating `sys.path` in the changed entrypoint. Its real
child-process delegation test and the changed-Python policy guard cover that
release path.

Only `main` is the release boundary. It requires `quality-gate` and
`tests (3.11)`. Outer PR #4119 remains the main-targeting platform carrier and
requires current-main reconciliation. Impact-interval PR #4133 merged after
its parent stack had already propagated and is not proven in #4119.

## Critical open programs

- #4142/#4144: variation graphics are substantial, but global sensitivity,
  bounded performance, downstream pinning, and release remain.
- #4146: club-builder core exists; manufacturing export, image fitting, full
  client workflow, and downstream qualification remain.
- #4158: wedge kinematics, clearance, turf, visualization, and forgiveness are
  feature-stack implementations without a protected release.
- #4191/#4201: wind, inverse flight, playback, and capability work exist on the
  stack; scientific, installed-package, and cross-surface release gates remain.
- #4218/#4234: toolstrip and selected layout fixes exist; camera issue #4284,
  complete persistence, high-DPI, keyboard, and visual baselines remain.
- #4260/#4267: four-surface parity is specified; ground contracts/transfer are
  partial and bounce, skid, roll, total distance, UI, and parity remain open.

## Required validation

```powershell
python scripts/rate_campaign_manifest.py
python -m pytest tests/rate_of_closure/test_campaign_release_manifest.py -q
python -m pytest tests/ops/test_maturin_swing_core_workflow.py -q
python -m ruff check scripts/rate_campaign_manifest.py `
  tests/rate_of_closure/test_campaign_release_manifest.py
python -m ruff format --check scripts/rate_campaign_manifest.py `
  tests/rate_of_closure/test_campaign_release_manifest.py
python -m mypy --ignore-missing-imports scripts/rate_campaign_manifest.py
```

For a feature change, also run the complete affected Rate Python/PyQt/shared
swing and React suites recorded in the per-tool handoff.

## Do not

- Do not mark a program released without a 40-character `main` merge SHA.
- Do not convert queued, cancelled, skipped, or failed checks into passing
  evidence.
- Do not retarget or rewrite the protected stack, force-push, or admin-merge.
- Do not duplicate carrier or test metadata in individual program records.
- Do not treat an UpstreamDrift launcher tile as four-surface parity.

Fleet handoff policy is tracked by Repository_Management #1393/#1397. Any
`src/**` handoff change also requires the same-commit `SPEC.md` update.
