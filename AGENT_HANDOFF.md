# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-09.

## Issue #4284 camera continuation

Isolated branch `feat/4284-camera-snap-tracking` normally merges exact carrier
head `de49580a3c0888b44f66dcc09bba2ab2fa33914a` with camera implementation
parent `91bdd77df4dfc3abe79d9d22108b31d07a7cfbe3`. It locally implements the
shared, UI-neutral camera command contract in Tools PyQt6 and React swing,
impact, and flight 3D views: exact Face On/Down the Line/Overhead/Isometric
snaps, opt-in bounded subject tracking, zoom-preserving Auto Fit, predictable
manual suspension, and one-action Recenter. No GitHub write or carrier exists
yet. UpstreamDrift consumers, rendered cross-platform review, hosted CI,
review, and protected release remain open; do not close #4284 on local evidence.

Exact camera-parent evidence is 107 React files / 649 tests. On the reconciled
merge tree, 138 affected Python/PyQt/carrier tests and 14 focused React camera
tests pass, plus Ruff format/check, camera-targeted mypy,
TypeScript, zero-warning ESLint, the 193-module production build, campaign
manifest validation, and diff/structural checks. Headless desktop and 700 px
camera-bar renders show no control overlap; this Qt runtime lacks usable fonts,
so native-font and browser visual review remains an integration gate.

Authorities are `docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md`,
`src/rate_of_closure/application/camera_commands.py`, and the cross-runtime
golden fixture under `web/src/model/__fixtures__/`. Every implementation commit
must keep this file and `src/rate_of_closure/AGENT_HANDOFF.md` current.

## Active Rate of Closure campaign

The active checkout is
`C:\Users\diete\Repositories\Tools-worktrees\toolstrip-workspace` on the
existing PR #4282 carrier `feat/4199-wind-workflow`. The local continuation
starts from exact published head
`18fe89201d657116bbca99922297c14968356c44` and composes four reviewed slices:
strict capability parsing and signed decimal entry; complete capability
diagnostics/result exports and quantitative React scatter scales; package-safe
static-web release entrypoints; and the strict `rate-of-closure-campaign/v1`
release-evidence authority. It has not yet been pushed; hosted CI remains due.

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
is the current open carrier on base `feat/4199-wind-scalar-adapter`.

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
