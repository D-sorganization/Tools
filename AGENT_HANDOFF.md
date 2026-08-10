# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-09.

## Active Rate of Closure campaign

The current bounded local continuation is
`C:\Users\diete\Repositories\Tools-worktrees\four-surface-capability` on
`feat/4264-four-surface-capability`, based exactly on carrier
`feat/4199-wind-workflow@de49580a3c0888b44f66dcc09bba2ab2fa33914a`.
It adds the strict `four-surface-capability/v1` partial inventory, deterministic
schema/canonical JSON generation, exact Tools source pin, evidence-path and
freshness validation, and explicit unsupported-with-reason UpstreamDrift
consumer states. It is local only: there is no push, PR, hosted CI, installed
consumer pin, protected merge, or release claim.

Local evidence for this slice: 15 capability-contract tests plus eight
campaign-manifest tests pass; 50 cited Python/PyQt workflow, workspace, and
export tests pass; and three cited React files / 32 tests pass. Deterministic
CLI validation/schema/canonical output, Python 3.10 import/schema parity,
exact mypy 1.13, Ruff, Ruff format, Black, assertion policy, docs governance,
and diff checks pass.

The parent source checkout remains
`C:\Users\diete\Repositories\Tools-worktrees\toolstrip-workspace` on the
existing PR #4282 carrier `feat/4199-wind-workflow`; this continuation starts
from its exact supplied carrier head
`de49580a3c0888b44f66dcc09bba2ab2fa33914a`. The inherited carrier composes
four reviewed slices:
strict capability parsing and signed decimal entry; complete capability
diagnostics/result exports and quantitative React scatter scales; package-safe
static-web release entrypoints; and the strict `rate-of-closure-campaign/v1`
release-evidence authority. The #4264 continuation has not been pushed; hosted
CI remains due.

Canonical files:

- `docs/release/four_surface_capability.v1.json` — partial four-surface matrix;
- `docs/release/four_surface_capability.v1.schema.json` — generated schema;
- `scripts/four_surface_capability.py` — generator and fail-closed validator;
- `docs/release/FOUR_SURFACE_CAPABILITY.md` — maintenance procedure;
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
- #4260/#4267: the first strict partial four-surface matrix exists, but the
  exhaustive inventory, immutable installed UpstreamDrift consumers, truthful
  React route, conformance runs, and protected release remain; ground
  contracts/transfer are partial and bounce, skid, roll, total distance, UI,
  and parity remain open.

## Required validation

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python scripts/four_surface_capability.py
python -m pytest tests/rate_of_closure/test_four_surface_capability.py -q
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
