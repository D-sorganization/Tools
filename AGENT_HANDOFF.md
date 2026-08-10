# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-09.

## Active Rate of Closure campaign

The current bounded continuation is
`C:\Users\diete\Repositories\Tools-worktrees\four-surface-capability` on
draft PR #4299 branch `feat/4264-four-surface-capability`, starting from exact
published head `dd58ff4113c2489d4af7be66c72ec9f58217a1d5`. Local declared-scope
commit `a7cc4e005bb2263abe55ba93142e2fb59d26657e` is normally merged with
exact current carrier `bb101cedd555d07d493aae998b46050c68660cdd`, preserving
the branch base. The continuation expands `four-surface-capability/v1` from
six curated records to the deterministic declared scope: all 15 campaign
programs, all 18 unique linked active release specifications, and the six
curated evidence-backed capabilities. There is no continuation push, hosted
CI, installed consumer pin, protected merge, or release claim.

Local evidence for the composed branch: 22 capability-contract tests plus eight
campaign-manifest tests pass; 50 cited Python/PyQt workflow, workspace, and
export tests pass; and three cited React files / 32 tests pass. Deterministic
CLI validation/schema/canonical output, Python 3.10 import/schema parity,
exact mypy 1.13, Ruff, Ruff format, Black, assertion policy, docs governance,
and diff checks pass.

The parent source checkout remains
`C:\Users\diete\Repositories\Tools-worktrees\toolstrip-workspace` on the
existing PR #4282 carrier `feat/4199-wind-workflow`. The local continuation
normally incorporates its exact head
`bb101cedd555d07d493aae998b46050c68660cdd`, which itself incorporates exact
corrected #4281 parent `958770049f0124dac0426a6dd62fd4edbf437e7a`
without changing the branch base `feat/4199-wind-scalar-adapter`. It composes
four reviewed slices:
strict capability parsing and signed decimal entry; complete capability
diagnostics/result exports and quantitative React scatter scales; package-safe
static-web release entrypoints; and the strict `rate-of-closure-campaign/v1`
release-evidence authority. The normal merge also carries the parent's
Python 3.10 compatibility, variation-export, and scalar-ensemble corrections.
No branch was rebased, retargeted, force-pushed, or published by this
continuation; hosted CI remains due.

Canonical files:

- `docs/release/four_surface_capability.v1.json` — declared-scope matrix;
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
is the current open carrier on base `feat/4199-wind-scalar-adapter`. Exact
corrected parent `958770049f0124dac0426a6dd62fd4edbf437e7a` is incorporated
through the normal local merge containing this handoff.

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
- #4260/#4267: the strict four-surface matrix is complete for its governed
  campaign-program/linked-spec boundary, but narrative features outside that
  structured registry, immutable installed UpstreamDrift consumers, truthful
  React routing, conformance runs, and protected release remain; ground
  contracts/transfer are partial and bounce, skid, roll, total distance, UI,
  and parity remain open.

## Required validation

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python scripts/four_surface_capability.py
python scripts/four_surface_capability.py --declared-scope
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
