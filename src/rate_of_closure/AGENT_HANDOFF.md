# AGENT_HANDOFF — rate_of_closure

> Update with every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-09.

## Issue #4284 local implementation

Draft PR #4298 publishes branch `feat/4284-camera-snap-tracking` with tested
camera evidence through immutable commit
`2095e748ddca2d7036bbd49a731528f5634daff9`, normally merged from exact carrier
`de49580a3c0888b44f66dcc09bba2ab2fa33914a` and camera implementation parent
`91bdd77df4dfc3abe79d9d22108b31d07a7cfbe3`. It has a shared Python/TypeScript
camera contract and adapters for PyQt6 Simulation/Flight and React Club,
Impact, and Flight 3D viewports. Canonical snap directions use x downrange,
y up, z right; face-on side is explicit. Tracking is opt-in, bounded, isolated
per viewport, preserves safe zoom, suspends after manual orbit, and resumes on
Recenter. Camera state is deliberately not persisted in this slice.

The published evidence adds adjacent solver-sample frame stepping to
React flight playback and Playwright coverage spanning play/pause/restart,
loop, speed, frame steps, wheel zoom, snap views, tracking suspension, and
Recenter. It also verifies control containment and canvas backing resolution
at 520 x 900 and 2x DPR. Do not report released: required follow-up is native
rendered review, hosted CI/review, normal stack integration, and UpstreamDrift
PyQt6/React consumer parity. See
`docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md` and issue #4284.

Evidence commit `2095e748` passes 39 focused Python/PyQt camera tests, the full
107-file / 650-test React suite, four Playwright browser tests across desktop
and constrained 2x-DPR Chromium, Ruff format/check, targeted mypy, TypeScript,
zero-warning ESLint, the 193-module Vite build, campaign validation, and diff
checks. Headless desktop and 700 px camera-control renders were inspected
without overlap; the local offscreen Qt font directory is unavailable, so
native-font/browser rendered review remains explicitly open. Browser
automation does not close the native visual or downstream parity gates.

This documentation-only successor records the already-published evidence
commit. `evidence_commit_sha` is immutable evidence, not an impossible
self-reference to the successor's own future SHA.

## Current continuation

The active local continuation is integrated directly on the existing PR #4282
carrier `feat/4199-wind-workflow`, starting from exact published head
`18fe89201d657116bbca99922297c14968356c44`. It adds strict cross-runtime
capability parsing, reliable signed decimal entry, complete ranked diagnostics
and result exports, quantitative React scatter annotations, package-safe
static-web release entrypoints, and the strict `rate-of-closure-campaign/v1`
release authority. It remains local until final gates pass and a normal
fast-forward push is made.

The direct web launcher dynamically loads the root bootstrap module without a
launcher-local `sys.path` mutation; its subprocess delegation test and the
changed-production-Python policy guard cover this release entrypoint.

Use these authorities together:

- `docs/release/rate_of_closure_campaign.v1.json` — machine state;
- `scripts/rate_campaign_manifest.py` — validation and generated JSON Schema;
- `docs/release/RATE_OF_CLOSURE_CAMPAIGN_MANIFEST.md` — update procedure;
- `docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md` — detailed history;
- `docs/specs/*.md` — scientific and interoperability contracts.

The manifest is intentionally normalized: programs reference shared carrier
and test-evidence records. It fails on missing primary issues, undeclared
references, duplicate IDs, malformed SHAs, absent repository evidence,
placeholders, and contradictory release claims.

## Carrier state

The capability stack was collapsed top-down on 2026-08-09:

1. #4294 Shot Optimizer UI merged into `feat/4197-capability-flight-evaluator`;
2. #4289 evaluator merged into `feat/4197-capability-observer`;
3. #4283 observer merged into `feat/4199-wind-workflow`.

These were feature-parent merges, not protected releases. Current carrier
#4282 remains upstream of #4281 → #4280 → #4279 → #4203 → #4202 → wedge and
variation parents. Preserve that dependency order and use normal merges only.

Outer platform PR #4119 targets `main` and still needs reconciliation. PR #4133
impact-interval dynamics merged into a historical feature parent after that
parent had already propagated, so its source and tests must be explicitly
reconciled before #4130 can close.

## Implemented on the feature stack

- PyQt6 and React variation workspaces: selectable input/contact/impact/shot
  scatter, matrix, landing dispersion, every-trial 3D swing arcs, quiet zones,
  linked selection, bounded raw rows, and lossless exports.
- Matched toolstrip/workspace controls, module visibility, independent plots,
  legend placement, zoom/autofit, replay, loop, and granular playback rate.
- Launch-monitor convention, D-plane, spatial target, inverse flight, wind,
  interactive flight playback, and capability-optimization contracts/workflows.
- Wedge kinematics, clearance, turf proxy, impact visualization, and
  forgiveness-analysis slices.

## Current limitations

- No complete campaign program has a qualified `main` release SHA.
- Capability v1 is still-air carry to first ground crossing; it excludes wind,
  bounce, roll, and total distance in that evaluator.
- Ground contracts and flight transfer are open carriers; qualified bounce,
  skid, roll, profiles, total-distance optimization, UI, and Rust/WASM parity
  remain.
- UpstreamDrift React has no native Rate route, and the Tools source pin and
  resolver behavior are not a qualified immutable parity boundary.
- Camera tracking/snap views (#4284), complete persistence adapters, high-DPI,
  keyboard, reduced-motion, and visual-regression matrices remain open.
- Local/rendered evidence is not protected CI or installed-package evidence.

## Most recent evidence

- Composed local continuation: 828 Rate Python/PyQt tests and 104 React files /
  642 tests passed; TypeScript, zero-warning ESLint, and the 188-module Vite
  production build passed. The manifest, schema JSON, Ruff, targeted mypy, and
  nine manifest/parity contracts pass on implementation head `2c1a77baa`.
  Hosted CI has not run on this composition.
- Hosted quality-gate run `31340032608` subsequently found mypy 1.13
  `no-any-return` findings at the Pydantic loader and Qt elapsed-timer adapter
  under `--follow-imports=skip`; both boundaries now narrow their return
  values explicitly. The exact Python 3.12/mypy 1.13 delta is clean across 54
  files; Ruff, 62 focused regression tests, and eight campaign-manifest tests
  also pass locally.
- Consolidated capability head `c1827bbdc`: 1,426 Python Rate/shared-swing
  tests and 624 React tests passed locally.
- Variation continuation `d71b0ea01`: 890 Python/PyQt/shared-swing tests and
  545 React tests passed locally, with one optional Rust skip.
- Toolstrip head `c36ca36e9`: the same 890/545 local suite totals passed.
- Hosted #4282 head `3186a265b1`: `swing_core` built, but cached plugin loading
  failed before parity collection; this is failed evidence, not a model pass.
- CI-isolation head `18fe89201`: focused workflow regression passes locally;
  the hosted non-skipped parity run is still required.

## Validation

```powershell
python scripts/rate_campaign_manifest.py
python -m pytest tests/rate_of_closure/test_campaign_release_manifest.py -q
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -m pytest tests/rate_of_closure src/shared/python/swing_sim -q
cd src/rate_of_closure/web
npm test -- --run
npm run type-check
npm run lint
npm run build
```

Run Ruff format/check and pinned delta mypy on every changed Python file. Do not
rewrite parents, force-push, bypass checks, infer unavailable values, or close
#4201 until exact protected merge, installed package, downstream pin, science,
performance, accessibility, documentation, and rollback evidence all exist.
