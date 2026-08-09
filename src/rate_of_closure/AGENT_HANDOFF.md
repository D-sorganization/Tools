# AGENT_HANDOFF — rate_of_closure

> Update with every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-09.

## Issue #4284 local implementation

Branch `feat/4284-camera-snap-tracking`, based on exact stack head
`a742ad6cc2853b170eb945c4d74a56ea23bdda33`, now has a shared Python/TypeScript
camera contract and adapters for PyQt6 Simulation/Flight and React Club,
Impact, and Flight 3D viewports. Canonical snap directions use x downrange,
y up, z right; face-on side is explicit. Tracking is opt-in, bounded, isolated
per viewport, preserves safe zoom, suspends after manual orbit, and resumes on
Recenter. Camera state is deliberately not persisted in this slice.

Do not report released: the branch remains local with no PR/carrier. Required
follow-up is rendered desktop/constrained-browser review, hosted CI/review,
normal stack integration, and UpstreamDrift PyQt6/React consumer parity. See
`docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md` and issue #4284.

Local gates: 83 affected Python/PyQt tests; 107 React files / 649 tests; Ruff
format/check; targeted mypy; TypeScript; zero-warning ESLint; 193-module Vite
build; campaign-manifest, diff, and changed-file structural checks. Headless
desktop and 700 px camera-control renders were inspected without overlap; the
local offscreen Qt font directory is unavailable, so native-font/browser
rendered review remains explicitly open.

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
