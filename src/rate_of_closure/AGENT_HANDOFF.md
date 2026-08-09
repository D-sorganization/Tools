# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-09

## 2026-08-09 Capability workflow contract and signed-input hardening

The local branch `feat/4201-capability-contract-hardening`, based on exact
carrier `18fe89201d657116bbca99922297c14968356c44`, now validates every numeric
and text primitive in `capability-optimization-workflow/v1` before Python model
construction. Python and TypeScript consume the same versioned parser-case
fixture, including accepted integral JSON numbers and rejected numeric strings,
booleans, fractional integer fields, and non-string identifiers/provenance.
This prevents cross-runtime acceptance drift and removes silent Python
`float(...)` / `str(...)` normalization from the public workflow boundary.

React optimizer numeric fields use a decimal text draft with explicit commit
semantics. Negative values can be entered by ordinary keyboard interaction;
the regression uses `userEvent` to clear, type `-3.5`, blur, run, and assert the
worker receives `-3.5`.

RED: Python `8 failed, 16 passed`; React `2 failed, 26 passed`. GREEN: the four
focused Python capability modules report `43 passed`, and the nine focused
React capability files report `69 passed`. Ruff check/format, focused mypy,
`tsc --noEmit`, zero-warning ESLint, production structural budgets, and diff
whitespace checks pass. The Vite production build passes with 187 modules and
no size warning. The change is local only; no push or PR mutation has occurred.

## 2026-08-09 Capability result diagnostics and engineering scatter

The isolated `feat/4201-capability-results-diagnostics` branch starts at
carrier `18fe89201`. Matched PyQt6/React alternative tables now expose the
complete retained diagnostic contract rather than only carry and dispersion:
objective score, miss CVaR, downside carry, complete/no-impact/failed counts,
failure fraction, confidence, extrapolation, limiting constraints, Pareto
status, and parameter units. Result-specific CSV and versioned JSON exports
retain those values separately from the existing lossless observation exports.

The shared React scalar scatter now renders quantitative grid/tick labels,
publishes the visible numeric ranges, and identifies complete/no-impact/failed
cohorts with paired-finite counts. The PyQt Matplotlib view already provided
numeric axes/grid/legend and remains the presentation reference. Keep issue
#4197 and release gate #4201 open until carrier integration, exact-head CI,
review, downstream parity, and protected release are proven.

Local verification passed 813 Rate Python/PyQt tests and 104 React files / 628
tests, plus Ruff, targeted mypy for the new export and tab modules, TypeScript,
zero-warning ESLint, and the 188-module Vite build. The unchanged carrier also
reproduces mypy 1.19.1's Python 3.13 internal serialization assertion when
checking `capability_results.py`; do not misreport that tool failure as a new
source-code type diagnostic.

## 2026-08-09 Web-release stability on the campaign carrier

The local branch `feat/4201-web-release-stability` starts from exact carrier
head `18fe89201` and makes no physics or product-layout changes. It fixes the
documented direct web script's package import, covers the exact path invocation
in a child Python process while replacing only the shared launcher delegate,
and removes the unsupported Tauri contract. Rate presently ships a static Vite
bundle for web sharing and PyQt6/PyInstaller for desktop use; there is no Rate
`src-tauri` project or qualified Tauri artifact.

The intermittent Flight Explorer suite failure was a test synchronization
defect around the real lazy Wind Strategy import. The parent-panel test now
awaits `vi.dynamicImportSettled()` inside React `act` before a synchronous role
assertion, instead of relying on a one-second polling window. The Wind Strategy
component retains its own focused tests. Integrate this local commit through
the campaign carrier after reviewing the recorded gates; do not represent it
as a Tauri implementation or a scientific-model change.

Verified locally: 813 Rate Python/PyQt tests passed with 15 existing warnings;
the launcher-focused file passed 9/9; 102 React files / 624 tests passed; and
the formerly load-sensitive Flight Explorer file passed five consecutive
focused runs. TypeScript, zero-warning ESLint, the 187-module production
build, focused Ruff check/format, targeted mypy, dependency installation/audit,
and diff checks are green. The repository-wide DRY contract still reports only
its baseline failures in two unrelated PyQt launcher files.

## 2026-08-08 Capability stack consolidated onto this branch

The three capability slices are **merged** and now live here on
`feat/4199-wind-workflow` at `c1827bbdc`. The per-slice "do not retarget,
rewrite, or merge ahead of that parent" notes in the entries below are
historical — they described the stack while it was still open, and they
no longer apply:

- #4283 capability observation stream and cancellation — merged
- #4289 model-backed capability flight evaluator — merged
- #4294 matched PyQt6/React Shot Optimizer workspaces — merged

They were folded top-down (child into its own parent, then that parent
into its parent), so no slice was released ahead of the one it depends
on. Only `main` is protected (`quality-gate` + `tests (3.11)`); none of
the `feat/4197-*` branches or this one carries protection, so these
merges were not gated by required checks. `quality-gate` passed in CI on
both #4289 and #4294 before merging.

Verified on this consolidated head: **1426 Python tests** (`tests/rate_of_closure`
plus `src/shared/python/swing_sim`, zero skips) and **624 React tests**.
The broader matrix (`tests (3.11)`, file-size-budget, detect-secrets) did
not complete because **25 of the 28 org runners were offline** — the jobs
starved rather than failed. Re-run them once the fleet is restored; that
matters before this branch approaches `main`, where protection applies.

Issue #4197 stays open: protected CI, independent review, the remaining
path to `main`, and downstream UpstreamDrift parity are all still owed.

## 2026-08-08 COMPLETION RECORD — capability-optimization-ui recovered

The interrupted optimization-UI slice was reviewed, re-verified,
repaired, and committed on `feat/4197-capability-optimization-ui`.
Recovery fixes beyond the dying agent's state: Ruff formatting and
import-sort in three files; a mypy-1.13 `call-arg` failure in
`capability_controls.py` (positional-after-star bounds unpacking
replaced with typed `_numeric_spec`/`_integer_spec` factories);
TypeScript errors in `CapabilityOptimizationPanel.test.tsx` (the mock
runner is now `vi.fn<CapabilityRunner>`); and eager panel import
bloating the main Vite chunk past 500 kB — `PrimaryWorkspacePanel` now
lazy-loads the Shot Optimizer behind `Suspense`, matching the
WindStrategyPanel precedent (main chunk 474.32 kB, no warning).

Verified gates on the committed head: 808 `tests/rate_of_closure` plus
615 swing_sim in-package tests passed with zero skips; 102 React files
/ 619 tests passed; Ruff check/format clean; CI-equivalent mypy 1.13
clean on all 10 changed src files; `tsc --noEmit`, zero-warning ESLint,
and the 187-module Vite build pass; changed-only 500-LOC budget and
`git diff --check` pass. Published as PR #4294 on
`feat/4197-capability-flight-evaluator`.

**Stack collapse order.** Only `main` carries branch protection
(`quality-gate` + `tests (3.11)`); none of `feat/4197-*` or
`feat/4199-wind-workflow` is protected, so these merges are not gated by
required checks. Each PR's base is its own parent branch, so the stack
collapses **top-down**: fold #4294 into `feat/4197-capability-flight-evaluator`,
then #4289 into `feat/4197-capability-observer`, then #4283 into
`feat/4199-wind-workflow`. Folding a child into its parent never releases
it ahead of that parent — the parent carries it forward. Merging
bottom-up instead strands each parent one slice behind and needs extra
reconciliation PRs.

The first hosted quality-gate run on #4294 rejected the dict-splat
construction of `CapabilityWorkflowInputs` (`**dict[str, float]` cannot
be proven against integer fields under the CI mypy context); commit
`101020b5b` builds the snapshot with explicit typed keyword arguments.
Do not "fix" the `tests (3.10)` `StrEnum` ImportError seen on PR #4280
by converting bare `from enum import StrEnum` to the compatibility
shim. `pyproject.toml` has declared `requires-python = ">=3.11"` since
2026-05-14, so those imports are correct; `shared.python.compatibility`
serves older subsystems only. Branch protection requires exactly
`quality-gate` and `tests (3.11)`, and `ci-standard.yml` states that
only the 3.11 lane is required (`fail-fast: false` protects it). The
3.10 lane is a stale-matrix artifact, not a merge blocker.

## 2026-08-08 Matched Capability Optimization Workspace

Active branch `feat/4197-capability-optimization-ui` is a normal child of
evaluator commit `c280407d432c153639bb266c9c721a014a129723` (draft PR #4289).
Do not retarget, rewrite, or merge it ahead of the evaluator and observation
parents.

PyQt6 and React now expose a discoverable Shot Optimizer module backed by the
same strict `capability-optimization-workflow/v1` document. It captures profile
and club IDs, capability center/spread, canonical positive-right launch and
spin-tilt frames, sourced fixed spin, target, objective, budgets, alternatives,
and seed. Both clients execute outside the UI thread, publish progress, cancel
without partial-result publication, rank alternatives, retain all observation
cohorts in `scalar-ensemble/v1`, and provide stage-qualified axis selection,
zoom/autofit, paged raw rows, lossless CSV, and stable JSON. Generic Python
scalar CSV serialization now lives in the UI-neutral variation layer.

Rendered QA in the live Vite app and a uniquely titled standalone PyQt window
verified execution, progress, large-result paging, plot controls, and layout.
It also drove fixes for hidden newly registered modules, ambiguous duplicate
axis labels, narrow split-pane geometry, and missing hover guidance. Current
verified gates: 808 Python/PyQt plus 615 swing_sim tests and 102 React files /
619 tests pass; Ruff, formatting, CI-equivalent mypy 1.13, TypeScript, ESLint,
the 187-module Vite production build, structural scans, and diff checks pass. V1 remains explicitly still-air, carry-only, and excludes
wind, bounce, roll, and total distance. Keep #4197 open for protected CI,
review, ordered merge, and downstream parity.

## 2026-08-08 Evaluator CI repair and descending-launch parity fix

Two defects were found and fixed on this branch after its first protected run,
both reported from the stacked child #4294:

1. **Descending-launch exception.** `flight.ts` starts the ball at `z = 0`, so a
   negative launch angle puts it below ground on step 1, which the `t > dt`
   guard skips. By step 2 the interpolation ratio went negative and
   extrapolated the ground crossing to a negative sample time, raising
   `RangeError: timeS must be nonnegative`. The observation layer absorbed it as
   an untyped `evaluator_exception` — 12 of 96 samples on a default driver
   search, where Python reported zero. The ratio is now clamped to `[0, 1]`, so
   these samples report `nonconverged`, matching Python exactly. A pinned
   regression test covers both reproducing samples; it fails without the clamp.
2. **Delta-mypy quality gate.** `result_derivation.py` had an unused
   `type: ignore` and an `Any` return. `_lerp_vector` now builds the 3-tuple
   explicitly (genuinely typed, no ignore needed in either import mode) and
   `_curve` coerces its result to `float`.

The other two red checks on #4289 were **not** code failures: `file-size-budget`
and `detect-secrets` both had their setup steps *cancelled* by runner
infrastructure. Re-run them rather than chasing a nonexistent violation.

Verified after the fixes: 1406 Python tests and 599 React tests pass; Ruff,
format, CI-equivalent mypy 1.13, TypeScript, zero-warning ESLint, and the
176-module Vite build pass.

## 2026-08-08 Model-Backed Capability Flight Evaluator

Active branch `feat/4197-capability-flight-evaluator` is a normal child of
`feat/4197-capability-observer` at exact parent
`49612946138b1021f80c9f8d2a4d06f1610825db` (draft PR #4283). Do not retarget,
rewrite, or merge it ahead of that parent.

The new shared Python and React-model adapters bind the real capability profile
and optimization request to the actual Waterloo/Penner flight model. They use
the established `ball_speed`, `launch_angle`, and `launch_direction` IDs,
explicit sourced per-club spin defaults for older profiles, paired optional
variable `total_spin` and `spin_axis_tilt`, positive-fade/right convention,
canonical target-frame trajectory/spin conversion, target residuals, all
available scalar metrics, typed nonconvergence, and fail-fast invariants.
Profile units, safe bounds, and physical domains are enforced before an
integration call. The post-impact evaluator never invents `no_impact`.

Independent-review corrections removed the global 2,686-rpm fallback, aligned
the canonical metric catalog with the existing app convention, unified
cross-runtime sampling validation, narrowed exception handling, and added a
shared 16-scalar parity fixture. Result, impact, and variation producers now
share the same gyro-projected tilt function. Current complete-suite evidence is
138 Python passes / four optional-Rust skips and 97 React files / 597 tests. Ruff,
formatting, targeted mypy, TypeScript, zero-warning ESLint, and the 176-module
Vite build also pass. This completes the qualified
evaluator prerequisite, not issue #4197: matched PyQt6/React authoring, worker,
progress/cancel, scatter/table/export, persistence, and rendered QA remain.

## 2026-08-07 Universal Variation Visualization Continuation

Branch `feat/4144-variation-export-continuation` is stacked on the published
toolstrip head `c36ca36e9` and continues issue #4144 without adding unrelated
toolstrip changes to PR #4279. The core PyQt6/React variation workspace already
has selectable input/contact/impact/shot scatter, a scatter matrix with
marginals, landing dispersion, and an all-trial 3D swing-arc overlay with
reference trace, principal spread, RMS variability, quiet zones, filtering,
and linked trial selection.

This continuation closes an export/accessibility parity gap: both clients now
export the complete selected scatter axes as CSV, including every trial,
typed outcome, and explicit unavailable values. PyQt now also exposes the raw
selected-axis rows in a bounded read-only table. Shared table population is
factored into `variation_trial_table.py`; the scalar scatter view is isolated
in `variation_scatter_view.py`. Changed production modules remain below 400
lines and changed functions remain at or below 50 lines.

The continuation is published as draft PR #4280. Current exact local evidence
on implementation commit `b3b09215e` is 890
Python/PyQt/shared swing tests passed with one expected optional-Rust skip and
15 existing warnings, and 89 React test files / 545 tests passed. Ruff,
formatting, Black, targeted mypy, TypeScript, zero-warning ESLint, the 166-module
Vite build, and `git diff --check` pass. This is local evidence only; no PR,
protected CI, review, merge, or default-branch release has yet been established.

Literal audit of "every many-trial simulation" found two dependent gaps that
must remain explicit:

- wind-strategy uncertainty retains paired per-strategy outcomes in both
  runtimes but has no PyQt/React workflow and does not feed the universal
  scalar plot facade;
- the capability optimizer stores aggregate statistics but discards the
  individual evaluation rows needed for honest scatter plots.

Issue #4199 already owns the wind UI/scatter requirement. Its adapter must use
composite row identity `(strategy_id, trial_index)`, retain completed,
nonconverged, and invalid cohorts, and report that impact variables are
unavailable because this runner begins at prescribed launch. Do not coerce it
into the impact-specific cohort enum or fabricate impact/landing values.

Branch `feat/4199-wind-scalar-adapter` is published as
[draft PR #4281](https://github.com/D-sorganization/Tools/pull/4281), stacked
on exact draft PR #4280 head `d71b0ea01b5659d3049ff05627c41f06481207e4`.
Implementation commit `4a28114aa` supplies that first UI-neutral model slice.
Python and React share the exact
snake-case `scalar-ensemble/v1` wire structure: structured provenance,
labeled stages/categories/cohorts, unit-bearing variables, RFC3986 composite
row identity, immutable nullable raw rows, and overall/per-cohort x/y/paired
availability. The wind adapter validates deterministic request/analysis trial
agreement, retains every actual and perfect-information status, and exposes
true/estimated wind, launch/aim, target, landing, miss, cost, and information
delta without invoking the flight solver. Impact variables remain honestly
absent because this analysis begins at launch.

Exact local gates for this branch are green: 906 Python/PyQt/shared-swing
tests passed with one expected optional-Rust skip and 15 existing warnings;
91 React files / 555 tests passed. Ruff, Ruff formatting, Black, focused mypy,
TypeScript, zero-warning ESLint, the 166-module Vite production build, and
`git diff --check` pass. Production Python modules remain below 400 lines and
no changed Python function exceeds 50 lines. This contract/adapter does not
complete #4199: the next slice still needs background execution, progress and
cancellation, PyQt/React scatter/strategy UI, persistence, and export wiring.

## 2026-08-07 Ground Model and Four-Surface Parity Revalidation

The user's requested rolling-ground and site-wide parity programs already
exist as [epic #4267](https://github.com/D-sorganization/Tools/issues/4267)
with children #4268-#4276 and
[epic #4260](https://github.com/D-sorganization/Tools/issues/4260) with
children #4261-#4266. Do not create duplicate epics. A fresh repository and
live-GitHub audit was recorded in
[the ground comment](https://github.com/D-sorganization/Tools/issues/4267#issuecomment-5222725556)
and [the parity comment](https://github.com/D-sorganization/Tools/issues/4260#issuecomment-5222726010).

There is no qualified landing/bounce/roll implementation yet. The current
airborne solvers stop at the relative launch plane and the shared
`TrajectoryPoint` omits terminal angular velocity. For a teed shot, translating
that relative trajectory into the course frame can leave the terminal ball
center at tee elevation. Complete #4268 and #4269 before implementing or
presenting bounce, roll, or total distance.

Reuse the Tools `GroundModelResult` fail-closed boundary, putting skid/roll
limiting cases, and turf provenance/variation/cancellation patterns. Adapt
UpstreamDrift's split terrain material, elevation, normal, and region contracts
through a one-way versioned DTO. Do not qualify Upstream's scalar landing
helper, heuristic putting spin relaxation, legacy duplicate terrain model, or
Rust `(1-friction)` contact law as production high-speed landing physics.

The parity matrix must keep Rate and Upstream's separate products distinct:
standalone Rate PyQt6/React, Upstream's Rate PyQt provider/React route, Shot
Tracer PyQt6/React, and the legacy ball-flight GUI. Current Upstream `main`
has no native Rate React route, and its Tools source resolvers disagree about
vendor-first versus sibling-first precedence. A launcher tile cannot satisfy a
calculation-parity row; #4261, #4262, and #4264 must make the runtime source,
exact Tools pin, and support state machine-verifiable.

## 2026-08-05 Advanced Wedge Impact Visualization

Branch `feat/4162-wedge-impact-visualization` extends issue #4162 on top of the
validated turf stack. It corrects the impact adapter to evaluate pose, twist,
and articulated wrist geometry at the exact event time rather than silently
using the nearest retained sample. The new versioned
`rate-of-closure.impact-scene/v1` contract carries complete scene geometry,
velocity components, metric equations, frames, assumptions, availability, and
screw-axis data without placing physics in either UI.

PyQt6 adds an exact-event Impact Inspector layer, locked physical axis scaling,
isometric/face-on/down-the-line cameras, and 300-DPI PNG, vector SVG, and strict
JSON export. React adds an orbitable and keyboard-controllable impact still,
the same named cameras and velocity toggles, visibly expandable engineering
metric definitions, and device-resolution PNG, true-primitive SVG, and JSON
exports. The web mirror now retains and shortest-arc interpolates the canonical
head rotation; the older limitation note saying it lacked full head pose is no
longer accurate for this branch.

Scientific boundaries remain explicit: articulated sources do not yet have an
independent torsional head state; the scene is rigid-body instantaneous
kinematics; illustrative turf profiles cannot support optimal-bounce or
forgiveness claims; and turf force is not replayed into the retained swing.

Current-head verification: all 576 Rate Python/PyQt tests passed (one existing
polynomial-generator legend warning); all 347 React/model tests passed; the
production Vite build, TypeScript, ESLint, Ruff, formatting, changed-module
strict mypy, and protected module-size budget passed. Headless Chrome visual QA
at 1600×1400 exercised named views, a vector toggle, keyboard orbit, and an
expanded metric definition with zero console exceptions/log errors. The new
web branch is running at `http://localhost:5260/`; the current PyQt process was
also launched successfully and remained responsive.

## 2026-08-05 Wedge Impact Inspector Integration

Draft PR #4173 (`feat/4163-impact-inspector`) integrates the draft variation
branch with the shared golf-club stack through wedge kinematics PR #4172. It
adds the first bounded implementation slice for wedge epic #4158 /
impact-inspector issue #4163:

- Canonical inspection time and event label on every `SimulationRun`: impact
  for hits, closest approach for misses.
- Exact `Jump to Impact` / `Jump to Closest Approach` controls in PyQt6 and
  React, with playback paused before the jump.
- `simulation/impact_kinematics.py`, which adapts retained pose/twist/contact
  data to `shared.python.golf_club.WedgeKinematicState` and preserves geometry
  provenance and model limitations.
- Engineering readouts in both clients for contact/reference AoA, remove-shaft
  counterfactual, shaft rotation and vertical velocity, face-normal rate,
  leading-edge/arc rate where available, and screw-axis distance.
- Restored manual angular velocity in the React simulation path; the previous
  hard-coded zero made all closure and shaft metrics false zeros.
- Deterministic midpoint tie-breaking for a flat maximum-speed plateau, so the
  manual source's automatic event is its documented square pose at 30 ms.

Physics boundary: articulated pendulum runs expose the measured wrist-to-head
shaft line but have no shaft-twist degree of freedom. The inspector reports
that absence rather than inventing torsional motion. The web mirror still does
not retain full head pose; its readout declares that limitation until WASM or a
canonical backend replaces the temporary TypeScript mirror.

Current-head release evidence: 1,006 Python/PyQt/shared-swing tests passed with
five optional Rust-wheel parity skips; 334 React/model tests and 12 swing-core
Rust tests passed; Vite production build, TypeScript, ESLint, Ruff, formatting,
and mypy across 165 source files passed. A focused post-refactor PyQt run passed
46 tests. Browser QA verified the 1,307 deg/s manual fixture at 30 ms and a
1.430 s closest-approach miss with zero console warnings/errors. Native-window
QA confirmed the control and readout are visible in the standalone PyQt6 app.

## Status Note

`src/rate_of_closure` and `src/shared/python/swing_sim` do **not exist on
`main` yet** — they land with PR #4119. This doc describes the tool as it
exists on the in-flight branch stack (`feat/impact-simulation-platform` →
`feat/investigation-suite` → `feat/course-showcase`) so the next agent has
full context the moment #4119 merges. If you're reading this on a fresh
`main` checkout and don't see `src/rate_of_closure/`, check out one of those
branches or wait for #4119 to land.

## Where This Tool Is Headed

Rate of Closure started as a single-page "closure rate" calculator (twist
model: GC-path vs impact-point-path gap, °/ft). Epics #4103 → #4120 → #4125 →
#4130 are growing it into a full swing → impact → ball-flight simulator with
PyQt6 + web parity and eventual public GitHub Pages distribution. Read
`src/rate_of_closure/README.md` (frame conventions, Cheetham dossier sourcing,
run instructions) before touching physics code.

## The #4119 → #4124 → #4129 PR Stack

Each PR consolidates a whole epic's stacked feature branches into one PR to
keep self-hosted CI load down (see `CLAUDE.md` — these are big diffs).

- **#4119** `feat/impact-simulation-platform` (epic #4103, open, auto-merge
  armed). Base tool (twist model, PyQt6 + React/Vite web, Cheetham dossier
  data) + STL clubheads/club library/inertial model + `swing_sim` shared
  package (`src/shared/python/swing_sim/`: swing sources, impact model ported
  from UpstreamDrift with 3 physics fixes, gear effect, 7 literature
  ball-flight models, goal-driven multi-start solver) + `swing-core` Rust
  crate (pyo3 + wasm) + app integration (simulation session, impact-time
  scrubber, screw-axis overlay via the rotation_converter adapter, video
  controls, CSV/JSON export, solver panel). 404 pytest + 72 vitest + 111
  cargo tests, all local gates green. Supersedes #4092/#4097/#4098/#4112-4118.

- **#4124** `feat/investigation-suite` (epic #4120, open, **draft state —
  do not merge yet**, stacked on #4119). Adds `rate_of_closure/plotting/`
  (40-variable data catalog, `PlotSpec`, Plots tab + custom-plot wizard),
  scale-separated Strike/Swing/Flight viewers + standalone Flight Explorer,
  `swing_sim/variation/` (seeded Monte Carlo/NoiseSpec engine, dispersion +
  Spearman sensitivity, Variation tab), and V4: glossary (60 terms),
  cold-user help system, "Derivation & Traceability" → "Calculation
  Description" rename, full-model derivations, hover-hint completeness sweep.
  Supersedes draft PRs #4121/#4122/#4123. 566 pytest + 125 vitest passing.

- **#4129** `feat/course-showcase` (epic #4125, open, **draft state — do
  not merge yet**, stacked on #4124). Merges `feat/realistic-heads` (H1:
  parametric club-type geometry, volumetric COG via divergence theorem),
  `feat/swing-kinetics` (H2: joint torque/force plots + 3D overlays from
  pendulum inverse dynamics), `feat/putting-vertical` (H3: `swing_sim`
  putting module + app Putting tab), then adds on top: H7 course scene
  (`ui/course.py` / `course_scene.py`) + target optimization
  (`swing_sim/solver/targets.py`, `TargetRegion`, `ImpactGoal.target_region`),
  and H6 showcase styling (`ui/pyqt6/app_style.py`, UpstreamDrift launcher
  visual language) + yards-default distance units. 413 pytest + 309 swing_sim
  - 174 vitest passing. H4 (AffineDrift putting research content) and H5
    (public release-management repo) are cross-repo and tracked in #4125
    directly, not in this PR.

**Do not merge #4124 or #4129 before their base merges** — SPEC.md sections
were unioned assuming sequential merge order; merging out of order will
produce conflicting/duplicate changelog rows.

## swing_sim Packages

`src/shared/python/swing_sim/` (introduced by #4119, home for physics shared
with UpstreamDrift via the established shared-module arrow):

- `swing_sim/flight/` — 7 literature ball-flight models (drag/lift/spin
  decay), citations in registry metadata.
- `swing_sim/impact/` — impact model ported from UpstreamDrift (offset-drop
  fix, opt-in 3×3 MOI tensor, inverted friction spin axis fix), gear effect,
  `SpringDamperImpactModel` (Kelvin-Voigt contact force history, 1e-7s steps)
  — this is the contact-force law epic #4130 will extend for the full
  contact-interval integration rather than duplicate.
- `swing_sim/solver/` — goal-driven multi-start least-squares solver;
  `targets.py` (added in #4129) adds `TargetRegion` for green/fairway
  optimization goals.
- `swing_sim/variation/` (added in #4124) — namespaced variable registry,
  `NoiseSpec`/`VariationPlan`, seeded parallel N-run Monte Carlo engine,
  dispersion/OAT sensitivity/Spearman/landing-ellipse stats.

Epic #4130 (Impact-Interval Club Dynamics) will add `impact_interval/` to
this same package — its home is explicitly `swing_sim` so UpstreamDrift
reaches it via vendor, per that epic's F2 phase description. Not started yet
(foundation-only epic, no PR).

## How #4103/#4120/#4125/#4130 Relate to This Tool

All four are rate_of_closure epics specifically (unlike the wider-monorepo
epics tracked in the root `AGENT_HANDOFF.md`, e.g. SCADA #4085-#4089).
#4103 is the foundation platform; #4120, #4125 are sequential feature waves
stacked directly on its PR; #4130 is a physics-depth epic that extends the
impact model #4103 introduced (contact-interval integration replacing the
instantaneous-impulse approximation) — foundation phase only so far.

## Web Mirror + GitHub Pages

The web mirror (`src/rate_of_closure/web/`, React/Vite/TS) is pinned
test-for-test against the PyQt6 model today (TS mirrors hand-written, not yet
WASM — that swap is explicitly deferred to Phase 7 of #4103). It builds to a
static bundle (`npm run build`). Rate does not currently contain a Tauri source
project; PyQt6/PyInstaller is its supported desktop package.

**There is no automated GitHub Pages CI deploy for this tool yet.** No
`.github/workflows/*.yml` references `rate_of_closure` or Pages deploy
actions as of this writing. The only Pages-publishing precedent in the repo
is `src/web_applications/unit_converter/unit-converter-app/DEPLOYMENT.md`'s
manual branch-folder publish (Settings → Pages → select branch/folder).
Phase 7 of #4103 ("GitHub Pages mirror updated (public share link), parity
tests as deploy gates") owns building a real workflow — do not improvise one
in an unrelated PR.

## Must-Read Architecture Pointers

1. `src/rate_of_closure/README.md` — frame conventions, unit conventions,
   dossier sourcing, run/build instructions.
2. `src/rate_of_closure/model.py` (base twist physics, no Qt) once #4119
   lands.
3. `src/shared/python/swing_sim/impact/` — the contact-force law shared
   with (and about to be extended by) epic #4130.
4. `rust_core/swing-core/` — pendulum EOM + plane projection, pyo3 + wasm
   targets, follows the `tools-core` feature-contract pattern.
5. `.github/workflows/maturin-swing-core.yml` (added by #4119) — Rust wheel
   build for this crate.

## Gate Commands (this tool)

```bash
python3 -m pytest tests/rate_of_closure src/shared/python/swing_sim -n auto --timeout=60
cd src/rate_of_closure/web && npm run test && npm run build && npx tsc --noEmit && npx eslint .
cargo test -p swing-core
python3 -m ruff check src/rate_of_closure src/shared/python/swing_sim
python3 -m mypy src/rate_of_closure src/shared/python/swing_sim
```

## Do-Not List

- Do not duplicate the Kelvin-Voigt contact-force law — #4130 requires
  `SpringDamperImpactModel` and the new `impact_interval/` package to share
  one implementation (DRY, explicit in the epic's binding standards).
- Do not exceed 500 LOC per file in `rate_of_closure`, `swing_sim`, or
  `swing-core` — sub-package instead.
- Do not hand-mirror physics into the TS web layer once WASM lands (Phase 7)
  — that's the whole point of the wasm-pack build; today's hand-written TS
  mirrors are a stopgap, not the target architecture.
- Do not merge the stacked PRs out of order (see stack section above).
- Do not invent citations in derivation docstrings or the AffineDrift
  putting research content (H4) — sourced/verifiable only, dossier
  discipline per epic #4125.

## Roadmap (ordered)

1. Merge #4119, then #4124, then #4129 in order.
2. Start epic #4130 Phase F1 (formulation document) — six-DOF rigid-club
   contact-interval derivation, validation program design.
3. Phase 7 of #4103: WASM swap + real Pages CI deploy workflow.
4. #4125 H4/H5: AffineDrift putting research content and the public
   release-management repo (both cross-repo, not started).

## 2026-08-07 Wind-Strategy Workflow Continuation

The active child branch is `feat/4199-wind-workflow`, published as
[draft PR #4282](https://github.com/D-sorganization/Tools/pull/4282) at exact
implementation head `fdcc25008`, stacked on exact PR #4281 head
`8b8690e8760d82ba814e8d95588d2540d28a6759`.  Do not fold this work into,
retarget, or merge it ahead of PR #4281.

This branch turns the shared `scalar-ensemble/v1` wind adapter into matched
end-user workflows.  Python runs the immutable request in a `QThread`; React
uses a real, lazy-loaded Vite module Worker.  Both expose exact `0..N`
progress, cancellation, current launch plus canonical landing target, trial
and wind-estimate controls, summaries, every scalar axis, explicit
completed/nonconverged/invalid availability, cohort-colored scatter, generic
all-row CSV, and fail-closed result invalidation.  Scatter controls include
pan/zoom, Auto Fit, toolbar-history reset, and movable/hidden legends in
PyQt; React includes zoom, Auto Fit, clipped marks, numeric ticks/gridlines,
and movable/hidden legend.  Captured calculation-basis regions make model,
seed, target, integration, risk, and aim-policy settings visible.

Final native-window QA at 1280 x 768 added matched ball-flight Loop controls
to PyQt and React and verified that Play/Pause, replay from landing, granular
speed, and continuous wrap all use the single owned animation clock.  The
PyQt wind workspace now separates a compact two-column Setup view from a
plot-first Results view, automatically selects Results after completion, and
keeps run/cancel/export plus progress/status visible in both views.  A live
five-trial run completed 5/5 and rendered its basis, summary, scatter, native
pan/zoom toolbar, Auto Fit, and legend-position control without overlap.

Lifecycle and safety details are contractual: the PyQt worker never reads
widgets, window shutdown cancels and joins it, queued stale signals are
ignored, and the main window explicitly stops Flight Explorer.  React
terminates its Worker on completion, error, cancellation, unmount, or consumed
input change.  Both clients preserve unavailable values as null/empty cells;
CSV strings and headers that could become spreadsheet formulas are
neutralized without altering numeric negatives.  PyQt accepts the complete
shared uint32 seed range.

Current local evidence on this working tree:

- `1350 passed, 5 skipped, 15 warnings` for the complete
  `tests/rate_of_closure` plus `src/shared/python/swing_sim` suite.  Skips are
  optional local Rust-wheel paths; warnings are the existing Hypothesis
  `norecursedirs` and empty polynomial-preview legend warnings.
- `94` React test files / `566` tests passed; focused playback and wind suites
  also pass.
- Ruff, Black, targeted mypy, TypeScript, zero-warning ESLint,
  `cargo test -p swing-core` (`12 passed`), and `git diff --check` pass.
- The 175-module production build emits separate wind-worker and lazy
  wind-workspace chunks; the main chunk is 472.34 kB and has no size warning.
- Every changed production source is at most 400 lines and every changed wind
  function is at most 50 lines.

This completes the #4199 current-launch workflow slice, not epic #4199 or the
universal many-run objective.  Capability optimization still discards its
individual evaluator rows.  The next child must add the optional streaming
`CapabilitySampleObservationV1` sink and cancellation hook described in
[issue #4197](https://github.com/D-sorganization/Tools/issues/4197#issuecomment-5223170071),
then adapt those rows to `scalar-ensemble/v1` without bloating the compact
optimization result.

Ground and four-surface parity remain open epics.  The latest executable
acceptance refinements are in
[ground #4267](https://github.com/D-sorganization/Tools/issues/4267#issuecomment-5223106106)
and
[parity #4260](https://github.com/D-sorganization/Tools/issues/4260#issuecomment-5223106465).
Do not treat a launcher tile as a fourth UI implementation, and do not equate
launch-monitor total displacement with accumulated ground path length.

### 2026-08-07 CI repair and current ground/parity findings

The first hosted run for PR #4282 found one actionable delta-mypy defect:
`WindStrategyLifecycleMixin.closeEvent` conflicted with Qt's nullable close
event signature under Python 3.12.  Commit
`424b4c395370aea26069386c070a65f7abe885bc` introduces a concrete
`WindStrategyGroupBox`, keeps worker teardown in the mixin, and gives the Qt
override the correct `QCloseEvent | None` contract.  Exact Python 3.12 mypy
now passes for 11 changed production files, as do Ruff, formatting,
`git diff --check`, and 19 focused wind/playback tests.  Do not merge until the
new protected checks and the entire parent stack are green and approved.

Read-only audits against current UpstreamDrift remote `main`
`0782853295e005af68818617e4725eb980890f43` found useful but unqualified
contact, terrain, turf, and putting-roll code.  Preserve the direction
`UpstreamDrift adapter -> Tools ground-run/v1 authority`; Tools must not import
UpstreamDrift.  Do not reuse the terrain serialization without fixing its lost
material fields, and do not start bounce/roll physics until first physical
sphere contact, arbitrary surface normal, target-frame conversion, and full
terminal angular velocity are available through a strict transfer contract.

The four-surface parity baseline is not complete: UpstreamDrift PyQt launches
the Tools native window, UpstreamDrift React has no Rate route, and Tools React
still has narrower impact/flight authority than Tools PyQt.  Close parity
through shared versioned calculation contracts and golden fixtures, not by
counting launchers, separate simulators, or copied model implementations.

## 2026-08-07 Capability-Observation Continuation

Active branch `feat/4197-capability-observer` is an unretargeted child of
`feat/4199-wind-workflow` at exact parent head
`6e3c1029f1f3a80ae09020ef7d0afacb3c0d5484`.
It is published as [draft PR #4283](https://github.com/D-sorganization/Tools/pull/4283);
the validated implementation/hardening commit is
`5c6073bd68ed4c8f23b343d4d11c2dc4277ea246`.

The Python and TypeScript capability optimizers now stream one immutable,
versioned observation for every attempted ensemble sample and support typed
cooperative cancellation before an evaluator call.  Existing callers and the
compact optimization result are unchanged.  Status normalization is fail
closed, evaluator exception text is not leaked, all valid metrics retain their
declared order and provenance, and malformed or incomplete landing results do
not become fabricated successes.

The Rate app adapters turn the stream into bounded `scalar-ensemble/v1`
datasets with a complete scalar flight catalog, null unavailable outputs,
nominal/perturbed parameters, target residuals, and source lineage.  They
require contiguous zero-based attempts, reject overflow before retention,
deep-copy/freeze TypeScript inputs, and serialize ASCII and Unicode fixtures
byte-identically across runtimes.

Current local gates: 120 Python tests passed with four optional Rust-wheel
skips; 96 React files / 580 tests passed; Python 3.12 mypy, Ruff, Black,
TypeScript, ESLint, Vite build, structural budgets, and diff checks pass.
Publish only as the next protected stacked draft PR, then keep #4197 open for
its remaining user-facing capability-optimization workflow.

An independent review blocked PR creation after the first branch push and the
four findings are now fixed locally.  Stable JSON uses a shared raw-number
policy rather than native runtime spelling; derived parameter labels uppercase
only an initial ASCII letter; public observations enforce complete landing and
empty incomplete-status metric invariants; and TypeScript compares parameter
declarations structurally instead of with delimiter-concatenated signatures.
Adversarial IEEE rounding/exponent/negative-zero, Unicode, control-delimiter,
non-finite, and status-matrix tests are green.  Current totals are 135 Python
flight/adapter tests passed with four optional Rust-wheel skips and 96 React
files / 584 tests passed, with the previously recorded lint, type, build, and
budget gates still green. The corrections are committed and published in draft
PR #4283. Monitor its protected checks and reviews together with PRs
#4279–#4282; do not retarget, rewrite, bypass, or merge this child ahead of
`feat/4199-wind-workflow`.

The first hosted CI Standard run found one actionable delta-mypy boundary in
the new `_capability_observation_runtime.py`: unchanged imported request fields
resolve as `Any` under `--follow-imports=skip`, so `total_count` now casts their
already contract-validated product to `int`. No runtime behavior changes. The
exact seven-file Python 3.12 mypy command, Ruff/format, diff check, and full
135-test flight/adapter suite pass with four optional Rust-wheel skips. This
fix and handoff update are `SELF`; resolve with `git rev-parse HEAD`, push
normally, and monitor the fresh protected checks.

The subsequent protected run's `detect-secrets` job classified the two pinned
Python/TypeScript SHA-256 parity digests as high-entropy strings. These are
expected deterministic test outputs, not secrets. Use only exact inline
`pragma: allowlist secret` annotations on the four digest constants; preserve
the repository baseline and scanner scope. Commit the annotations and this
handoff together on the parent capability-observer branch, then re-run the
protected stack without force-pushing or bypassing the gate.
