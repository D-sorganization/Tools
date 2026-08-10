# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-09

## 2026-08-09 Universal Variation Visualization Continuation

Branch `feat/4144-variation-export-continuation` remains based on
`feat/4218-toolstrip-workspace` and now contains exact corrected parent
`3f67ed466fefc8991db9c4409f921f25e1c37142` through the normal merge containing
this handoff. The core PyQt6/React variation workspace has selectable
input/contact/impact/shot scatter, a scatter matrix with marginals, landing
dispersion, and an all-trial 3D swing-arc overlay with reference trace,
principal spread, RMS variability, quiet zones, filtering, and linked trial
selection.

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

The current local continuation normally incorporates exact corrected parent
`3f67ed466fefc8991db9c4409f921f25e1c37142` while preserving #4280's
`feat/4218-toolstrip-workspace` base. Its current focused evidence is `46`
Python 3.11 tests and `19` real-Python-3.10.20 tests with PyQt6 installed,
including the merged compatibility and variation widget contracts. React
passes `1 file / 8 tests`, TypeScript, and focused zero-warning ESLint. Ruff,
format, pinned mypy 1.13, documentation governance, ancestry/SPEC assertions,
and diff checks pass. SPEC 1.14.10 is the unique child entry for the
scatter-export and accessible-row-table source change. This propagation is
local only and has not been pushed.

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
on exact corrected draft PR #4280 head
`38ed58cab96842a3007e76a855db83ee2452b8fd`. That parent is incorporated by a
normal local merge while preserving base
`feat/4144-variation-export-continuation`; no branch was rebased, retargeted,
force-pushed, or published by this continuation.
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

The original broad local gates remain green: 906 Python/PyQt/shared-swing tests
passed with one expected optional-Rust skip and 15 existing warnings; 91 React
files / 555 tests passed. After corrected-parent propagation, focused evidence
is `49 passed` on Python 3.11 and `49 passed` on real CPython 3.10.20 with
PyQt6/scientific dependencies present, plus `2 React files / 10 tests`,
TypeScript, and zero-warning focused ESLint. Ruff check/format passes all four
child Python files; pinned mypy 1.13 passes both production modules and the
wind-adapter test after its status helper received an exact `Literal` type.
Production Python modules remain below 400 lines and no changed Python function
exceeds 50 lines. SPEC 1.14.11 uniquely records the child change above parent
1.14.10. This contract/adapter does not complete #4199: the next slice still
needs background execution, progress and cancellation, PyQt/React
scatter/strategy UI, persistence, and export wiring.

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

## 2026-08-09 Workspace Python 3.10 compatibility completion

The #4279 continuation now contains exact corrected parent
`08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` through normal local merge
`a340fabefa443d47325c5538f342683b38c01ade`, without rebasing, retargeting,
force-pushing, or publication. The workspace command and view enums use the
shared `StrEnum` runtime compatibility contract with native enum typing kept
under `TYPE_CHECKING`; `_workspace_validation` uses the shared `UTC` value.
Stable identifiers, enum values, UTC serialization, schemas, and behavior are
unchanged.

The combined regression inspects the runtime branch for all nine inherited
and child string-enum modules plus both UTC modules and directly executes the
three child workspace modules. Focused Python 3.11 evidence is `126 passed`.
Real CPython 3.10.20 evidence is `14 passed` plus successful dotted imports of
10 Rate/shared targets. Ruff check/format is clean, and pinned mypy 1.13 passes
all 11 affected production files under the changed-file CI contract. Keep the
root, tool, campaign handoffs and SPEC 1.14.9 entry in the implementation
commit; protected CI and review remain required after a normal push.

## 2026-08-09 Workspace stack parent propagation

Draft PR #4279 keeps base `feat/4181-launch-monitor-registry`. Exact published
parent head `08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` is carried by a normal
merge without rebasing, retargeting, force-pushing, or publishing this local
continuation. Source changes applied cleanly; overlapping handoff/specification
history was reconciled monotonically. The result retains the parent's facade
and Python 3.10 repairs plus the child's complete File/View/Tools registry,
strict workspace document, matched PyQt6/React toolstrips, module
visibility/order, playback, and independent plot controls.

The reconciled ancestry passes `126` focused Python/PyQt tests and `8 React
files / 32 tests`. Ruff check/format passes all 28 relevant Python files, and
CI-pinned mypy 1.13 passes the 18 changed production files and both facade
tests. Mypy caught an untyped Qt `isChecked()` result in the child's
`legend_visible()` boundary; the method now returns an explicit `bool` without
changing runtime behavior. The affected simulation GUI rerun is `29 passed`;
documentation governance and diff checks pass. Keep this handoff, the root
handoff, the campaign handoff, and `SPEC.md` together in the merge commit.

## 2026-08-09 Python 3.10 enum import repair

PR #4203 now owns the earliest fix for seven inherited Rate/shared swing
modules that imported Python 3.11's `enum.StrEnum` directly. Runtime code uses
the established shared compatibility helper while mypy sees the native enum
only inside `TYPE_CHECKING`. This preserves wire values and enum semantics on
supported interpreters and lets the repository's hosted Python 3.10 lane
collect the modules. The focused evidence is 64 tests, Ruff/format, pinned
mypy 1.13 across eight changed files, and a real CPython 3.10.20 source/runtime
probe. The exact published #4203 head is
`ab7de5a47977417e02926c3fbc7476002e82b690`; propagate it through the existing
stack without changing bases.

A follow-up scan found the torque-profile controller importing
`datetime.UTC` directly. That parent-owned Python 3.11 boundary now uses
`shared.python.compatibility.UTC`; persisted timestamps and torque-profile
behavior are unchanged. Re-run the focused torque-profile UI tests and the
real Python 3.10 compatibility probe before propagation.

## 2026-08-09 Launch-registry parent CI repair

PR #4203 exact-head run `31199764932` reached the Python test lanes but failed
during Linux collection, before behavioral assertions. The two in-package
flight/solver facade tests were collected as `src.shared...` modules while
their absolute aliases crossed into the editable `shared...` namespace. They
now import their sibling facade package relatively, preserving the pinned
public API contract and production behavior. Reproduce with pytest
`--import-mode=importlib`; keep the separate Rust missing-`libpython3.11`
failure classified as runner infrastructure. Publish normally and propagate
the repaired parent through the existing stack without changing PR bases.
Verification is `12 passed` on Windows and `12 passed` on WSL Python 3.11
under importlib collection; Ruff/format and exact mypy 1.13 pass for both
changed modules. The dataclass metadata assertion remains active behind an
explicit test-only `Any` introspection boundary.

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
static bundle (`npm run build`) and carries Tauri scripts for desktop
packaging, same as other web tools in the repo.

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
