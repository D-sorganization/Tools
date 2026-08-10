# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-10

## 2026-08-10 #4143 Python/React Golden Ball-Setup Parity

The isolated `feat/4143-tee-parity-fixture` child begins at exact draft PR
#4203 head `31cbc007d4c85b5479b7cd0fb0969124eab2af67`. One versioned JSON fixture now
drives both Python and React ball-support tests with explicit SI units and the
ground-plane-to-ball-bottom height reference. The shared cases cover Driver
and non-Driver defaults, user overrides, Ground zero effective height,
derived ball-center geometry and serialization, negative/NaN/infinite height
rejection, and backward-compatible migration of a legacy run without
`ball_setup`.

Evidence is 18 passing Python tee/parity tests and 24 passing React
tee/persistence/parity tests, plus green TypeScript, ESLint, Vite production
build, Ruff check, and Ruff format. Production model and UI code are unchanged.
Do not close #4143: rendered Playwright/PyQt Ground/Tee evidence, protected
current-head CI/review, and release to `main` remain. This exact parent predates
the strict campaign manifest on a divergent branch, so no downstream manifest
was copied into this bounded test-only child.

## 2026-08-10 Launch-Registry Child Propagation

Draft PR `#4203` (`feat/4181-launch-monitor-registry`) retains its
`feat/4189-dplane` base and normally merges original child head
`08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` with exact parent head
`b443fdbed7064c5db0320106013c8413e3e24356`.

The reconciliation preserves the child branch's responsive
`SimulationViewControlsMixin` and delegates its persisted D-plane layer
checkboxes to the parent's `ImpactLayerControls` helper. The compatibility
mapping and helper mapping are the same object, which prevents duplicate state
while retaining existing UI automation and persistence behavior. Both PyQt
modules remain below the 500-line limit.

The original child already exceeded the protected budget in swing sources,
the plotting catalog, and the PyQt main window. Focused modules now own the
triple-pendulum model, immutable plotting-variable contract, and versioned tab
state. Compatibility imports remain the same objects, while the former
monoliths fall to 282, 459, and 494 lines. Focused evidence is green across 36
PyQt simulation/layout tests, 38 plotting/navigation tests, and 21 simulation
source/export tests. Combined verification is green across 1,249 Python tests
with six explicit skips, 521 React tests and all web gates, 12 Rust tests,
real CPython 3.10 checks, scoped Ruff/Black/pinned MyPy, and repository
governance/security checks. The changed-file 500-line gate passes all 107
candidate files. A full-tree audit still reports untouched `kinetics.py` and
`torque_profile_panel.py`; neither differs in this propagation. Independent
staged review found no actionable findings after 95 additional focused tests.
Current-head protected CI and required repository review remain release gates.

## 2026-08-09 Python 3.10 enum import repair

PR #4203 now owns the earliest fix for seven inherited Rate/shared swing
modules that imported Python 3.11's `enum.StrEnum` directly. Runtime code uses
the established shared compatibility helper while mypy sees the native enum
only inside `TYPE_CHECKING`. This preserves wire values and enum semantics on
supported interpreters and lets the repository's hosted Python 3.10 lane
collect the modules. The focused evidence is 64 tests, Ruff/format, pinned
mypy 1.13 across eight changed files, and a real CPython 3.10.20 source/runtime
probe. This commit follows exact published #4203 head `9dbceff76`; propagate it
through the existing stack without changing bases.

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

## 2026-08-10 D-Plane Child Propagation

Draft PR `#4202` (`feat/4189-dplane`) retains its
`feat/4162-wedge-impact-visualization` base and normally merges original child
head `b4abec03bccfbdd87ddf91427159c5c2332c21dd` with exact parent head
`6704a3e541a3e74c28b4a284530d1a21269dd340`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the frame-explicit D-plane contract.

Persisted D-plane layer controls now live in a focused helper so
`simulation_view.py` again satisfies the protected 500-line budget while
retaining the existing UI-automation compatibility seam. Combined verification
is green: 93 focused and 825 scoped Python tests (two optional `build123d`
skips), 360 React tests and all web gates, real CPython 3.10.20
compilation/UTC, scoped Ruff/Black/MyPy, and repository governance gates. The
exact parent's 12 unchanged `swing-core` tests remain applicable because this
child has no Rust delta. The 17-error broad MyPy Qt/NumPy baseline in 11
untouched files remains separate. Protected CI and required review remain
release gates.

## 2026-08-10 Impact-Visualization Child Propagation

Draft PR `#4179` (`feat/4162-wedge-impact-visualization`) retains its
`feat/4166-wedge-turf-physics` base and normally merges original child head
`0eb804e70887c788421332369e42792411aff55a` with exact parent head
`bfa83aedc88ead380babc73a699377d98b971006`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the exact-event scene contract.

Combined verification is green: 58 focused and 739 scoped Python tests (two
optional `build123d` skips), 347 React tests and all web gates, real CPython
3.10.20 compilation/UTC, scoped Ruff/Black/MyPy, and repository governance
gates. The exact parent's 12 unchanged `swing-core` tests remain applicable
because this child has no Rust delta. The 17-error broad MyPy Qt/NumPy baseline
in 11 untouched files remains separate. Protected CI and required review
remain release gates.

## 2026-08-10 Turf-Physics Child Propagation

Draft PR `#4178` (`feat/4166-wedge-turf-physics`) retains its
`feat/4161-wedge-ground-clearance` base and normally merges original child
head `aaae3f73e17dbfaad5cca1dc6f49559b3aebe9d5` with exact parent head
`9ea93e92563280ec34bca682ad44d7409edd7a02`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the provenance-gated turf model.

Combined verification is green: 56 focused and 732 scoped Python tests (two
optional CAD-dependency skips), real CPython 3.10.20 checks, scoped
Ruff/Black/MyPy, and repository governance gates. The unchanged TypeScript and
Rust surfaces retain the exact parent's green 345 React and 12 Rust test
evidence. The 17-error broad MyPy Qt/NumPy baseline in 11 untouched files
remains separate. Protected CI and required review remain release gates.

## 2026-08-10 Ground-Clearance Child Propagation

Draft PR `#4174` (`feat/4161-wedge-ground-clearance`) keeps its
`feat/4163-impact-inspector` base and normally merges original child head
`880a6465fc872cf3d6650283db154ddc41793a31` with exact parent head
`9ddaff3b6bca542fd7a2befc7d7b0ae53910a60a`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the ground-clearance analysis.

Combined verification is green: 56 focused and 703 scoped Python tests (two
optional `build123d` skips), 345 React tests and all web gates, 12 Rust tests,
real CPython 3.10.20 compile/UTC checks, scoped Ruff/Black/MyPy, and repository
governance checks. The existing 17-error broad MyPy Qt/NumPy baseline across
11 untouched files is documented, not expanded. Current-head protected CI and
required review remain pending.

## 2026-08-10 Python 3.10 Repair Propagation

Draft child PR `#4173` (`feat/4163-impact-inspector`) retains its
`feat/4144-variation-visualizations` base and normally merges original child
head `3c43955aaeb3964ff8c3ef2748d626baae518b76` with exact parent head
`22b66b560652b78de84141344c4ddd9a92a83b26`. This carries the shared
Python 3.10-compatible UTC export and the source-wide AST guard into the wedge
impact inspector without changing the persistence schema or user-visible
timestamp format.

Combined-stack verification is green across 63 focused Python tests, all 562
Rate tests, all 334 React tests, TypeScript/ESLint/Vite gates, 12 `swing-core`
tests, real CPython 3.10.20 compile/UTC checks, Ruff/Black, focused pinned MyPy
1.13, and repository governance checks. The broad MyPy sweep retains 17
pre-existing Qt/NumPy typing findings in 11 untouched files. The PR must remain
draft until its exact-head protected checks complete and required review
approves. Do not retarget, rewrite, force-push, admin-merge, or count
infrastructure failures as passing evidence.

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
