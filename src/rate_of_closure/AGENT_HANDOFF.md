# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-10

## 2026-08-10 #4111 assembly-to-simulation adapter (draft PR #4341)

Independent publication review found and repaired one desktop ownership gap:
changing the Simulation tab's club now invalidates the binding in the Club
panel that owns it, so export status and solver state cannot disagree. The
new signal-bound regression and the combined 67-test adapter/PyQt/GUI suite
pass serially. Focused Ruff/format, CI-pinned Mypy 1.13, the 27-test React
suite, TypeScript, zero-warning ESLint, and the 125-module production build are
clean. The first test invocation exposed only a missing test fixture and was
corrected before implementation evidence. The repository's generic 500-LOC
file gate still reports three pre-existing oversized modules touched by this
stack; the repository's baseline-aware 1200-line module budget passes.

Local branch `feat/4111-assembly-simulation-adapter` starts at exact published
PR #4338 head `6b55a6b01e8029712217185f0e0ebf2a421be20e` and is published
as draft PR #4341 while remaining **not released**. It adds a fail-closed adapter from the
validated selected-spec to `ClubAssembly` binding to the existing impact
boundary. Desktop consumes the bound head mass and may rotate the authoritative
head-CG tensor as `R I_head R^T` only when a complete selected-head-to-app
attitude is explicitly declared. The manual source declares that pose; current
double/triple pendulum sources do not, so their tensor capability stays
unavailable. The scalar-only browser solver consumes the validated head mass
only and reports tensor/CG unavailability. Neither surface ever substitutes
assembled-club mass, CG, or inertia for head properties, and a miss records
every property as `not_used` without impact/flight. PyQt/React expose binding
status and clear mismatched selection state; exports include a non-duplicating
capability ledger. The binding fixture remains synthetic qualified-analysis
evidence. Canonical campaign handoff/release-manifest files are still absent on
this lineage, so no release state was invented.

Final local evidence on this continuation: 42 focused Python simulation,
contact, ball-setup, export, and adapter tests pass; three focused PyQt binding,
invalidation, and source-status tests pass; and 27 React adapter, persistence,
Club-panel, and Simulation-panel tests pass. Focused Ruff lint/format, Black,
CI-pinned Mypy 1.13 (eight changed sources), repository module-size and docs
governance gates, React ESLint/type-check, and the production Vite build are
clean. A first broad 14-worker GUI run saturated shared Qt setup and timed out;
the same new GUI coverage passes serially. A broader serial regression then
caught and drove the fix that prevents unbound manual runs from constructing a
binding-only attitude.

## 2026-08-10 assembly-binding lint follow-up

The post-split TypeScript compiler passes on exact local head
`e9e75614f29a0d6fb4925bc2b096155a9f9edf25`. ESLint then found one
test-only explicit `any`; the assertion view now uses a narrow structural
inspection type instead. Runtime behavior, wire bytes, identities, sidecar
content, and scientific availability remain unchanged and `not_released`.
The focused UI test also now waits for the asynchronous re-import result
instead of matching the already-present selection-cleared status.

## 2026-08-10 Authoritative Transfer Snapshot

The publication-hardening follow-up keeps
`fdce377f123f925bf5768b666619927886f17ae9` intact as its parent and moves the
matrix, rotation, eigenvalue, and inertia-validation operations into the
cohesive 88-line `clubAssemblyMassMath.ts`. The remaining wire/schema and
assembly-orchestration module is 316 lines; every production TypeScript file
new to this binding slice is at most 328 lines. No schema, canonical identity,
digest, validation, or exported-sidecar behavior is changed. The formatter
command timed out under workstation load, and no further gate was started after
the coordinating agent's explicit boundary. Prior focused evidence remains
valid, while the final TypeScript/ESLint/build state remains unverified. This
follow-up is **not released**, and the absent campaign handoff/manifest paths
described below remain absent on this lineage.

Local branch `feat/4111-club-assembly-binding` starts from exact published
parent `66da8f024450ad5e7940a8005227f3d6612a85d8` and remains **not released**.
The parent contains no campaign handoff at
`docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md` and no release manifest
at `docs/release/rate_of_closure_campaign.v1.json`; this continuation therefore
does not create release state or claim delivery. It updates this canonical Rate
handoff, the root handoff, README/specification, and tensor contract. Fresh
bounded evidence is 14 passing Python binding/legacy-sidecar tests, 5 passing
browser binding tests, two previously passing focused PyQt import/export/error
tests, clean focused Ruff, and clean CI-pinned Mypy 1.13 across five changed
sources. A three-file React run, the final TypeScript check after its
ArrayBuffer typing repair, and an earlier full GUI run timed out without failure
output on the overloaded workstation. TypeScript/ESLint/build are therefore
unverified on this local head and the overloaded broad commands should not be
repeated.

Issue #4111's selected-club export gap is now implemented: the mesh-defining
subset of the current `ClubSpec`—type/style, head mass, loft, and optional
curvature—serializes through the deterministic parametric generator; name,
length, lie, CG, and MOI do not drive the representative mesh. Both Club panels
export conventional millimetre STL coordinates with canonical axes in the
header/tooltip and safe bounded filenames. PyQt uses atomic destination
replacement; React uses the same deterministic geometry/header contract and a
browser-local `model/stl` object URL that is released after success or click
failure. Tests cover geometry/extents, filename edge cases, runtime validation,
browser success/failure status, cancel, serialization/write failure, and
preservation of an existing native target on replace failure. No full tensor
is derived: the current shaft-axis scalar enters the CG-centered impact
equation only as an isotropic-equivalent compatibility approximation with an
explicit axis/reference mismatch. See
`docs/development/rate_of_closure_clubhead_tensor_contract.md` for the measured
or density-integrated tensor and complete-frame contract required to enable it.
Both panels also export a deterministic
`rate_of_closure.clubhead_engineering/1` JSON sidecar. It SHA-256-identifies the
unchanged companion STL, declares the head/STL frames and identity-plus-scale
transform, and carries the selected head mass with explicit representative-
input provenance. Partial CG offsets and the shaft-axis scalar are
`evidence_only`; the complete CG, full tensor, world attitude, and assembly
properties remain `unavailable` with no substitute value. The shared
`golf_club.ClubAssembly` contract is authoritative when supplied. The new
`rate_of_closure.club_assembly_binding/1` boundary binds it to the exact selected
spec only after cross-language canonical SHA-256 identities, qualified source
authority, SI units, one head, exact head mass, explicit frame transform, and
complete physical tensors validate. Both panels provide a discoverable import,
clear it on an identity-defining edit, and expose complete head/assembly mass
properties only in a subsequent bound sidecar. Duplicate fields, oversized
documents, absent/unknown fields, and all identity/frame/property mismatches
fail closed. The driver golden is synthetic qualified-analysis test data, not
manufacturer data. Source authority is preserved but not independently
certified; dynamic world attitude and simulation tensor injection remain
unavailable rather than inferred.

Local evidence for this continuation must be read from the latest entry below;
the earlier 511 Python/PyQt and 328 React counts describe the parent sidecar
head, not this unpushed binding continuation.

The tool now exists on the #4119 branch and its large descendant stack, but it
is still not released to Tools `main`. PR #4119 remains the dependency root.
This branch includes current `main` plus the following review repairs:

1. `launch_web.py` imports `rate_of_closure.gui_registration` absolutely, so
   the Tools launcher can execute it by file path.
2. `gui_registration.py` declares SciPy alongside PyQt6, Matplotlib, and NumPy,
   matching the eagerly imported solver stack.
3. `SimulationPanel.tsx` runs Auto τ with an explicit null-time input and
   records that same input's signature, preventing a stale fixed-time rerun.
4. Parametric-head station refinement carries an explicit four-coordinate
   tuple type, closing the only scoped mypy finding without changing geometry.
5. `ClubCanvas.tsx` is below the 400-line production-file cap after pure mesh
   transformation, shading, projection, and arrow drawing moved to
   `clubCanvasRendering.ts`; focused helper tests pin face alignment, placement,
   sorting, and lighting.

Current integrated evidence: 948 Python tests pass with one expected missing-
Rust-wheel skip; 312 React tests pass across 42 files; repository Ruff
lint/format, changed-file Mypy across 191 source files, React ESLint,
TypeScript, and production build are clean. The final Rate model/mesh slice
also passes Mypy 1.13 and a newer verifier plus 134 focused tests. The next
agent must verify the exact GitHub head and protected checks, obtain the
required approval, and merge normally. Do not claim #4119 or any epic complete
before protected release to `main`.

The first protected run at the fully typed head passed Mypy and Ruff but found
six empty `swing_sim/*/tests/__init__.py` package markers in the changed-test
assertion gate. The repository allowlist now exempts source-local test-package
markers only; it does not exempt executable test modules.

The following protected run reached the module-size gate and found the
repository formatter's Movement Optimizer `motion_tabs.py` change at 1,216
lines. A behavior-neutral documentation/call-layout compaction brings it to
1,152 lines; the exact size budget and 80 focused Movement Optimizer tests pass.

The repository's authoritative formatter is Ruff. The current-main merge
required a mechanical `ruff format` normalization of 101 Python files,
including four Rate/swing-sim files; there is no Rate behavior change, and
both repository-wide Ruff lint and format checks are clean afterward.

The next protected gate exposed legacy Mypy debt because those mechanically
formatted files became part of the changed-file set. Rate UI/kinetics,
variation, shared core, model, mesh, club geometry, simulation, canvas, and
course-rendering boundaries now carry explicit NumPy/scalar types; dynamic
solver goal dictionaries flow through validated `ImpactGoal.from_mapping`.
The complete changed-file Mypy surface is locally clean. These repairs add no
ignores and do not change model values or physics equations.

After the root lands, preserve dependency order. The highest-value remaining
UI slice is #4225's actual multi-viewport compositor; #4224 still needs
responsive/DPR-aware plot redraw, measured non-overlapping legends, and
clipping-safe current-state exports. Ground/tee #4143 is implementation-ready
on #4325 and needs release/checklist reconciliation rather than new physics.

The older sections below are retained as architecture/history; this snapshot
is authoritative when their dated state differs.

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
