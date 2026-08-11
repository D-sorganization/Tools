# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-10

## 2026-08-10 #4111 assembly-to-simulation adapter (local, not released)

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
PR #4338 head `6b55a6b01e8029712217185f0e0ebf2a421be20e` and remains
**not released** with no GitHub write. It adds a fail-closed adapter from the
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

- Publication-hardening follow-up on the same **not released** branch preserves
  `fdce377f123f925bf5768b666619927886f17ae9` as its parent and extracts the
  three-dimensional matrix, rotation, eigenvalue, and inertia-validation
  operations into `clubAssemblyMassMath.ts`. The new math module is 88 lines;
  `clubAssemblyMassProperties.ts` is 316 lines, and every production TypeScript
  file introduced by the binding slice is at most 328 lines. Schema, canonical
  identity bytes/digests, validation order, and sidecar behavior are unchanged.
  Under the continuing workstation overload, the combined formatter/count
  command timed out and the direct worktree-local Prettier executable was not
  present because dependencies are supplied through the sibling junction. At
  the coordinating agent's gate boundary, no further test/type/lint/build
  command was started; the prior evidence and explicit unverified
  TypeScript/ESLint/build boundary below remain authoritative. The absent
  campaign-handoff/manifest explanation below remains unchanged.

- Local continuation `feat/4111-club-assembly-binding` is based exactly on the
  published sidecar parent `66da8f024450ad5e7940a8005227f3d6612a85d8` and is
  **not released**. It adds the validated selected-spec ↔ shared-assembly
  boundary described below without changing the parent or writing to GitHub.
  On that exact parent, neither
  `docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md` nor
  `docs/release/rate_of_closure_campaign.v1.json` exists, so this commit does
  not invent a release manifest or claim campaign delivery. The canonical root
  and Rate handoffs, Rate README/specification, and tensor-contract document are
  updated instead. Fresh bounded evidence: 14 Python binding/legacy-sidecar
  contract tests and 5 browser binding tests pass; the two focused PyQt import,
  export, mismatch, and invalidation tests passed before the final documentation
  and typing-only adjustments. CI-pinned Mypy 1.13 is clean across the five
  changed Python sources, and focused Ruff lint/format are clean. A three-file
  React test command, the final TypeScript check after its ArrayBuffer typing
  repair, and an earlier full GUI command timed out without failure output under
  workstation load. Final TypeScript/ESLint/build status is therefore
  unverified on this local head; do not repeat those overloaded broad commands.

- Issue #4111 now has a tested mesh-defining-`ClubSpec`-subset → deterministic
  binary STL path and discoverable PyQt and React export actions. The generator
  uses type/style, head mass, loft, and face curvature—not name, length, lie,
  CG, or MOI. It computes in SI metres and exports conventional millimetre
  coordinates with canonical axes in the header/tooltip and hardened filenames.
  PyQt uses atomic destination replacement; React uses a browser-local
  `model/stl` object URL that is released after success or click failure. The
  3×3 tensor is intentionally not inferred. The current shaft-axis scalar is
  used by the CG-centered impact equation only as
  an isotropic-equivalent compatibility approximation with an explicit
  axis/reference mismatch. The exact unblock contract is recorded in
  `docs/development/rate_of_closure_clubhead_tensor_contract.md`.
- Issue #4111 now also has matched, discoverable PyQt/React engineering-sidecar
  export (`rate_of_closure.clubhead_engineering/1`). The JSON records the exact
  unchanged companion-STL SHA-256/size, declared head/STL frames and transform,
  selected head mass provenance, and a machine-readable capability ledger.
  Partial CG offsets and the shaft-axis scalar are evidence only; complete CG,
  full symmetric CG tensor, world attitude, and assembly properties fail closed
  as unavailable. No default-mass assembly or uniform-density mesh tensor is
  inferred. The strict `rate_of_closure.club_assembly_binding/1` continuation
  now provides that explicit boundary: deterministic cross-language spec and
  assembly identities, qualified authority, SI units, unique-head/exact-mass,
  frame-transform, and physical-tensor validation precede availability. PyQt
  and React provide discoverable import and clear a retained binding on an
  identity-defining selection edit. Duplicate/oversized/unknown/mismatched data
  fails closed. The golden assembly is synthetic qualified-analysis test data,
  not manufacturer data; authority is preserved but not certified. World
  attitude and simulation tensor injection remain unavailable. The parent
  sidecar head's 511 Python/PyQt and 328 React counts do not cover this local,
  unpushed continuation; use its newest focused evidence entry below.
- The protected Rate of Closure release still starts at Tools PR **#4119**
  (`feat/impact-simulation-platform`). Every later Rate/Wedge/D-plane,
  workspace, variation, wind, camera, and ground-model PR depends on this
  root reaching `main`; feature-branch merges are not production releases.
- This branch has been reconciled with `origin/main` at
  `3e895991599e55ad151a9580e41d1978ee3911d1` by an ordinary merge. Conflict
  resolution preserves both the root simulation stack and current fleet CI,
  golf-club, and handoff contracts.
- The three live non-outdated review findings on #4119 are repaired together:
  the web file-path launcher uses an absolute package import, PyQt dependency
  discovery declares SciPy, and React Auto τ runs the null-time input
  atomically instead of closing over the prior fixed time. Focused regression
  tests cover all three behaviors.
- Post-merge scoped quality gates also make the parametric-head station
  refinement's fixed four-coordinate tuple contract explicit to mypy; this is
  a type-only clarification and does not change generated mesh values.
- The CI Mypy 1.13 failures in Rate's NumPy/STL, club geometry, simulation,
  PyQt canvas, and Matplotlib patch boundaries are repaired with explicit
  types and casts only. The exact scoped CI mode passes 11 files, 134 focused
  tests pass, and scoped Ruff lint/format pass; runtime behavior is unchanged.
- The next protected run passed changed-file Mypy and both Ruff gates, then
  exposed six empty source-local test-package `__init__.py` markers in the
  assertion-quality gate. The repository allowlist now classifies only those
  package markers as support files; executable test modules remain enforced.
- After that repair, the protected module-size gate identified the formatter-
  touched Movement Optimizer `motion_tabs.py` at 1,216 lines. Concise comments,
  docstrings, and Ruff-normalized call layout reduce it to 1,152 lines without
  behavior changes; the exact budget passes and 80 focused GUI/theme/vector
  tests pass.
- The first protected `quality-gate` run at the repaired head exposed nine
  misplaced Ruff suppressions in Sidekick GUI tests inherited from `main`.
  Their intentionally retained QApplication references now use `_app`, so the
  required lint gate can evaluate the Rate root instead of failing at baseline.
- The remaining outdated #4119 review concern is now addressed too:
  `ClubCanvas.tsx` is 391 lines, with mesh transformation, painter sorting,
  lighting, projection drawing, and velocity-arrow rendering extracted into a
  tested 234-line `clubCanvasRendering.ts` module. The legacy 400-line cap no
  longer blocks the root release.
- The authoritative repository format gate is `ruff format --check .` (not
  Black). After the current-main merge exposed 101 legacy mismatches, this
  branch mechanically normalized those Python files with `ruff format`; Ruff
  lint and format checks are now clean. This is formatting-only across the
  affected tools, with no material domain behavior or handoff change.
- That broad formatter reconciliation caused the changed-file Mypy gate to
  inspect legacy Rate/shared numerical boundaries. The Rate UI, kinetics,
  solver-goal, variation, flight/impact/swing/reference/solver, model, mesh,
  club-geometry, simulation, and plotting groups now use explicit NumPy
  result types and narrow scalar normalization at dynamic boundaries.
  `ImpactGoal.from_mapping` replaces the unsafe dynamic `**dict` path. All
  132 protected findings are cleared without blanket ignores or changes to
  numerical equations; the final model/mesh slice is independently clean in
  Mypy 1.13 and a newer verifier, with 134 focused tests passing.
- Before release, push the merge normally, verify the new exact PR head, wait
  for protected CI, resolve only the addressed review threads with linked
  evidence, obtain the required approval, and merge through ordinary branch
  protection. Never force-push, retarget, or bypass the review gate.
- Current downstream priorities after #4119: land the parent-first stack;
  shepherd ground/tee parity PR #4325; finish the real PyQt/React multi-
  viewport compositor (#4225); then finish measured plot geometry/export
  parity (#4224). Open epic checklists remain acceptance ledgers and must not
  be closed merely because partial PR evidence exists.

## Where This Repo Is Headed

Tools is the D-sorganization fleet's shared engineering-tools monorepo (45+
tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). The current
center of gravity is `src/rate_of_closure`, being grown from a single
closure-rate calculator into a full swing → impact → ball-flight simulation
platform under **Repository_Management#1390** (this handoff rollout) and a
stack of golf-simulation epics:

| Epic                                                                          | Status (one line)                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #4103 — Swing–Impact–Ball-Flight Simulation Platform                          | Phases 0-6 implemented on branch `feat/impact-simulation-platform`, consolidated into PR **#4119** (open, auto-merge armed, awaiting review). Phase 7 (WASM web parity swap, Pages CI) still open.                  |
| #4120 — Investigation & Variation Suite (plotting/viewers/Monte Carlo/help)   | V1-V4 implemented, stacked on #4119, consolidated into PR **#4124** (open, draft-for-review, no auto-merge yet — targets `feat/investigation-suite`, itself stacked on #4119).                                      |
| #4125 — Realistic Clubs/Kinetics/Putting/Public Release Mgmt/Showcase Styling | H1-H7 implemented, stacked on #4124, consolidated into PR **#4129** (open, draft-for-review, targets `feat/course-showcase`, stacked on #4124). H5 (public release-management repo) is cross-repo, not yet started. |
| #4130 — Impact-Interval Club Dynamics (contact-interval rigid-body model)     | Foundation epic only (F1 formulation doc not yet started); no PR yet. Next major physics wave after #4125 lands.                                                                                                    |

The separate shared Club Builder epic #4146 is active. Its first dependency
slice, #4147, lives on `feat/4147-club-builder-core` and establishes the
UI-independent assembly mass/CG/inertia, frame, length-datum, and persistence
contracts that the later shaft, CAD, export, fitting, and UI issues consume.

See `src/rate_of_closure/AGENT_HANDOFF.md` for the detailed stack breakdown
and architecture pointers for this tool specifically.

## Must-Read Architecture Pointers

1. `CLAUDE.md` — repo-wide conventions, CI gate list, cross-repo dependency
   rules (Tools is a leaf dependency; UpstreamDrift and Gasification_Model
   consume it).
2. `docs/architecture/CANONICAL_TOPOLOGY.md` — canonical repo topology policy.
3. `SPEC.md` — living specification; §12 Change Log requires a dated row for
   every PR touching `src/` (enforced by `spec-check.yml`, see gates below).
4. `src/rate_of_closure/AGENT_HANDOFF.md`, `src/pendulum_simulator/AGENT_HANDOFF.md`,
   `src/rotation_converter/AGENT_HANDOFF.md` — per-tool handoff docs.
5. `docs/AGENT_HANDOFF_TEMPLATE.md` — template for adding a handoff doc to a
   new tool.

## In-Flight Branches (what stacks on what)

```
main
 └─ feat/impact-simulation-platform   (PR #4119, epic #4103, auto-merge armed)
     └─ feat/investigation-suite      (PR #4124, epic #4120, stacked on #4119)
         └─ feat/course-showcase      (PR #4129, epic #4125, stacked on #4124)
docs/agent-handoff-1390               (this branch, off origin/main, Repository_Management#1390)
```

## Gate Commands (repo-wide)

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m contract                     # API contract tests (downstream-facing)
python3 -m pytest -m integration --timeout=60     # cross-repo integration
```

SPEC freshness (CI job `spec-freshness` in `.github/workflows/spec-check.yml`):
any PR touching `src/**`, `tests/**`, `config/**`, `pyproject.toml`,
`Cargo.toml`, `package.json`, or `requirements.txt` must also modify
`SPEC.md` in the same PR, or carry the `spec-exempt` label. Runs on the
`d-sorg-fleet` self-hosted runner.

## Do-Not List

- Do not modify public function signatures in `src/shared/python/**` without
  opening coordinated migration issues in UpstreamDrift and Gasification_Model
  (see `CLAUDE.md` Cross-Repo Dependencies).
- Do not import across package boundaries (e.g. `signal_processing_studio`
  importing from `sidekick.process_calculators`) — LoD is enforced.
- Do not exceed the 500-LOC file budget on new/modified files in the golf-sim
  packages (`rate_of_closure`, `swing_sim`, `swing-core`) — sub-package,
  don't grow monoliths.
- Do not use `git commit --no-verify` / `--push --no-verify` to bypass hooks;
  see `CLAUDE.md` Hook bypass policy.
- Do not regenerate the sidekick API baseline (`tests/sidekick_api_baseline.json`)
  without coordinating a breaking-change migration.
- Do not merge #4124 or #4129 ahead of their base (#4119, #4124 respectively)
  — they are stacked and will conflict/duplicate SPEC.md sections if merged
  out of order.
- Do not hand-roll a GitHub Pages deploy workflow for `rate_of_closure/web`
  yet — Phase 7 of #4103 owns this; today Pages hosting elsewhere in the repo
  (e.g. `unit_converter`) is done via manual branch-folder publish, not CI.

## Short-Term Roadmap (ordered)

1. Land PR #4119 (base platform) — currently the long pole; everything else
   stacks on it.
2. Get #4124 out of draft-for-review and merge into `feat/investigation-suite`
   → cascades onto #4119.
3. Get #4129 out of draft-for-review and merge into `feat/course-showcase`
   → cascades onto #4124.
4. Start #4130 Phase F1 (formulation document) once #4125's stack is in.
5. Phase 7 of #4103: WASM swap for the web mirror + real Pages CI deploy for
   `rate_of_closure/web`.
6. #4125 H5: stand up the public release-management repo (cross-repo, not
   started).
