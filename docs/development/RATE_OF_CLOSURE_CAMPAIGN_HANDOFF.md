# Rate of Closure Campaign Handoff

## 2026-08-11 pinned-Ruff propagation into D-plane visualization

- Immediate child PR `#4202` keeps base
  `feat/4162-wedge-impact-visualization`.
- Exact clean child head `ba4aa35cc384d51ed3aa52eb532a67e960669c27`
  normally merges exact parent `#4179` head
  `7e5dfecf569b39dbbf8cc2101c7426cbc53a2771` without rewriting history.
- The D-plane ndarray typing repair, frame-explicit geometry, pinned Ruff
  `0.14.10` files, and all visualization, turf, wedge, impact, variation,
  handoff, and specification histories remain additive. No scientific or UI
  behavior changes. Pinned Ruff `0.14.10` check/format verification and 129
  focused D-plane, impact, solver, kinetics, PyQt, and layout tests are green.
  Documentation, minimum-test, SPEC-version, and diff gates are also green.
  Protected current-head checks and parent-first release order remain open.

## 2026-08-10 PR #4202 D-plane ndarray typing repair

- The repair is based on exact published PR `#4202` head
  `b443fdbed7064c5db0320106013c8413e3e24356` and retains base
  `feat/4162-wedge-impact-visualization`.
- CI Standard run `31384810375`, job `93442745760`, reported two pinned MyPy
  1.13 `no-any-return` errors in private NumPy conversion/projection helpers.
  Explicit ndarray local boundaries resolve those errors without changing
  geometry, validation, frames, serialized contracts, or UI behavior.
- RED/GREEN evidence is exact: two errors reproduced before the edit and zero
  afterward. Twenty-four focused D-plane/impact tests, seven metadata/pre-push
  contract tests, scoped Ruff/Ruff-format/Black, docs governance, minimum-test,
  module-size, changed-file-size, and diff checks pass. Three exploratory
  CI-workflow contract tests retain unrelated older-branch toolcache/env drift;
  no workflow file is changed here.
- This repair does not alter the stacked base, publish the branch, change its
  draft state, or authorize a merge. Parent-first ordering, protected CI, and
  review remain required.

## 2026-08-10 Propagation into 3D D-plane geometry

- Immediate child PR `#4202` keeps base
  `feat/4162-wedge-impact-visualization`.
- Its original head `b4abec03bccfbdd87ddf91427159c5c2332c21dd`
  normally merges exact parent `#4179` head
  `6704a3e541a3e74c28b4a284530d1a21269dd340`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the typed,
  frame-explicit D-plane, spin-loft, visualization, and export contracts.
- Persisted D-plane layer controls are extracted into a focused helper so the
  simulation view satisfies the protected 500-line module budget without a
  behavior or compatibility-seam change.
- Combined-stack verification is green: 93 focused and 825 scoped Python tests
  (two optional `build123d` skips), 360 React tests and all web gates, real
  CPython 3.10.20 compilation/UTC, scoped Ruff/Black/MyPy, and repository
  governance gates. The exact parent's 12 unchanged `swing-core` tests remain
  applicable because this child has no Rust delta. The inherited 17-error broad
  MyPy Qt/NumPy baseline in 11 untouched files remains outside scope.
  Protected CI and required review remain release gates.
## 2026-08-11 pinned-Ruff propagation into wedge impact visualization

- Immediate child PR `#4179` keeps base `feat/4166-wedge-turf-physics`.
- Exact clean child head `ea7acebf033379d6beefd70eb51027ebd3d01be7`
  normally merges exact parent `#4178` head
  `188f491ccc88a335ad36afdd66b52289e2e24808` without rewriting history.
- Parent Ruff `0.14.10` formatting and all existing visualization, turf, wedge,
  impact, variation, handoff, and specification history remain additive. No
  scientific or UI behavior changes. Pinned Ruff `0.14.10` check/format
  verification and 130 focused impact-scene, solver, kinetics, PyQt/layout,
  wedge-clearance, and turf-model tests are green. Documentation, minimum-test,
  SPEC-version, and diff gates are also green. Protected current-head checks
  and parent-first release order remain open.

## 2026-08-10 Propagation into wedge impact visualization

- Immediate child PR `#4179` keeps base `feat/4166-wedge-turf-physics`.
- Its original head `0eb804e70887c788421332369e42792411aff55a`
  normally merges exact parent `#4178` head
  `bfa83aedc88ead380babc73a699377d98b971006`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the
  exact-event, locked-scale, exportable impact-scene contract.
- Combined-stack verification is green: 58 focused and 739 scoped Python tests
  (two optional `build123d` skips), 347 React tests and all web gates, real
  CPython 3.10.20 compilation/UTC, scoped Ruff/Black/MyPy, and repository
  governance gates. The exact parent's 12 unchanged `swing-core` tests remain
  applicable because this child has no Rust delta. The inherited 17-error broad
  MyPy Qt/NumPy baseline in 11 untouched files remains outside scope.
  Protected CI and required review remain release gates.
## 2026-08-11 pinned-Ruff propagation into wedge turf physics

- Immediate child PR `#4178` keeps base `feat/4161-wedge-ground-clearance`.
- Exact clean child head `ca567fe7d3fa48b1900ad3098045f4200cfe86a7`
  normally merges exact parent `#4174` head
  `3e1b44cf42f4c0838149e0bc8e88ce4cb79b72b0` without rewriting history.
- Parent Ruff `0.14.10` formatting and all existing turf, wedge, impact,
  variation, handoff, and specification history remain additive. No scientific
  or UI behavior changes. Workflow-pinned Ruff check/format, 127 focused tests,
  and documentation, minimum-test, SPEC-version, and diff gates are green.
  Protected current-head checks and parent-first release order remain open.
## 2026-08-10 Propagation into wedge turf physics

- Immediate child PR `#4178` keeps base `feat/4161-wedge-ground-clearance`.
- Its original head `aaae3f73e17dbfaad5cca1dc6f49559b3aebe9d5`
  normally merges exact parent `#4174` head
  `9ea93e92563280ec34bca682ad44d7409edd7a02`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the passive,
  provenance-gated turf-contact model and explicit force-coupling boundary.
- Combined-stack verification is green: 56 focused and 732 scoped Python tests
  (two optional CAD-dependency skips), real CPython 3.10.20 checks, scoped
  static analysis, and repository governance gates. With no web or Rust delta,
  the exact parent's green 345 React and 12 Rust tests remain applicable. The
  inherited 17-error broad MyPy Qt/NumPy baseline in 11 untouched files remains
  outside scope.
## 2026-08-11 pinned-Ruff propagation into wedge ground clearance

- Immediate child PR `#4174` keeps base `feat/4163-impact-inspector`.
- Exact clean child head `01ecf9a7b1922d1609fb99093226799a0b564704`
  normally merges exact parent `#4173` head
  `bd48852d303db6281ed5891d4a271d99e76a94e6` without rewriting history.
- Parent Ruff `0.14.10` formatting and all existing wedge, impact, variation,
  handoff, and specification history remain additive. No scientific or UI
  behavior changes. Workflow-pinned Ruff check/format, 98 focused tests, and
  documentation, minimum-test, SPEC-version, and diff gates are green.
  Protected current-head checks and parent-first release order remain open.
## 2026-08-10 Propagation into wedge ground clearance

- Immediate child PR `#4174` keeps base `feat/4163-impact-inspector`.
- Its original head `880a6465fc872cf3d6650283db154ddc41793a31`
  normally merges exact parent `#4173` head
  `9ddaff3b6bca542fd7a2befc7d7b0ae53910a60a`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the swept
  wedge ground-clearance model, persistence, PyQt, and React surfaces.
- Combined-stack verification is green: 56 focused and 703 scoped Python tests
  (two optional CAD-dependency skips), 345 React tests and all web gates, 12
  Rust tests, real CPython 3.10.20 checks, scoped static analysis, and
  repository governance gates. The inherited 17-error broad MyPy Qt/NumPy
  baseline in 11 untouched files remains outside scope.
## 2026-08-11 pinned-Ruff propagation into impact inspector

- Exact repaired parent `#4167` head
  `91dc2174578a4fc472907d7141ca44c9ef36d3ab` is merged normally into child
  `#4173`; branch/base identities remain unchanged.
- The merge carries only the documented five-file Ruff `0.14.10` mechanical
  formatting delta plus its canonical handoff/spec evidence. Impact-inspector
  and variation behavior are unchanged.
- Protected current-head checks and the later investigation-suite carrier into
  the root branch remain mandatory release gates.
## 2026-08-10 Propagation into impact inspector

- Immediate child PR `#4173` keeps base `feat/4144-variation-visualizations`.
- Its original head `3c43955aaeb3964ff8c3ef2748d626baae518b76`
  normally merges exact parent `#4167` head
  `22b66b560652b78de84141344c4ddd9a92a83b26`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC compatibility repair and its AST guard are inherited
  additively alongside the existing shared wedge impact-inspector contract.
- Combined-stack verification is green: 63 focused and 562 total Rate Python
  tests; 334 React tests plus type-check, lint, and production build; 12 Rust
  tests; real CPython 3.10.20 compile/UTC checks; scoped static analysis and
  repository governance gates. The broad MyPy sweep retains 17 pre-existing
  Qt/NumPy typing findings in 11 untouched files. Current-head protected CI and
  required review remain pending release evidence.

## Dependency position

PR `#4167` (`feat/4144-variation-visualizations`) is the base-most open Rate
feature above the already merged `feat/investigation-suite` carrier. Later
wedge, D-plane, launch-monitor, workspace, wind, capability, and ground work
depends on this line and must receive any repair through ordinary parent
propagation; child branches must not be rewritten.

## 2026-08-11 pinned-Ruff repair

- Exact PR `#4167` head `3c19aaa9d3e812e4659053735a2955d62a080d34`
  carries the same five changed Python blobs that fail Ruff `0.14.10` format
  checking on immediate child `#4173`.
- The five files are mechanically formatted with the workflow-pinned version;
  there is no scientific, persistence, API, schema, test, or UI behavior
  change and no claim that variation epic `#4144` is complete.
- The repair must still pass current-head protected checks and travel through
  the ordinary investigation-suite carrier before the root PR can reach main.

## 2026-08-10 Python 3.10 repair

- Protected CI at exact pre-repair head
  `edaa56358a9ccf47809533fcab28e6415b336771` collected 13 Rate test modules
  unsuccessfully because `datetime.UTC` does not exist on Python 3.10.
- The torque-profile controller now consumes the repository's existing
  `shared.python.compatibility.UTC` export.
- A source-tree AST guard rejects future direct imports and unaliased or
  aliased `datetime.UTC` module-attribute access anywhere under
  `src/rate_of_closure`.
- Local evidence is green: 27 focused controller/history/AST tests and the
  complete 554-test Rate suite on Python 3.13; real CPython 3.10.20 compatibility
  import; Ruff check/format; focused pinned MyPy 1.13; detect-secrets;
  touched-file size and diff checks.

## Truthful release state

This is an actionable compatibility repair, not completion of issue `#4144` or
the variation epic. Current-head protected CI, required review, dependency
propagation, and ordinary merges remain required. Runner download timeouts,
missing toolcache/link libraries, cancelled jobs, and queued jobs are tracked
as infrastructure and are never counted as green evidence.

Every implementation commit must update this file, both other canonical
handoffs, and `SPEC.md`, or explicitly record no material handoff change and
the reason.
