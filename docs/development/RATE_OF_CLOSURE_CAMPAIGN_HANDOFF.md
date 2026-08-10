# Rate of Closure Campaign Handoff

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
