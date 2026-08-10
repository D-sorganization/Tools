# Rate of Closure Campaign Handoff

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
