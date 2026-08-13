# Rate of Closure Campaign Handoff

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
