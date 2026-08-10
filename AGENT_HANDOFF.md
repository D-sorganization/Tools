# Agent Handoff

## Active branch

- Branch: `feat/4144-variation-visualizations`
- Pull request: `#4167`
- Base: `feat/investigation-suite`
- Pre-repair head: `edaa56358a9ccf47809533fcab28e6415b336771`

## Current work

- The variation-visualization implementation remains the base-most open Rate
  feature in its stack.
- Protected Python 3.10 collection exposed a direct `datetime.UTC` import in
  `torque_profile_controller.py`. The repair routes `UTC` through the existing
  shared compatibility module and adds a complete Rate-source AST guard for
  direct imports plus unaliased and aliased module-attribute access.
- Local verification is green: 27 focused controller/history/AST tests and the
  complete Rate suite (554 tests) pass on Python 3.13; the compatibility export
  is verified with the real CPython 3.10.20 interpreter; Ruff check/format,
  focused pinned MyPy 1.13, detect-secrets, touched-file size, and diff checks
  pass.

## Release boundaries

- Do not treat cancelled file-size, dependency-download, missing toolcache, or
  PyO3 link-library jobs as passing evidence.
- Do not mark the draft ready or merge it until current-head protected checks
  complete and required review approves.
- Preserve the existing base; never force-push, retarget, admin-merge, or
  bypass required checks.

## Durable handoff policy

Every implementation commit must update this file,
`src/rate_of_closure/AGENT_HANDOFF.md`,
`docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md`, and `SPEC.md` in the
same commit, or explicitly record why no material handoff change occurred.
