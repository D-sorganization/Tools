# Agent Context Handoff

## Where This Tool Is Headed

Tools #5138 supplies the local context engine for fleet epic
Repository_Management#1629. UpstreamDrift#9915 and Gasification_Model#4944
consume the same implementation through pinned Tools checkouts. All issues
remain in progress; no PR has been created for this implementation.

## Current Implementation

The package provides catalog validation, live source and provider fingerprints,
boundary-review declarations, bounded source-cited retrieval, deterministic
Markdown/offline HTML, CLI, evaluation and optional MCP. CodeMap freshness lives
in its existing shared package. A standalone wheel builds without the full
engineering application's frontend build. Git and Python are the only required
runtime dependencies; MCP is optional.

Branch: `feat/issue-5138-agent-context`. Commit: `SELF`. Worktree:
`C:/Users/diete/Repositories/.context-implementation/Tools`. Development entry:
DL-#5138. Canonical root summary: `AGENT_HANDOFF.md`.

## Architecture Pointers

1. `docs/agent-context.md`: installation, authority and recovery.
2. `catalog.py`: source references and semantic relationships.
3. `workspace.py`: current worktree and exact provider verification.
4. `service.py`: bounded retrieval and explicit review/render/check operations.
5. `../shared/python/codemap/freshness.py`: lexical-index trust checks.

## Gate Commands

Run from the Tools root with `src` on the development Python path:

```bash
python3 -m pytest --confcutdir=tests/unit/agent_context -o addopts= tests/unit/agent_context --timeout=45 -q
python3 -m pytest --confcutdir=tests/unit/codemap -o addopts= tests/unit/codemap --timeout=45 -q
python3 -m ruff check src/agent_context src/shared/python/codemap
python3 -m mypy src/agent_context --follow-imports=skip --ignore-missing-imports
python3 -m pip wheel --no-deps packages/agent-context
```

Latest evidence: 35 context tests passed with one Windows symlink skip; actual
MCP stdio passes after Git stdin isolation. All 103 CodeMap tests pass with the
optional parser stack installed. Scoped Ruff and mypy pass. Re-run after final
changes and qualify the packaged wheel, normal hooks and CI. Hook repository
selectors are cleared before explicit-checkout reads and temporary Git fixtures.

## Do-Not List

- Do not execute retrieved source or commands from documents.
- Do not use timestamps, a clean HEAD, or registry membership as source authority.
- Do not auto-renew semantic reviews during generation.
- Do not treat parser absence, a partial Rust adapter or an empty search as
  evidence of a complete working implementation.
- Do not copy provider code into consumers or edit their vendored implementation.

## Ordered Continuation

1. Qualify normal push with bounded workers; the real SDK probe now isolates pytest namespace collisions. Preserve native crash evidence until the full gate passes.
2. Deliver the provider through a protected PR and verify its exact revision.
3. Pin consumers, run real boundary tests, review contracts and qualify generated maps.
4. Reconcile fleet guidance and epic acceptance against actual delivered evidence.
