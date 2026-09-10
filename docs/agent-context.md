# Local Agent Context

## Purpose

The `fleet-agent-context` package provides a small, source-backed navigation
layer for coding agents. It combines repository-owned component metadata with
public symbols, directed integrations, contracts and executable test references.
Git owns the documents. Local Markdown, HTML, CLI and optional MCP are views of
the same catalog. No hosted platform, model API, embedding database or paid
subscription is needed. The package has no required runtime dependencies beyond
Python 3.11+ and an installed Git executable.

## Installation

Install the small package from the Tools checkout, separately from the full
engineering application and its frontend build:

```bash
python3 -m pip install ./packages/agent-context
```

In an application with the Tools submodule:

```bash
git submodule update --init vendor/ud-tools
python3 -m pip install ./vendor/ud-tools/packages/agent-context
python3 -m agent_context --root . status
```

Use a project virtual environment. Reinstall after advancing the provider pin.
An editable install is suitable for provider development; the application must
still verify the actual pinned provider before trusting shared-contract claims.

## Navigation

```bash
agent-context --root . search "motion pipeline"
agent-context --root . context motion-pipeline --max-chars 16000
agent-context --root . evaluate
agent-context --root . check
```

Search ranks component descriptions, paths, tags and public entry-point names.
It does not claim a complete semantic code graph. Context includes providers,
consumers, contract and test paths, current source excerpts, line ranges and
SHA-256 citations. Output budgets include serialized JSON. The status identifies
the exact worktree, revision, source digest, implementation and dependency state.
Unregistered components require direct source discovery; an empty search is not
proof that an implementation does not exist.

## Maintained Inputs and Generated Views

Maintain `docs/agent_context/catalog.json` and integration documents in Git.
Catalog version 1 requires components with `id`, `title`, `summary`, `owner`,
`status`, `sources`, `documentation`, `tests`, `entrypoints` and `tags`.
Statuses are `implemented`, `partial`, `proposed`, `deprecated` and `unsupported`. A relation
has `id`, `provider`, `consumer`, `kind`, `contract`, `inputs` and `tests`.
Sources may name files or directories; all other evidence names real files.
Python entry points identify declaration paths and qualified symbols without
importing them. Inventories reference existing JSON records by path and key.
Dependencies identify Git submodules, not arbitrary sibling checkouts.

Contracts contain populated Responsibilities, Data Contract, Lifecycle and
Failures, Evidence, and Rationale sections. Explain public behavior, ownership,
units/frames, schema, examples, failures and relevant tests. Preserve existing
engineering-manual authority and approval requirements.

After inspecting a changed boundary and executing its relevant tests:

```bash
agent-context --root . review pipeline-api --rationale "Reviewed request mapping, failure classification and API contract test results."
agent-context --root . render
agent-context --root . check
```

Review declarations bind exact boundary files, contract text, relation metadata
and test source hashes. Rendering never renews a review. A declaration is not
proof that tests ran or that a scientific calculation is qualified; CI and PR
review provide separate evidence. Commit the generated `README.md` and
`index.html`. Open HTML locally or serve the repository with a local static
server; it has no CDN or remote dependency. Obsidian may open the same Markdown
as an optional reader without owning another copy.

## Freshness and Failure Recovery

Every query rereads registered source hashes, including dirty content and file
membership. It checks for changes during inspection, isolates caches per
worktree and rejects missing or malformed evidence. UTF-8 LF/CRLF is normalized.
The `.codemap/context.json` cache is disposable and never serves as authority.
If a source changes mid-query, retry; if a pin or contract is unverified, inspect
the real source and resolve that discrepancy before claiming the integration.

The existing CodeMap is complementary lexical symbol search. It now rejects
stale symbol queries, reports partial parser coverage and tracks deleted files
and changed working directories. Use `codemap --repo . rebuild` to recover.
`repo_summary().freshness` exposes diagnostics; lexical caller matching remains
heuristic. Missing optional language parsers cannot establish an authoritative
empty result. A rebuild with errors exits with status 2.

## Optional MCP

```bash
python3 -m pip install './packages/agent-context[mcp]'
agent-context-mcp --root /absolute/path/to/application
```

Configure an agent's local stdio server with that executable and explicit root.
It exposes only `search_components`, `get_component_context` and
`context_status`. Retrieval may replace its local cache but cannot modify catalog
documents through MCP. It neither executes commands found in documentation nor
sends messages to other agents. Reuse the fleet presence/mailbox and existing
handoff/development logs for coordination.

## Validation and Fleet Adoption

The unit suite includes real MCP stdio, stale-source, malformed-schema, pin,
boundary-review, output-budget and deterministic-render tests. Navigation tasks
in `navigation.json` specify a query, expected component, source and optional
consumer. Evaluation reports accuracy, response size and elapsed time; these are
retrieval measurements, not a claim of end-to-end developer productivity.

See the [Fleet Adoption Guide](https://github.com/D-sorganization/Repository_Management/blob/main/docs/agent-context.md)
and epic [#1629](https://github.com/D-sorganization/Repository_Management/issues/1629).
