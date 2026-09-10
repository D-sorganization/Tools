# codemap — repo-aware code map

A tree-sitter symbol index + SQLite FTS5 search layer that lets the in-app
chat (and external coding agents) answer questions like "where is
`apply_theme` called from?" without re-scanning the tree every prompt.

Full design: [`SPEC.md` codemap package baseline](../SPEC.md#12-change-log).

## Install

```bash
pip install -e ".[codemap]"            # core CLI + indexer
pip install -e ".[codemap,codemap-mcp]"  # also install MCP server
```

The canonical implementation is `src/shared/python/codemap/`. Applications
consume Tools through their pinned `vendor/ud-tools` checkout; do not copy or
edit the vendor implementation. The smaller `fleet-agent-context` package has
no CodeMap parser dependencies: install this optional environment separately.
For reviewed module relationships, see [Local Agent Context](agent-context.md).

### Windows note

`tree_sitter_languages` is incompatible with `tree_sitter >= 0.22` on
Windows wheels at the time of writing, so this package depends on the
per-language packages (`tree_sitter_python`, `tree_sitter_javascript`,
`tree_sitter_typescript`, `tree_sitter_rust`, `tree_sitter_markdown`)
instead. They all ship Windows wheels.

## First rebuild

```bash
cd /path/to/repo
codemap rebuild
```

This walks the repo (respecting `.gitignore`), parses every supported file
(`.py`, `.js`, `.mjs`, `.ts`, `.tsx`, `.rs`, `.md`), and writes
`.codemap/index.db` + `.codemap/manifest.json`. The `.codemap/` directory
is gitignored.

Historical design targets were cold rebuild < 10 s, search p50 < 30 ms and
DB < 50 MB. These are targets, not current benchmark results. Source freshness
verification adds work proportional to the supported corpus; measure the actual
checkout and environment before making performance claims.

## Search

```bash
codemap search "wgs reactor"
codemap search "apply_theme" --kind function -k 5
codemap who-calls ChatDockWidget._on_message
codemap info
codemap export               # writes .codemap/exports/code_map.jsonl.gz
```

## Incremental rebuild

```bash
codemap rebuild --since HEAD~1
```

The revision is an incremental hint. Rebuild also reconciles actual worktree
membership and content, including dirty, new and deleted supported files.
Unchanged source hashes skip parsing only when parser and dependency identity
still match. Parser failures are recorded and a partial rebuild exits with status 2.

Before using symbol results, CodeMap checks source hashes, membership, schema,
parser implementation and dependency versions. Stale queries fail rather than
returning an apparently current graph. `repo_summary().freshness` reports the
reason. Rebuild or inspect source directly; an incomplete graph cannot prove that
there are no callers or imports. Keep each worktree's `.codemap/` disposable and
ignored. A background refresh or watcher is an optimization, not merge enforcement.

Qt translation catalogs also use `.ts`. CodeMap recognizes well-formed XML with
a `TS` root as `qt-translation` resources and emits no code edges for them. Their
content still participates in freshness checks. Malformed XML, entity-bearing
documents and invalid TypeScript still fail validation. The existing `defusedxml`
runtime dependency parses these resources; its version is part of index identity.

## Watcher daemon

For on-save reindexing (debounced 500 ms):

```bash
codemap-watch
```

Inspect `.codemap/watcher.log` for failures. Queries still enforce freshness.

## MCP integration

The `codemap-mcp` console script exposes `search_code`, `get_symbol`,
`who_calls`, `imports_of`, and `repo_summary` as MCP tools over stdio.

### Claude Code (`.mcp.json`)

```json
{
  "mcpServers": {
    "codemap": {
      "command": "codemap-mcp",
      "env": { "CODEMAP_REPO_ROOT": "/path/to/repo" }
    }
  }
}
```

### Codex (`~/.codex/config.toml`)

```toml
[mcp_servers.codemap]
command = "codemap-mcp"
env = { CODEMAP_REPO_ROOT = "/path/to/repo" }
```

If `CODEMAP_REPO_ROOT` is unset the server falls back to the current
working directory and walks up to find the enclosing git repo.

## Python API

The in-app chat backend imports `codemap.api` directly:

```python
from codemap import search_code, who_calls, get_symbol, repo_summary

for hit in search_code("convert kinetic refs to json", k=5):
    print(hit.symbol.qualified, hit.symbol.path, hit.score)

callers = who_calls("ChatDockWidget._on_message")
```

## Graph Navigation

`who_calls` identifies candidate callers and `imports_of` exposes declared
imports. The Python API also offers `neighbors(symbol, hops=1, repo_root=...)`
for a focused inbound/outbound neighborhood. Lexical matching is heuristic;
dynamic dispatch, plugins and runtime configuration need source inspection.
Use the reviewed component graph for integration meaning and cite actual code
and tests before changing an interface. No graph establishes scientific approval.

## Schema (summary)

| Table         | Notes                                                           |
| ------------- | --------------------------------------------------------------- |
| `files`       | one row per indexed file; path, language, content hash, imports |
| `symbols`     | one row per function/class/method/struct/heading                |
| `symbols_fts` | FTS5 virtual table over name + qualified + sig + docstring      |
| `meta`        | schema version, etc.                                            |

Source slices are **not** stored — only line ranges + a content hash (BLAKE3
when available, otherwise BLAKE2b) so
incremental rebuilds can skip unchanged symbols. Queries return paths +
line ranges; the chat backend opens the file on click-through.

## Embeddings (future)

The design calls for an opt-in semantic search layer (`onnxruntime` +
`gte-small`, stored in a `sqlite-vec` virtual table). That layer is
**not** in this release; the placeholder lives at
`codemap/embeddings.py` and a future PR will wire it in behind
`codemap rebuild --embed`.

## Tests

```bash
python -m pytest tests/unit/codemap/
```

Covers schema init/idempotency, golden parser tests for Python + TS,
cold + incremental rebuild on a 3-file fixture, and the public API
(`search_code`, `get_symbol`, `who_calls`, `imports_of`, `repo_summary`,
`neighbors`).
