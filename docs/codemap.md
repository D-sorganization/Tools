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

The package vendors as `src/shared/python/codemap/`; both UpstreamDrift and
Gasification_Model consume it through their existing symlink into Tools.

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

Performance budget: cold rebuild < 10 s/repo on the Tools fleet, search
p50 < 30 ms, DB < 50 MB without embeddings (per design §7).

## Search

```bash
codemap search "wgs reactor"
codemap search "apply_theme" --kind function -k 5
codemap who-calls ChatDockWidget._on_message
codemap info
codemap export --jsonl       # writes .codemap/exports/code_map.jsonl.gz
```

## Incremental rebuild

```bash
codemap rebuild --since HEAD~1
```

Calls `git diff --name-only HEAD~1..HEAD` and re-parses only the changed
files. Hash-based deduplication means even a full rebuild skips files
whose content hasn't changed.

A git `post-commit` hook running the above keeps the index current without
a daemon:

```bash
# .git/hooks/post-commit
#!/bin/sh
codemap rebuild --since HEAD~1 >/dev/null 2>&1 &
```

## Watcher daemon

For on-save reindexing (debounced 500 ms):

```bash
codemap-watch
```

Logs to `.codemap/watcher.log`. ~3 MB resident.

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

## Schema (summary)

| Table         | Notes                                                          |
| ------------- | -------------------------------------------------------------- |
| `files`       | one row per indexed file; path, language, blake3 hash, imports |
| `symbols`     | one row per function/class/method/struct/heading               |
| `symbols_fts` | FTS5 virtual table over name + qualified + sig + docstring     |
| `meta`        | schema version, etc.                                           |

Source slices are **not** stored — only line ranges + a blake3 hash so
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
