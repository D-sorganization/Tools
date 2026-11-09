# Generated Baselines: Reproducibility Contract & Bounded Views

**Issue:** Tools #5995, #5998, #5999  
**Tooling:** `python -m src.tools.generated_baselines --check|--print-spec-baseline|--summary|--query [--offset N] [--limit N] [--max-bytes B] [--full]`

---

## 1. Inventory & Baseline Classification

Tools tracks four large (>1 MB) data/specification files in repository tree:

- `.benchmarks/baseline.json` (2.1 MB): pytest-benchmark timing oracle (environment-dependent).
- `tests/sidekick_api_baseline.json` (1.1 MB): Sidekick public AST symbol oracle (deterministic).
- `tests/api_baselines/swing_sim_api_baseline.json` (1.0 MB): Swing Sim public AST oracle (deterministic).
- `SPEC.md` (1.0 MB): Fleet specification and historical changelog ledger (deterministic).

Repository blob paths total 91.26 MiB; these four files sum ~5.3 MiB (~5.8%).

## 2. Reproducibility & Governance

1. **Deterministic vs Environment-Dependent:** API baselines and SPEC changelogs are deterministic AST/fragment extractions. Benchmark baselines reflect hardware timing on clean benchmark runners.
2. **Reviewed Updates Only:** Baseline regeneration flags (`--regenerate-api-baseline`, `--benchmark-json`, `--print-spec-baseline`) require review approval from respective teams (Sidekick Architecture, Swing Sim, Maintainers).
3. **Integrity Enforcement:** CI and local developers run `python -m src.tools.generated_baselines --check`. JSON baselines enforce exact full-file SHA-256 digests; `SPEC.md` (`spec_changelog_collation`) pins SHA-256 digests of non-changelog sections, the change-log header, and the reviewed baseline row tail while requiring any prepended change-log rows to match `changes/*.md` collation provenance in `docs/development/DEVELOPMENT_LOG.md`. Intentional edits to non-changelog `SPEC.md` sections refresh the anchor via `--print-spec-baseline`.

## 3. Bounded Context Views for AI Agents

- **The Problem:** Reading 2.1 MB raw JSON consumes ~500k model tokens. In `.benchmarks/baseline.json`, over 98% of payload consists of raw timing sample arrays (`stats["data"]`).
- **Bounded Tooling Solution:** `src/tools/generated_baselines.py` exposes lightweight views:
  - `--summary`: Strips raw samples, exposing key statistics (`mean`, `min`, `max`, counts) under <2 KB.
  - `--query <PATTERN> --baseline <ID>`: Enforces an explicit serialized UTF-8 cap (`DEFAULT_MAX_QUERY_BYTES = 10_000`, default page limit `25`), reporting exact `match_count` / `total_matches`, `returned_matches`, `truncated`, `has_more`, `item_truncated`, and deterministic `next_offset` continuation (`--offset`, `--limit`, `--max-bytes`). Oversized single modules, symbols, or benchmarks are compacted within `max_bytes` so pagination never stalls; `--full` (`QueryOptions(full_data=True)`) provides an explicit untruncated route when needed.
  - Context footprint drops by >98% while full regression test oracles remain byte-identical on disk.

## 4. Storage & Ignore Evaluation

- **Git LFS:** Git LFS manages checkout bandwidth for binary blobs but does not reduce agent context consumption when working copies are hydrated. Git tracking is retained for complete offline regression testing.
- **Provider Search Ignores:** Standard ignore configurations (`.cursorignore`, search filters) shield benchmark and raw artifact dumps from recursive indexing, while `src/tools/generated_baselines.py` provides targeted queries.

## 5. Non-Regression Guarantees

No baseline deletion, no tolerance weakening, no arbitrary file splitting, no Sidekick packaging modifications, and no history rewriting.
