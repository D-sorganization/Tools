# Tools #5388 canonical inventory repair report

## Decision and scope

Root decision receipt 5915645743 accepted the exact canonical two-file inventory delta. The existing mapper semantics remain unchanged. The new browser test's lexical association with Rust `debounce` is not evidence that Rust code executes or is covered by this test. No mapper, test prose, generator design, tolerance, timeout, or CI behavior was changed for inventory.

## Baseline and result

- Starting HEAD: `3d4d3825ef72ba626cd89d66d2649508c1476810` (merged-main baseline includes `3c62bbdf20cf0a1ffc878aa6a76f6865143ff3b5`).
- Before canonical generation, inventory `--check` failed on `manuals/tools/manifests/module-inventory/entries-rust-core-file-watcher.json`.
- Canonical generation changed only that shard and `manuals/tools/manifests/module-inventory.json`.
- Shard: 23,063 -> 23,142 bytes; LF content hash `79d0aea5e1feb6b6ec19a4e4aa5510f84de86832da83d82f5a288ff00dc61730` -> `930bd995c95f6459d766e53ee52d54ef1cc69ee3338b4b6f05a0e5deaa832be4`.
- The shard retains five entries. Only `rust_core/file_watcher/src/debounce.rs` gains the path `src/web_applications/unit_converter/tests/test_keyboard_escape.py` in `traceability.test_paths` (5 -> 6). The descriptor hash updates to the new shard hash; entry count remains five.
- All other inventory projection was unchanged by canonical generation.

## Validation

- `py -3.12 -m pytest -q -n 0 tests/architecture/test_tools_module_inventory_contract.py` — 23 passed, one existing `qt_api` config warning. This includes the existing freshness CLI check, deterministic projection test, schemas, and stale-check behavior.
- `py -3.12 -m scripts.build_tools_module_inventory --check` — passed after canonical generation.
- `git diff --check` — passed.
- Browser tests were not repeated; prior accepted RED/GREEN evidence remains in the browser portability report.

## Files and publication status

Expected local commit scope: `.github/workflows/ci-standard.yml`, `tests/ops/test_ci_test_shards.py`, `docs/development/tools5388-browser-portability-report.md`, and the two generated inventory files above. Browser regression and SPEC changes are already in accepted ancestor commits; historical SPEC/HANDOFF rows were preserved. No push, PR-ready transition, merge, or manual CI rerun was performed.

The earlier browser report's description of the stale inventory as unrelated was corrected; the original inventory assessment report remains preserved as historical diagnostic evidence. Root review remains required before publication.
