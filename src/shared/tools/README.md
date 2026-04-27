# Shared Tool Helpers

This directory hosts **cross-tool utilities** that are consumed by more than one
tool under `src/`. It exists to prevent duplication and to give small, generic
helpers a stable home that is not tied to any single product tree.

Related issue: Tools #1997 (500 LOC CI gate).

## When to put code here

Place a module in `src/shared/tools/` when:

- The logic is genuinely cross-cutting (two or more tools need it).
- It is free of UI framework imports (no PyQt6, no web framework).
- It has no upstream dependency on a specific tool package.
- You can describe it in a short docstring without referring to a product.

If the helper is specific to one tool, keep it in that tool's package. If it is
scientific / engineering library code shared with UpstreamDrift or
Gasification_Model, it belongs in `src/shared/python/` — not here.

## Conventions

All modules in this directory must follow these conventions:

1. **Type hints** on every public function and class. Use `from __future__
   import annotations` and prefer `collections.abc` types.
2. **Design by Contract (DbC).** Every public function validates its inputs
   and raises `TypeError` for wrong types and `ValueError` for out-of-range
   values. Document preconditions in the docstring.
3. **Under 200 LOC per module.** If a helper grows larger, split it by
   concern. The repo-wide 500 LOC CI gate applies here too, but the informal
   target for this directory is 200 LOC so helpers stay easy to reuse.
4. **No `print()` calls.** Use the `logging` module.
5. **No side-effects on import.** Modules should be safely importable from
   any tool without triggering network, file-system, or GUI work.
6. **Tests required.** Add tests under `tests/shared/tools/` mirroring the
   module path. New helpers without tests will not be accepted.

## Layout

```
src/shared/tools/
  README.md          (this file)
  <helper>.py        (one module per concern)
```

Sub-packages are permitted when a group of closely-related helpers grows past
a single module. Keep imports shallow: `from shared.tools import foo`, not
chains across more than two package levels.
