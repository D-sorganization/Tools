"""Subprocess bootstrap that reproduces the cross-repo consumer ``sys.path``.

This repo is editable-installed in CI/dev venvs, which registers a meta-path
finder (and/or ``.pth`` entries) that map *bare* top-level names such as
``rotation_converter`` and ``contracts`` onto ``src/``. That convenience is
exactly what cross-repo consumers (e.g. UpstreamDrift) do **not** have: they
put only the repository root on ``sys.path`` and import under the ``src.``
namespace.

To make the import-contract assertion faithful, this bootstrap strips those
conveniences before the target import runs:

* removes editable-install meta-path finders,
* removes ``src``/``src/shared/python`` style shim entries from ``sys.path``
  (keeping stdlib and ``site-packages`` so real third-party deps still load),
* clears import caches and purges any already-bound bare modules,
* inserts only the repository root.

Usage (in a subprocess)::

    python tests/_import_contract_bootstrap.py <repo_root> <dotted.module>
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _norm(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _is_src_shim(path: str, repo_root: str) -> bool:
    norm = _norm(path)
    root = _norm(repo_root)
    if "site-packages" in norm or "dist-packages" in norm:
        return False
    # Shim entries are the repo's own src trees added to sys.path directly.
    return norm == f"{root}/src" or norm.startswith(f"{root}/src/")


def main() -> int:
    repo_root = sys.argv[1]
    target = sys.argv[2]

    # 1. Keep only the three standard import-system finders. Editable-install
    #    and virtualenv finders register custom meta-path finders that resolve
    #    bare top-level names (``rotation_converter``, ``contracts``, ...) onto
    #    ``src/`` — exactly the convenience a cross-repo consumer lacks.
    import importlib.machinery as _machinery

    _standard = (
        _machinery.BuiltinImporter,
        _machinery.FrozenImporter,
        _machinery.PathFinder,
    )
    sys.meta_path = [finder for finder in sys.meta_path if finder in _standard]

    # 2. Drop the repo's src shim path entries (keep stdlib + site-packages).
    sys.path = [p for p in sys.path if not _is_src_shim(p, repo_root)]
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()

    # 3. Purge any already-bound bare top-level modules that the editable
    #    finder may have eagerly registered, so the import is resolved fresh.
    for name in list(sys.modules):
        mod = sys.modules.get(name)
        origin = getattr(getattr(mod, "__spec__", None), "origin", None)
        if origin and _is_src_shim(str(Path(origin).parent), repo_root):
            del sys.modules[name]

    # 4. Consumer contract: only the repo root on the path.
    sys.path.insert(0, repo_root)

    importlib.import_module(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
