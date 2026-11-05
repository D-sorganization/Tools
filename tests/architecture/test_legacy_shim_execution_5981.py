"""Execute the legacy ``upstream_drift_tools`` shim in-process (Tools #5981).

``config/coverage_policy.json`` ratchets ``src/shared/python/upstream_drift_tools``
at 100%. Its only module is a compatibility shim that the import-alias finder
never executes in-process (``upstream_drift_tools`` is served by the alias
finder, and #5737 moved the real-shim check into a subprocess that coverage does
not see), so the nightly scored the package 0.0% although its tests passed.

This test runs the real shim file in the measured process, then restores the
interpreter-global import state so it cannot leak into other tests (#5737).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

SHIM = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "shared"
    / "python"
    / "upstream_drift_tools"
    / "__init__.py"
)


@pytest.fixture()
def preserved_import_state():
    """Snapshot and restore ``sys.modules`` and ``sys.meta_path``."""
    modules = dict(sys.modules)
    meta_path = list(sys.meta_path)
    try:
        yield
    finally:
        sys.meta_path[:] = meta_path
        sys.modules.clear()
        sys.modules.update(modules)


def test_shim_file_exists() -> None:
    assert SHIM.is_file()


@pytest.mark.usefixtures("preserved_import_state")
def test_shim_executes_and_aliases_canonical_sidekick() -> None:
    import shared.python.sidekick as canonical

    with pytest.warns(DeprecationWarning, match="upstream_drift_tools is deprecated"):
        namespace = runpy.run_path(
            str(SHIM), run_name="shared.python.upstream_drift_tools"
        )

    assert namespace["_canonical"] is canonical
