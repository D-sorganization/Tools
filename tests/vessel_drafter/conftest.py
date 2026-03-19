"""Configure sys.path and module mocks for vessel_drafter tests.

build123d is an optional CAD dependency not installed in the test environment.
We install a MagicMock-based stub in sys.modules before vessel_drafter submodules
are imported so that module-level ``from build123d import ...`` statements
succeed without the real library.

We also ensure that the correct vessel_drafter package
(src/vessel_drafter/python/vessel_drafter) is loaded, not the shallow
src/vessel_drafter/ wrapper that is on sys.path via the ``src`` pythonpath
entry. The workaround: insert the correct python path at index 0 and evict
any stale cached vessel_drafter modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Ensure the correct vessel_drafter package is on sys.path (index 0)
# ---------------------------------------------------------------------------

_VESSEL_DRAFTER_PYTHON = (
    Path(__file__).resolve().parent.parent.parent / "src" / "vessel_drafter" / "python"
)

if _VESSEL_DRAFTER_PYTHON.exists():
    _path_str = str(_VESSEL_DRAFTER_PYTHON)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
    elif sys.path.index(_path_str) != 0:
        sys.path.remove(_path_str)
        sys.path.insert(0, _path_str)

    # Evict stale vessel_drafter module cached from src/vessel_drafter/__init__.py.
    # Only evict if the cached __file__ is NOT under the correct python path.
    _stale_keys = [
        key
        for key, mod in list(sys.modules.items())
        if key == "vessel_drafter" or key.startswith("vessel_drafter.")
        if getattr(mod, "__file__", None)
        and not str(getattr(mod, "__file__", "")).startswith(_path_str)
    ]
    for key in _stale_keys:
        del sys.modules[key]

# ---------------------------------------------------------------------------
# build123d stub — installed before vessel_drafter.* is imported.
# MagicMock as the module object satisfies any attribute access or call.
# ---------------------------------------------------------------------------

if "build123d" not in sys.modules:
    _build123d_mock = MagicMock()
    _build123d_mock.__name__ = "build123d"
    _build123d_mock.__spec__ = None
    _build123d_mock.__package__ = "build123d"
    sys.modules["build123d"] = _build123d_mock
