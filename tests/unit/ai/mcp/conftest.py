"""Local conftest for MCP unit tests.

Bootstraps the ``src.shared.python.ai.mcp.*`` package tree so test imports of
the form ``from src.shared.python.ai.mcp...`` (the dominant convention in this
repo) resolve correctly under pytest's rootdir-driven discovery.

This mirrors the pattern used by ``tests/shared/python/ai/test_adapter_contract.py``
to work around the fact that ``src/python/src`` is also on ``pythonpath`` and
carries its own ``__init__.py``, which shadows the implicit namespace ``src``
package rooted at the repository root.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PACKAGE_STUBS: list[tuple[str, str]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.mcp", "src/shared/python/ai/mcp"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    existing = sys.modules.get(_mod_name)
    target = str(_REPO_ROOT / _rel_path)
    if existing is None:
        stub = types.ModuleType(_mod_name)
        stub.__path__ = [target]
        sys.modules[_mod_name] = stub
    else:
        # Ensure existing module's __path__ includes our target so submodules
        # resolve regardless of which package object pytest already loaded.
        existing_path = list(getattr(existing, "__path__", []) or [])
        if target not in existing_path:
            existing_path.insert(0, target)
            existing.__path__ = existing_path  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Stub modules consumed transitively by ``tool_registry`` that have no
# concrete implementation in this repo yet (see ``test_adapter_contract``
# for the same workaround).
# ---------------------------------------------------------------------------
import logging as _logging  # noqa: E402

_logging_pkg = sys.modules.setdefault(
    "src.shared.python.logging_pkg",
    types.ModuleType("src.shared.python.logging_pkg"),
)
_logging_cfg_name = "src.shared.python.logging_pkg.logging_config"
_logging_cfg = sys.modules.get(_logging_cfg_name)
if _logging_cfg is None:
    _logging_cfg = types.ModuleType(_logging_cfg_name)
    sys.modules[_logging_cfg_name] = _logging_cfg
_logging_cfg.get_logger = _logging.getLogger  # type: ignore[attr-defined]
_logging_cfg.setup_logging = lambda *_a, **_kw: None  # type: ignore[attr-defined]
