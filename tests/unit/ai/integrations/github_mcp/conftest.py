"""Local conftest for GitHub MCP integration unit tests.

Mirrors ``tests/unit/ai/mcp/conftest.py`` — bootstraps the
``src.shared.python.ai.integrations.github_mcp.*`` package tree so test
imports of the form ``from src.shared.python.ai.integrations.github_mcp ...``
resolve correctly under pytest's rootdir-driven discovery.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PACKAGE_STUBS: list[tuple[str, str]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.mcp", "src/shared/python/ai/mcp"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
    (
        "src.shared.python.ai.integrations.github_mcp",
        "src/shared/python/ai/integrations/github_mcp",
    ),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    existing = sys.modules.get(_mod_name)
    target = str(_REPO_ROOT / _rel_path)
    if existing is None:
        stub = types.ModuleType(_mod_name)
        stub.__path__ = [target]
        sys.modules[_mod_name] = stub
    else:
        existing_path = list(getattr(existing, "__path__", []) or [])
        if target not in existing_path:
            existing_path.insert(0, target)
            existing.__path__ = existing_path  # type: ignore[attr-defined]
