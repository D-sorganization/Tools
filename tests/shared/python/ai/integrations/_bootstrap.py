"""Shared bootstrap helpers for isolated AI integration client tests."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

_PACKAGE_STUBS: tuple[tuple[str, str | None], ...] = (
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
    ("src.shared.python.ai.exceptions", None),
    ("src.shared.python.ai.types", None),
)


def _ensure_module(name: str, rel_path: str | None, root: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if rel_path is not None:
        module.__path__ = [str(root / rel_path)]  # type: ignore[attr-defined]
    return module


def bootstrap_integration_client_test(root: Path) -> None:
    """Install lightweight module stubs required by isolated integration tests."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for module_name, rel_path in _PACKAGE_STUBS:
        _ensure_module(module_name, rel_path, root)

    logging_config = sys.modules["src.shared.python.logging_pkg.logging_config"]
    logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
    logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

    exceptions = sys.modules["src.shared.python.ai.exceptions"]
    if not hasattr(exceptions, "ToolExecutionError"):
        exceptions.ToolExecutionError = Exception  # type: ignore[attr-defined]

    ai_types = sys.modules["src.shared.python.ai.types"]
    if not hasattr(ai_types, "ToolResult"):
        ai_types.ToolResult = dict  # type: ignore[attr-defined]
