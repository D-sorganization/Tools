"""Shared bootstrap helpers for isolated AI integration client tests."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

# Namespace-package shims only. These give the ``src.shared.python.ai`` import
# chain a ``__path__`` so the real submodules (``exceptions``, ``types``,
# ``tool_registry``, the integration clients) load from disk. We deliberately do
# NOT stub ``ai.exceptions`` or ``ai.types`` here: the real modules import
# cleanly and other AI test modules (adapter contract, classify_error, the CLI
# adapters) import concrete names like ``AIConnectionError``/``AgentChunk`` from
# them. Inserting bare stubs would shadow the real modules in ``sys.modules`` for
# the rest of the collection and break those tests (regression seen when the
# stubs covered ``exceptions``/``types``).
_PACKAGE_STUBS: tuple[tuple[str, str | None], ...] = (
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
)


def _ensure_module(name: str, rel_path: str | None, root: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if rel_path is not None and not hasattr(module, "__path__"):
        module.__path__ = [str(root / rel_path)]  # type: ignore[attr-defined]
    return module


def bootstrap_integration_client_test(root: Path) -> None:
    """Install lightweight module stubs required by isolated integration tests."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for module_name, rel_path in _PACKAGE_STUBS:
        _ensure_module(module_name, rel_path, root)

    logging_config = sys.modules["src.shared.python.logging_pkg.logging_config"]
    if not hasattr(logging_config, "get_logger"):
        logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
    if not hasattr(logging_config, "setup_logging"):
        logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
