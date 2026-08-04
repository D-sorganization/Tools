"""Guard: the backend-security suite must not silently skip on real defects.

``test_backend_security.py`` wraps the first-party backend import (``main``,
``project_import``, ``models``) in a ``try/except`` that skips the whole module
when the backend is unavailable. That handler must catch ONLY
``ModuleNotFoundError`` -- a genuinely missing dependency. If it catches a bare
``Exception``, a ``NameError`` / ``SyntaxError`` / ``ImportError`` inside the
security-critical backend would silently skip the entire auth/zip-bomb suite
instead of failing loudly (issue #3745).

This test parses the file with ``ast`` so it runs regardless of whether the
backend itself is importable in the current environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TARGET = Path(__file__).with_name("test_backend_security.py")


def _module_level_handlers(tree: ast.Module) -> list[ast.ExceptHandler]:
    handlers: list[ast.ExceptHandler] = []
    for node in tree.body:
        if isinstance(node, ast.Try):
            handlers.extend(node.handlers)
    return handlers


def test_backend_import_guard_only_catches_module_not_found() -> None:
    tree = ast.parse(_TARGET.read_text(encoding="utf-8"))
    handlers = _module_level_handlers(tree)

    assert handlers, "expected a module-level try/except guarding backend imports"

    for handler in handlers:
        exc_type = handler.type
        assert exc_type is not None, (
            "bare 'except:' would swallow real backend defects and skip the "
            "security suite"
        )
        assert isinstance(
            exc_type, ast.Name
        ), "import guard must catch a single named exception, not a tuple/attr"
        assert exc_type.id == "ModuleNotFoundError", (
            "backend import guard must narrow to ModuleNotFoundError so that a "
            "NameError/SyntaxError/ImportError in the backend fails loudly "
            f"instead of skipping the suite (found 'except {exc_type.id}')"
        )
