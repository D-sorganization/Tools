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

REPO_ROOT = Path(__file__).resolve().parents[2]
_TARGET = Path(__file__).with_name("test_backend_security.py")


def _module_level_handlers(tree: ast.Module) -> list[ast.ExceptHandler]:
    handlers: list[ast.ExceptHandler] = []
    for node in tree.body:
        if isinstance(node, ast.Try):
            handlers.extend(node.handlers)
    return handlers


def _try_imports_main(try_node: ast.Try) -> bool:
    for child in ast.walk(try_node):
        if isinstance(child, ast.Import):
            if any(alias.name == "main" for alias in child.names):
                return True
        if isinstance(child, ast.ImportFrom) and child.module == "main":
            return True
    return False


def _broad_handlers_for_main_import(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not _try_imports_main(node):
            continue
        for handler in node.handlers:
            if handler.type is None:
                violations.append(f"{path}: bare except around main import")
                continue
            if isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                violations.append(f"{path}: except Exception around main import")
    return violations


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
        assert isinstance(exc_type, ast.Name), (
            "import guard must catch a single named exception, not a tuple/attr"
        )
        assert exc_type.id == "ModuleNotFoundError", (
            "backend import guard must narrow to ModuleNotFoundError so that a "
            "NameError/SyntaxError/ImportError in the backend fails loudly "
            f"instead of skipping the suite (found 'except {exc_type.id}')"
        )


def test_endpoint_import_guards_do_not_use_broad_handlers() -> None:
    """#5924: alarm/alicat endpoint tests must not skip arbitrary import failures."""
    targets = (
        Path(__file__).with_name("test_alicat_manager.py"),
        REPO_ROOT / "src/p1am_control_system/backend/tests/test_alarm_ack_engine.py",
    )
    violations: list[str] = []
    for path in targets:
        assert path.is_file(), path
        violations.extend(_broad_handlers_for_main_import(path))
    assert not violations, "; ".join(violations)
