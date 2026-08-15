"""Python 3.10 compatibility guards for Rate of Closure datetime use."""

from __future__ import annotations

import ast
from pathlib import Path

RATE_SOURCE = Path(__file__).parents[2] / "src" / "rate_of_closure"


def _uses_python_311_datetime_utc(source: str) -> bool:
    """Return whether *source* accesses the Python 3.11-only UTC export."""
    tree = ast.parse(source)
    datetime_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "datetime"
    }
    return any(
        (
            isinstance(node, ast.ImportFrom)
            and node.module == "datetime"
            and any(alias.name == "UTC" for alias in node.names)
        )
        or (
            isinstance(node, ast.Attribute)
            and node.attr == "UTC"
            and isinstance(node.value, ast.Name)
            and node.value.id in datetime_aliases
        )
        for node in ast.walk(tree)
    )


def test_guard_rejects_datetime_module_utc_attribute() -> None:
    """The unaliased module form must not evade the compatibility guard."""
    assert _uses_python_311_datetime_utc("import datetime\nvalue = datetime.UTC\n")


def test_guard_rejects_aliased_datetime_module_utc_attribute() -> None:
    """The aliased module form must not evade the compatibility guard."""
    assert _uses_python_311_datetime_utc("import datetime as dt\nvalue = dt.UTC\n")


def test_rate_sources_do_not_import_datetime_utc_directly() -> None:
    """Python 3.10 lacks ``datetime.UTC``; use the shared compatibility export."""
    violations: list[str] = []
    for path in sorted(RATE_SOURCE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if _uses_python_311_datetime_utc(source):
            violations.append(str(path.relative_to(RATE_SOURCE)))

    assert violations == []


def _imports_strenum_unguarded(source: str) -> bool:
    """Return whether *source* imports ``enum.StrEnum`` at module scope.

    ``enum.StrEnum`` is Python 3.11+. An import nested inside a ``try`` or a
    version check is the accepted compatibility pattern; only a module-scope
    import breaks collection on the 3.10 lane.
    """
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "enum"
        and node.level == 0
        and any(alias.name == "StrEnum" for alias in node.names)
        for node in tree.body
    )


def test_rate_sources_do_not_import_strenum_unguarded() -> None:
    """Python 3.10 lacks ``enum.StrEnum``; use the shared compatibility export.

    Unguarded imports here are not a style question: they raise ImportError at
    module scope, which fails pytest *collection* and aborts the whole 3.10 lane
    before any test runs, so a single one hides every other result.
    """
    violations: list[str] = []
    for path in sorted(RATE_SOURCE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if _imports_strenum_unguarded(source):
            violations.append(str(path.relative_to(RATE_SOURCE)))

    assert violations == []


def test_strenum_guard_accepts_the_compatibility_patterns() -> None:
    """The guard must reject only module-scope imports, not guarded ones."""
    unguarded = "from enum import StrEnum" + chr(10)
    unguarded_pair = "from enum import Enum, StrEnum" + chr(10)
    guarded = chr(10).join(
        (
            "try:",
            "    from enum import StrEnum",
            "except ImportError:",
            "    StrEnum = str",
        )
    )
    shimmed = "from shared.python.compatibility import StrEnum" + chr(10)

    assert _imports_strenum_unguarded(unguarded)
    assert _imports_strenum_unguarded(unguarded_pair)
    assert not _imports_strenum_unguarded(guarded)
    assert not _imports_strenum_unguarded(shimmed)
