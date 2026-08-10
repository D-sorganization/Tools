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
