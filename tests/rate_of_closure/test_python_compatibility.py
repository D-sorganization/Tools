"""Python-version compatibility contracts for Rate of Closure modules."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
#: Trees whose modules must run on the oldest interpreter in the CI matrix.
#: Discovered rather than enumerated: an allowlist silently ignores whatever a
#: later change adds, which is exactly how eight direct ``enum.StrEnum`` imports
#: reached the 3.10 lane.
SCANNED_TREES = (
    Path("src/rate_of_closure"),
    Path("src/shared/python/swing_sim"),
    Path("scripts"),
)
#: The only module allowed to reference the 3.11 names directly is the shim.
COMPATIBILITY_SHIM = Path("src/shared/python/compatibility.py")


def _scanned_modules() -> list[Path]:
    found: list[Path] = []
    for tree in SCANNED_TREES:
        root = REPOSITORY_ROOT / tree
        if not root.is_dir():
            continue
        found.extend(
            path.relative_to(REPOSITORY_ROOT)
            for path in sorted(root.rglob("*.py"))
            if path.relative_to(REPOSITORY_ROOT) != COMPATIBILITY_SHIM
        )
    return found


def _modules_referencing(symbol: str) -> list[Path]:
    """Return scanned modules that import ``symbol`` at all."""
    hits: list[Path] = []
    for relative_path in _scanned_modules():
        source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        if symbol not in source:
            continue
        tree = ast.parse(source, filename=str(relative_path))
        if any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == symbol for alias in node.names)
            for node in ast.walk(tree)
        ):
            hits.append(relative_path)
    return hits


STRING_ENUM_MODULES = tuple(_modules_referencing("StrEnum"))
UTC_MODULES = tuple(_modules_referencing("UTC"))


WORKSPACE_EXECUTION_TARGETS = (
    (Path("src/rate_of_closure/application/_workspace_validation.py"), ()),
    (Path("src/rate_of_closure/application/commands.py"), ("AppCommandId",)),
    (
        Path("src/rate_of_closure/view_workspace.py"),
        ("ViewKind", "ViewLayout", "LegendPlacement"),
    ),
)


@pytest.mark.parametrize("relative_path", STRING_ENUM_MODULES)
def test_string_enums_use_the_shared_python310_compatibility_contract(
    relative_path: Path,
) -> None:
    """Prevent newer interpreters from masking direct ``enum.StrEnum`` imports."""
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path)
    assert _runtime_import_modules(tree, "StrEnum") == {"shared.python.compatibility"}


@pytest.mark.parametrize("relative_path", UTC_MODULES)
def test_utc_uses_the_shared_python310_compatibility_contract(
    relative_path: Path,
) -> None:
    """Prevent Python 3.11 from masking a direct ``datetime.UTC`` import."""
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path)

    assert _runtime_import_modules(tree, "UTC") == {"shared.python.compatibility"}


def _runtime_import_modules(tree: ast.Module, symbol: str) -> set[str | None]:
    """Return module names that import a symbol on the runtime path."""
    runtime_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "TYPE_CHECKING":
                runtime_nodes.extend(node.orelse)
                continue
        runtime_nodes.append(node)
    return {
        node.module
        for runtime_node in runtime_nodes
        for node in ast.walk(runtime_node)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == symbol for alias in node.names)
    }


@pytest.mark.parametrize(("relative_path", "enum_names"), WORKSPACE_EXECUTION_TARGETS)
def test_workspace_string_enum_modules_execute_through_compatibility_contract(
    relative_path: Path,
    enum_names: tuple[str, ...],
) -> None:
    """Exercise the child modules without package-initializer side effects."""
    module = _load_source_module(REPOSITORY_ROOT / relative_path)

    for enum_name in enum_names:
        enum_type = getattr(module, enum_name)
        member = next(iter(enum_type))
        assert isinstance(member, str)
        assert str(member) == cast(Any, member).value


@pytest.mark.parametrize(
    ("timestamp", "expected_microsecond"),
    [
        ("2026-08-07T12:00:00Z", 0),
        ("2026-08-07T12:00:00.1Z", 100_000),
        ("2026-08-07T12:00:00.12Z", 120_000),
        ("2026-08-07T12:00:00.123Z", 123_000),
        ("2026-08-07T12:00:00.1234Z", 123_400),
        ("2026-08-07T12:00:00.12345Z", 123_450),
        ("2026-08-07T12:00:00.123456Z", 123_456),
    ],
)
def test_workspace_timestamp_precision_is_stable_on_supported_python_versions(
    timestamp: str,
    expected_microsecond: int,
) -> None:
    """Accept every timestamp precision representable without data loss."""
    module = _load_source_module(REPOSITORY_ROOT / UTC_MODULES[0])

    parsed = module.utc_datetime(timestamp, "timestamp")

    assert parsed.microsecond == expected_microsecond
    assert parsed.tzinfo is module.UTC


def test_workspace_timestamp_rejects_precision_beyond_microseconds() -> None:
    """Never silently truncate a persisted timestamp on newer interpreters."""
    module = _load_source_module(REPOSITORY_ROOT / UTC_MODULES[0])

    with pytest.raises(ValueError, match="at most six fractional digits"):
        module.utc_datetime("2026-08-07T12:00:00.1234567Z", "timestamp")


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        ("2026-08-07T12:00:00.Z", "ISO-8601 UTC timestamp"),
        ("2026-08-07T12:00:00", "ending in Z"),
        ("2026-08-07T12:00:00+00:00", "ending in Z"),
        ("2026-8-07T12:00:00Z", "ISO-8601 UTC timestamp"),
        ("2026-08-07 12:00:00Z", "ISO-8601 UTC timestamp"),
    ],
)
def test_workspace_timestamp_rejects_noncanonical_utc_spellings(
    timestamp: str,
    message: str,
) -> None:
    """Keep the persisted UTC grammar strict and interpreter-independent."""
    module = _load_source_module(REPOSITORY_ROOT / UTC_MODULES[0])

    with pytest.raises(ValueError, match=message):
        module.utc_datetime(timestamp, "timestamp")


def _load_source_module(source_path: Path) -> ModuleType:
    """Load one module through its runtime compatibility imports."""
    module_name = f"_rate_of_closure_compat_{source_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load compatibility target: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_the_compatibility_guard_is_not_scanning_an_empty_set() -> None:
    """A discovery-based guard must prove it actually found modules to check.

    Without this, a refactor that moves or renames the scanned trees would make
    every parametrized case above disappear and the suite would still be green.
    """
    assert len(STRING_ENUM_MODULES) >= 10
    assert len(UTC_MODULES) >= 2
    assert COMPATIBILITY_SHIM not in set(STRING_ENUM_MODULES)
