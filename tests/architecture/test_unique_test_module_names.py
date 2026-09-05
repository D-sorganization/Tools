"""Every test module must resolve to a unique import name (Tools #4913).

Under pytest's default ``prepend`` import mode a test file's module name is its
path relative to the first ancestor directory *without* an ``__init__.py``.
Two files with the same basename in two such directories collide
(``import file mismatch``), and which one errors depends on collection order.
That is what made four modules fail to collect only in a whole-tree run:
``test_contracts.py`` (tests/unit vs tests/shared/python/golf_club vs
tests/shared/python/sidekick/lab/mocap), ``test_schema.py`` and
``test_import_alias.py``. This guard fails the moment a new collision lands.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SKIP_DIRS = frozenset(
    {
        "replicants",
        "archive",
        "legacy",
        "experimental",
        ".git",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        ".venv",
        "venv",
        "htmlcov",
        ".pytest_cache",
        ".hypothesis",
    }
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _prepend_mode_module_name(test_file: Path) -> str:
    """Reproduce ``_pytest.pathlib.resolve_package_path`` naming."""
    directory = test_file.parent
    parts = [test_file.stem]
    while (directory / "__init__.py").is_file():
        parts.insert(0, directory.name)
        directory = directory.parent
    return ".".join(parts)


def _test_files(base: str) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT / base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py") and (
                name.startswith("test_") or name.endswith("_test.py")
            ):
                found.append(Path(dirpath) / name)
    return found


def test_tests_tree_has_no_colliding_module_names() -> None:
    by_name: dict[str, list[str]] = defaultdict(list)
    for test_file in _test_files("tests"):
        by_name[_prepend_mode_module_name(test_file)].append(
            test_file.relative_to(REPO_ROOT).as_posix()
        )
    collisions = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    assert not collisions, (
        "test modules would shadow each other under pytest's prepend import mode; "
        "add an __init__.py to the directory so the module gets a package-qualified "
        f"name: {collisions}"
    )
