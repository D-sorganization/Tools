"""Regression guards for Tools issue #3335.

Production package imports must not mutate process-global ``sys.path`` as a
fallback for nested source layouts. Launcher scripts remain responsible for
calling the documented bootstrap when they need source-tree execution.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SHARED_PYTHON = SRC_ROOT / "shared" / "python"
URDF_VIEWER_APP = SRC_ROOT / "web_applications" / "urdf_viewer"

CONTRACT_FILES = [
    SRC_ROOT / "shared" / "python" / "sidekick" / "theme" / "__init__.py",
    SRC_ROOT / "signal_processing_studio" / "__init__.py",
    SRC_ROOT / "urdf_builder_gui" / "__init__.py",
    SRC_ROOT / "web_applications" / "urdf_viewer" / "app.py",
]

IMPORT_CASES = [
    (
        "sidekick.theme",
        (str(SHARED_PYTHON),),
        (
            "sidekick.theme",
            "theme",
        ),
    ),
    (
        "signal_processing_studio",
        (str(SRC_ROOT),),
        ("signal_processing_studio",),
    ),
    (
        "urdf_builder_gui",
        (str(SRC_ROOT),),
        ("urdf_builder_gui",),
    ),
    (
        "app",
        (str(URDF_VIEWER_APP), str(SRC_ROOT), str(SHARED_PYTHON)),
        ("app", "urdf_builder_gui"),
    ),
]


def _evict_modules(prefixes: tuple[str, ...]) -> None:
    dotted_prefixes = tuple(f"{prefix}." for prefix in prefixes)
    for module_name in list(sys.modules):
        if module_name in prefixes or module_name.startswith(dotted_prefixes):
            sys.modules.pop(module_name, None)


@pytest.mark.parametrize("module_name,path_entries,module_prefixes", IMPORT_CASES)
def test_imports_do_not_mutate_global_sys_path(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    path_entries: tuple[str, ...],
    module_prefixes: tuple[str, ...],
) -> None:
    _evict_modules(module_prefixes)
    for path_entry in reversed(path_entries):
        monkeypatch.syspath_prepend(path_entry)
    importlib.invalidate_caches()

    before = tuple(sys.path)
    importlib.import_module(module_name)

    assert tuple(sys.path) == before


@pytest.mark.parametrize("source_path", CONTRACT_FILES)
def test_bootstrap_contract_files_do_not_call_sys_path_insert_or_append(
    source_path: Path,
) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"insert", "append"}:
            continue
        receiver = node.func.value
        if (
            isinstance(receiver, ast.Attribute)
            and receiver.attr == "path"
            and isinstance(receiver.value, ast.Name)
            and receiver.value.id == "sys"
        ):
            violations.append(f"{source_path}:{node.lineno}")

    assert not violations


def test_nested_package_bridges_are_package_scoped_only() -> None:
    signal_init = (SRC_ROOT / "signal_processing_studio" / "__init__.py").read_text(
        encoding="utf-8"
    )
    urdf_init = (SRC_ROOT / "urdf_builder_gui" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "__path__.insert(0, _canonical_str)" in signal_init
    assert "__path__.insert(0, _canonical_str)" in urdf_init
    assert "import sys" not in signal_init
    assert "import sys" not in urdf_init
