"""Python-version compatibility contracts for Rate of Closure modules."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
STRING_ENUM_MODULES = (
    Path("src/rate_of_closure/application/commands.py"),
    Path("src/rate_of_closure/simulation/manual_delivery.py"),
    Path("src/rate_of_closure/view_workspace.py"),
    Path("src/shared/python/swing_sim/conventions/registry.py"),
    Path("src/shared/python/swing_sim/flight/capability_contract.py"),
    Path("src/shared/python/swing_sim/flight/impact_solution_contract.py"),
    Path("src/shared/python/swing_sim/flight/inverse_contract.py"),
    Path("src/shared/python/swing_sim/flight/result_contract.py"),
    Path("src/shared/python/swing_sim/impact/dplane.py"),
)
UTC_MODULES = (
    Path("src/rate_of_closure/application/_workspace_validation.py"),
    Path("src/rate_of_closure/ui/pyqt6/torque_profile_controller.py"),
)
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
        assert str(member) == member.value


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
